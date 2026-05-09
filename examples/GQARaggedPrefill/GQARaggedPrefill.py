"""Standalone GQA ragged prefill (causal) example for Solar.

This example uses the SOL-ExecBench benchmark definition:
  016_gqa_ragged_prefill_causal_h32_kv4_d128

Inputs are loaded from workload UUID:
  007ddabb-3c8c-48a1-a693-c0618d32243c
"""

import json
import math
import os
import uuid
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file


WORKLOAD_UUID = "007ddabb-3c8c-48a1-a693-c0618d32243c"
# WORKLOAD_UUID = "ebf7188b-4b31-4746-b57b-fa25b53f5e3e"
# WORKLOAD_UUID = "1f97930e-1a06-4e84-9875-09a22fff8a7c"
PROBLEM_REL_PATH = (
    "FlashInfer-Bench/016_gqa_ragged_prefill_causal_h32_kv4_d128"
)


def _default_benchmark_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "SOL-ExecBench" / "data" / "benchmark"


def _load_workload_entry(workload_file: Path, target_uuid: str) -> dict:
    with workload_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("uuid") == target_uuid:
                return row
    raise ValueError(f"workload uuid not found: {target_uuid}")


def _resolve_input_path(raw_path: str, benchmark_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute() and path.exists():
        return path

    candidates = [
        benchmark_dir / path,
        benchmark_dir.parent / path,
        benchmark_dir.parent.parent / path,
        Path(__file__).resolve().parents[3] / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"unable to resolve safetensors path: {raw_path}")


def _make_rng_seed(workload_uuid: str) -> int:
    return uuid.UUID(workload_uuid).int & 0xFFFFFFFF


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        sm_scale: torch.Tensor,
    ):
        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        len_indptr = qo_indptr.shape[0]

        assert num_qo_heads == 32
        assert num_kv_heads == 4
        assert head_dim == 128

        assert total_q == int(qo_indptr[-1].item())
        assert total_kv == int(kv_indptr[-1].item())

        device = q.device
        scale = float(sm_scale.item()) if isinstance(sm_scale, torch.Tensor) else float(sm_scale)

        output = torch.zeros(
            (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
        )
        lse = torch.full(
            (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
        )

        gqa_ratio = num_qo_heads // num_kv_heads

        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        total_macs = 0

        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())

            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            q_batch = q_f32[q_start:q_end]
            k_batch = k_f32[kv_start:kv_end]
            v_batch = v_f32[kv_start:kv_end]

            num_q_tokens = q_batch.shape[0]
            num_kv_tokens = k_batch.shape[0]
            delta = num_kv_tokens - num_q_tokens

            k_expanded = k_batch.repeat_interleave(gqa_ratio, dim=1)
            v_expanded = v_batch.repeat_interleave(gqa_ratio, dim=1)

            logits = torch.einsum("qhd,khd->qhk", q_batch, k_expanded) * scale

            q_positions = torch.arange(num_q_tokens, device=device)
            kv_positions = torch.arange(num_kv_tokens, device=device)
            causal_mask = kv_positions[None, :] < (q_positions[:, None] + 1 + delta)
            logits = logits.masked_fill(~causal_mask[:, None, :], float("-inf"))

            lse_batch = torch.logsumexp(logits, dim=-1) / math.log(2.0)
            lse[q_start:q_end] = lse_batch

            attn_weights = torch.softmax(logits, dim=-1)
            output_batch = torch.einsum("qhk,khd->qhd", attn_weights, v_expanded)
            output[q_start:q_end] = output_batch.to(torch.bfloat16)

        return output, lse


def get_inputs():
    benchmark_dir = Path(os.getenv("SOL_EXECBENCH_DATA", _default_benchmark_dir()))
    problem_dir = benchmark_dir / PROBLEM_REL_PATH
    workload_file = problem_dir / "workload.jsonl"

    entry = _load_workload_entry(workload_file, WORKLOAD_UUID)
    axes = entry["axes"]
    inputs = entry["inputs"]

    total_q = int(axes["total_q"])
    total_kv = int(axes["total_kv"])
    num_qo_heads = 32
    num_kv_heads = 4
    head_dim = 128

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_make_rng_seed(WORKLOAD_UUID))

    q = torch.randn(
        (total_q, num_qo_heads, head_dim), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    k = torch.randn(
        (total_kv, num_kv_heads, head_dim), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    v = torch.randn(
        (total_kv, num_kv_heads, head_dim), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)

    qo_info = inputs["qo_indptr"]
    kv_info = inputs["kv_indptr"]

    qo_path = _resolve_input_path(qo_info["path"], benchmark_dir)
    kv_path = _resolve_input_path(kv_info["path"], benchmark_dir)

    qo_indptr = load_file(str(qo_path))[qo_info["tensor_key"]].to(torch.int32)
    kv_indptr = load_file(str(kv_path))[kv_info["tensor_key"]].to(torch.int32)

    sm_scale = torch.tensor(
        float(inputs["sm_scale"]["value"]), dtype=torch.float32
    )

    return [q, k, v, qo_indptr, kv_indptr, sm_scale]


if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    model = Model()
    q, k, v, qo_indptr, kv_indptr, sm_scale = get_inputs()
    output, lse = model(q, k, v, qo_indptr, kv_indptr, sm_scale)
    print(f"q shape: {tuple(q.shape)}, dtype={q.dtype}")
    print(f"k shape: {tuple(k.shape)}, dtype={k.dtype}")
    print(f"qo_indptr len: {qo_indptr.numel()}")
    print(f"output shape: {tuple(output.shape)}, dtype={output.dtype}")
    print(f"lse shape: {tuple(lse.shape)}, dtype={lse.dtype}")

    total_macs = 0
    _, num_qo_heads, head_dim = q.shape
    for b in range(qo_indptr.shape[0] - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())

        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())

        if q_start >= q_end or kv_start >= kv_end:
            continue

        num_q_tokens = q_end - q_start
        num_kv_tokens = kv_end - kv_start
        total_macs += num_q_tokens * num_kv_tokens * num_qo_heads * head_dim * 2  # for matmul and attn output

    print(f"total MACs: {total_macs}")
    total_bytes = (
        (q.numel() + k.numel() + v.numel()) * q.element_size() + 
        (qo_indptr.numel() + kv_indptr.numel()) * qo_indptr.element_size() +
        output.numel() * output.element_size() +
        lse.numel() * lse.element_size()
    )
    print(f"total bytes: {total_bytes}")