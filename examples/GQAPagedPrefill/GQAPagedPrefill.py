"""Standalone GQA paged prefill (causal) example for Solar.

This example uses the SOL-ExecBench benchmark definition:
  014_gqa_paged_prefill_causal_h32_kv4_d128_ps1

Inputs are loaded from workload UUID:
  9fb16120-0ab1-4769-95c9-1961e27b546a
"""

import json
import math
import os
import uuid
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file


# WORKLOAD_UUID = "9fb16120-0ab1-4769-95c9-1961e27b546a"
# WORKLOAD_UUID = "c3c5535c-1829-4618-b629-129c0190dfc4"
WORKLOAD_UUID = "a56d392f-5368-45ba-b75b-7fcd2f773a01"

PROBLEM_REL_PATH = "FlashInfer-Bench/014_gqa_paged_prefill_causal_h32_kv4_d128_ps1"


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
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        sm_scale: torch.Tensor,
    ):
        total_q, num_qo_heads, head_dim = q.shape
        num_pages, page_size, num_kv_heads, _ = k_cache.shape
        len_indptr = qo_indptr.shape[0]
        num_kv_indices = kv_indices.shape[0]

        assert num_qo_heads == 32
        assert num_kv_heads == 4
        assert head_dim == 128
        assert page_size == 1

        assert total_q == int(qo_indptr[-1].item())
        _ = num_kv_indices

        device = q.device

        output = torch.zeros(
            (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
        )
        lse = torch.full(
            (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
        )

        gqa_ratio = num_qo_heads // num_kv_heads

        q_f32 = q.to(torch.float32)
        # Flatten page dimension since page_size=1
        k_cache_f32 = k_cache.squeeze(1).to(torch.float32)
        v_cache_f32 = v_cache.squeeze(1).to(torch.float32)

        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())

            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            page_ids = kv_indices[kv_start:kv_end].to(torch.long)

            # Number of KV tokens is equal to number of pages for page_size=1
            k_batch = k_cache_f32[page_ids]
            v_batch = v_cache_f32[page_ids]
            num_kv_tokens = k_batch.shape[0]

            # Get queries for this sequence
            q_batch = q_f32[q_start:q_end]
            num_q_tokens = q_batch.shape[0]

            delta = num_kv_tokens - num_q_tokens

            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx

                max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)
                if max_kv_idx <= 0:
                    continue

                q_pos = q_batch[q_idx]

                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio

                    q_head = q_pos[h]
                    k_head = k_batch[:max_kv_idx, kv_head]
                    v_head = v_batch[:max_kv_idx, kv_head]

                    logits = torch.matmul(q_head, k_head.T)
                    logits_scaled = logits * sm_scale

                    lse[global_q_idx, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

                    attn = torch.softmax(logits_scaled, dim=-1)
                    out_head = torch.matmul(attn, v_head)
                    output[global_q_idx, h] = out_head.to(torch.bfloat16)

        return output, lse


def get_inputs():
    benchmark_dir = Path(os.getenv("SOL_EXECBENCH_DATA", _default_benchmark_dir()))
    problem_dir = benchmark_dir / PROBLEM_REL_PATH
    workload_file = problem_dir / "workload.jsonl"

    entry = _load_workload_entry(workload_file, WORKLOAD_UUID)
    axes = entry["axes"]
    inputs = entry["inputs"]

    total_q = int(axes["total_q"])
    num_pages = int(axes["num_pages"])
    num_qo_heads = 32
    num_kv_heads = 4
    head_dim = 128
    page_size = 1

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_make_rng_seed(WORKLOAD_UUID))

    q = torch.randn(
        (total_q, num_qo_heads, head_dim), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    k_cache = torch.randn(
        (num_pages, page_size, num_kv_heads, head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    v_cache = torch.randn(
        (num_pages, page_size, num_kv_heads, head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    qo_info = inputs["qo_indptr"]
    kv_info = inputs["kv_indptr"]
    indices_info = inputs["kv_indices"]

    qo_path = _resolve_input_path(qo_info["path"], benchmark_dir)
    kv_path = _resolve_input_path(kv_info["path"], benchmark_dir)
    idx_path = _resolve_input_path(indices_info["path"], benchmark_dir)

    qo_indptr = load_file(str(qo_path))[qo_info["tensor_key"]].to(torch.int32)
    kv_indptr = load_file(str(kv_path))[kv_info["tensor_key"]].to(torch.int32)
    kv_indices = load_file(str(idx_path))[indices_info["tensor_key"]].to(torch.int32)

    sm_scale = torch.tensor(float(inputs["sm_scale"]["value"]), dtype=torch.float32)

    return [q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]


if __name__ == "__main__":
    model = Model()
    q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale = get_inputs()
    output, lse = model(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)
    print(f"q shape: {tuple(q.shape)}, dtype={q.dtype}")
    print(f"k_cache shape: {tuple(k_cache.shape)}, dtype={k_cache.dtype}")
    print(f"v_cache shape: {tuple(v_cache.shape)}, dtype={v_cache.dtype}")
    print(f"qo_indptr len: {qo_indptr.numel()}, kv_indptr len: {kv_indptr.numel()}")
    print(f"kv_indices len: {kv_indices.numel()}")
    print(f"output shape: {tuple(output.shape)}, dtype={output.dtype}")
    print(f"lse shape: {tuple(lse.shape)}, dtype={lse.dtype}")

    total_macs = 0
    _, num_qo_heads, head_dim = q.shape
    page_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    mac_terms = 0

    for b in range(qo_indptr.shape[0] - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())

        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())

        if q_start >= q_end or kv_start >= kv_end:
            continue

        num_q_tokens = q_end - q_start
        num_kv_tokens = kv_end - kv_start  # page_size=1, one token per index
        delta = num_kv_tokens - num_q_tokens

        for q_idx in range(num_q_tokens):
            max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)
            if max_kv_idx <= 0:
                continue
            mac_terms += max_kv_idx

    total_macs = mac_terms * num_qo_heads * head_dim * 2
    print(f"total MACs: {total_macs}")

    unique_pages = torch.unique(kv_indices).numel()
    kv_cache_touched_elems = unique_pages * page_size * num_kv_heads * head_dim

    total_bytes = (
        q.numel() * q.element_size()
        + (kv_cache_touched_elems * 2) * k_cache.element_size()  # k_cache + v_cache
        + (qo_indptr.numel() + kv_indptr.numel() + kv_indices.numel()) * qo_indptr.element_size()
        + output.numel() * output.element_size()
        + lse.numel() * lse.element_size()
    )
    print(f"total bytes: {total_bytes}")
