#!/usr/bin/env bash
set -euo pipefail

# Run Solar processing + einsum pipeline for GQA ragged prefill example.
#
# This example uses SOL-ExecBench problem:
#   FlashInfer-Bench/016_gqa_ragged_prefill_causal_h32_kv4_d128
# with workload UUID:
#   007ddabb-3c8c-48a1-a693-c0618d32243c

ARCH="${1:-${SOLAR_ARCH:-H100_PCIe}}"
PRECISION="${2:-${SOLAR_PRECISION:-bf16}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_FILE="${SCRIPT_DIR}/GQARaggedPrefill.py"
OUT_BASE="${SOLAR_GQA_RAGGED_PREFILL_OUTPUT_DIR:-${SCRIPT_DIR}/output}"
GRAPH_OUT="${OUT_BASE}/graph"
EINSUM_OUT="${OUT_BASE}/einsum"
ANALYSIS_OUT="${OUT_BASE}/analysis"
PERF_OUT="${OUT_BASE}/perf"
TIMELOOP_OUT="${OUT_BASE}/timeloop"

if ! mkdir -p "${GRAPH_OUT}" "${EINSUM_OUT}" "${ANALYSIS_OUT}" "${PERF_OUT}" "${TIMELOOP_OUT}"; then
  echo "❌ Failed to create output directories under: ${OUT_BASE}" >&2
  exit 1
fi

cd "${SOLAR_ROOT}"

echo "==> Processing model -> ${GRAPH_OUT}"
python3 -m solar.cli.process_model \
  --model-file "${MODEL_FILE}" \
  --output-dir "${GRAPH_OUT}" \
  --save-graph \
  --force-rerun

echo "==> Converting pytorch graph -> ${EINSUM_OUT}"
python3 -m solar.cli.toeinsum_model \
  --graph-path "${GRAPH_OUT}/pytorch_graph.yaml" \
  --output-dir "${EINSUM_OUT}" \
  --no-copy-graph \
  --save-graph

echo "==> Analyzing einsum graph -> ${ANALYSIS_OUT}"
python3 -m solar.cli.analyze_model \
  --einsum-graph-path "${EINSUM_OUT}/einsum_graph_renamed.yaml" \
  --output-dir "${ANALYSIS_OUT}"

echo "==> Predicting perf (arch=${ARCH}, precision=${PRECISION}) -> ${PERF_OUT}"
python3 -m solar.cli.predict_perf_model \
  --analysis-path "${ANALYSIS_OUT}/analysis.yaml" \
  --output-dir "${PERF_OUT}" \
  --arch-config "${ARCH}" \
  --precision "${PRECISION}"

echo "==> Converting to Timeloop format -> ${TIMELOOP_OUT}"
python3 -m solar.cli.totimeloop \
  --einsum-graph-path "${EINSUM_OUT}/einsum_graph_renamed.yaml" \
  --output-dir "${TIMELOOP_OUT}"

echo ""
echo "Done."
echo ""
echo "=== GQA Ragged Prefill Example Outputs ==="
echo "PyTorch graph:   ${GRAPH_OUT}/pytorch_graph.yaml"
echo "Einsum graph:    ${EINSUM_OUT}/einsum_graph.yaml"
echo "Einsum renamed:  ${EINSUM_OUT}/einsum_graph_renamed.yaml"
echo "Graph PDF:       ${EINSUM_OUT}/einsum_graph.pdf"
echo "Analysis:        ${ANALYSIS_OUT}/analysis.yaml"
echo "Perf:            ${PERF_OUT}/perf_${ARCH}.yaml"
echo "Timeloop graph:  ${TIMELOOP_OUT}/timeloop_graph.yaml"
