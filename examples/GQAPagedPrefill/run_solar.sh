#!/usr/bin/env bash
set -euo pipefail

# Run Solar processing + einsum pipeline for GQA paged prefill example.
#
# This example uses SOL-ExecBench problem:
#   FlashInfer-Bench/014_gqa_paged_prefill_causal_h32_kv4_d128_ps1
# with workload UUID:
#   9fb16120-0ab1-4769-95c9-1961e27b546a

ARCH="${1:-${SOLAR_ARCH:-H100_PCIe}}"
PRECISION="${2:-${SOLAR_PRECISION:-bf16}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_FILE="${SCRIPT_DIR}/GQAPagedPrefill.py"
OUT_BASE="${SOLAR_GQA_PAGED_PREFILL_OUTPUT_DIR:-${SCRIPT_DIR}/output}"
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
echo "=== GQA Paged Prefill Example Outputs ==="
echo "PyTorch graph:   ${GRAPH_OUT}/pytorch_graph.yaml"
echo "Einsum graph:    ${EINSUM_OUT}/einsum_graph.yaml"
echo "Einsum renamed:  ${EINSUM_OUT}/einsum_graph_renamed.yaml"
echo "Graph PDF:       ${EINSUM_OUT}/einsum_graph.pdf"
echo "Analysis:        ${ANALYSIS_OUT}/analysis.yaml"
echo "Perf:            ${PERF_OUT}/perf_${ARCH}.yaml"
echo "Timeloop graph:  ${TIMELOOP_OUT}/timeloop_graph.yaml"
