#!/usr/bin/env bash
# Launch flashinfer and TRT-LLM servers with debug dumps enabled then run
# debug_divergence.py to capture tensors.

set -euo pipefail

# Debug env for TRT-LLM MLA backend
export SGLANG_MLA_DEBUG_TRTLLM=1
export SGLANG_MLA_DEBUG_TRTLLM_LAYER_ID=-1      # dump every layer reached
export SGLANG_MLA_DEBUG_TRTLLM_STEPS=10         # number of decode steps to capture
export SGLANG_MLA_DEBUG_TRTLLM_DIR="divergence_debug_verbose/trtllm"

# Debug env for FlashInfer MLA backend
export SGLANG_MLA_DEBUG_FLASHINFER=1
export SGLANG_MLA_DEBUG_FLASHINFER_LAYER_ID=-1  # dump every layer reached
export SGLANG_MLA_DEBUG_FLASHINFER_STEPS=10
export SGLANG_MLA_DEBUG_FLASHINFER_DIR="divergence_debug_verbose/flashinfer"

# Make sure output dirs are clean
rm -rf "${SGLANG_MLA_DEBUG_TRTLLM_DIR}" "${SGLANG_MLA_DEBUG_FLASHINFER_DIR}" || true

# Run the helper that starts both servers, waits, then runs comparer.
python debug_divergence.py "$@"