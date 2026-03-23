#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
ADAPTER_PATH="${2:-outputs/qwen25_7b_medqa_lora}"
OUTPUT_DIR="${3:-outputs/qwen25_7b_medqa_merged}"

python -m src.merge_lora \
  --base-model "${BASE_MODEL}" \
  --adapter-path "${ADAPTER_PATH}" \
  --output-dir "${OUTPUT_DIR}"
