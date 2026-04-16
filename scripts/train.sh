#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/train_qwen25_7b_laptop6gb_qlora.yaml}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/.cache/huggingface/transformers}"

python -m src.train_lora --config "${CONFIG_PATH}"
