# Qwen2.5-7B Medical QA Fine-tuning (LoRA)

This project fine-tunes `Qwen/Qwen2.5-7B-Instruct` on a supervised medical QA dataset from CSV (`Question`, `Answer`) using LoRA/QLoRA.

## 1) Expected dataset layout

The current repo already matches this structure:

```text
data/
  train.csv/
    train.csv
```

CSV required columns:
- `Question`
- `Answer`

## 2) What is included

- `src/train_lora.py`: training entrypoint (SFT + LoRA)
- `src/data.py`: CSV to chat-format dataset conversion
- `src/merge_lora.py`: optional adapter merge into base model
- `configs/train_qwen25_7b_lora.yaml`: training config
- `Dockerfile` + `docker-compose.yml`: GPU-ready containerized run
- `scripts/cloud_vm_setup_ubuntu.sh`: one-time Docker + NVIDIA runtime setup for Ubuntu VM

## 3) Cloud GPU VM requirements

Recommended minimum:
- NVIDIA GPU with >=24GB VRAM (A10/A5000/L4/RTX 4090 or better)
- Ubuntu 22.04+
- NVIDIA driver installed and working (`nvidia-smi`)

One-time VM setup (Ubuntu):

```bash
bash scripts/cloud_vm_setup_ubuntu.sh
```

Then reconnect to your shell session (or log out/in).

## 4) Configure training

Main config file:
- `configs/train_qwen25_7b_lora.yaml`

Important knobs:
- `data.csv_path`: path to your train CSV
- `data.max_seq_length`: start with 1024 or 2048 depending on GPU memory
- `training.per_device_train_batch_size`
- `training.gradient_accumulation_steps`
- `lora.r`, `lora.alpha`, `lora.dropout`

For a quick smoke test, set:
- `data.limit_rows: 200`
- `training.num_train_epochs: 0.1`

## 5) Run with Docker (recommended on cloud VM)

Option A: use docker compose default command:

```bash
docker compose up --build
```

Option B: run explicit command:

```bash
docker compose run --rm qwen-finetune bash scripts/train.sh configs/train_qwen25_7b_lora.yaml
```

Optional if model access requires token:

```bash
export HF_TOKEN=your_huggingface_token
```

Outputs are saved in:
- `outputs/qwen25_7b_medqa_lora`

Training logs are saved in:
- `outputs/qwen25_7b_medqa_lora/logs/train.log` (full text log)
- `outputs/qwen25_7b_medqa_lora/logs/metrics.jsonl` (step-by-step metrics)
- `outputs/qwen25_7b_medqa_lora/logs/trainer_log_history.json` (full trainer history)
- `outputs/qwen25_7b_medqa_lora/logs/tb` (TensorBoard events)

Watch logs in real time:

```bash
tail -f outputs/qwen25_7b_medqa_lora/logs/train.log
```

Run TensorBoard:

```bash
docker compose run --rm --service-ports qwen-finetune \
  tensorboard --logdir /workspace/outputs/qwen25_7b_medqa_lora/logs/tb --bind_all --port 6006
```

Then open `http://<VM_PUBLIC_IP>:6006`.

## 6) Optional: merge LoRA into full model

```bash
docker compose run --rm qwen-finetune bash scripts/merge.sh \
  Qwen/Qwen2.5-7B-Instruct \
  outputs/qwen25_7b_medqa_lora \
  outputs/qwen25_7b_medqa_merged
```

## 7) Notes for stability and memory

- If OOM occurs:
  - Reduce `max_seq_length`
  - Reduce `per_device_train_batch_size`
  - Increase `gradient_accumulation_steps`
- Keep `load_in_4bit: true` for QLoRA memory efficiency.
- Start with `attn_implementation: sdpa`. If your stack supports flash attention, you can switch to `flash_attention_2`.

## 8) Local non-Docker run (optional)

```bash
pip install -r requirements.txt
python -m src.train_lora --config configs/train_qwen25_7b_lora.yaml
```

