# Qwen2.5-7B Medical QA Fine-tuning (LoRA)

This project fine-tunes `Qwen/Qwen2.5-7B-Instruct` on a supervised medical QA dataset from CSV (`Question`, `Answer`) using LoRA/QLoRA.

## 1) Expected dataset layout

The current repo already matches this structure:

```text
data/
  train/
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
- `configs/train_qwen25_7b_kaggle_lite.yaml`: base config for Kaggle low-VRAM mode
- `configs/train_qwen25_7b_laptop6gb_qlora.yaml`: recommended QLoRA config for 6GB laptop GPUs
- `notebooks/kaggle_train_qwen25_7b_lite.ipynb`: end-to-end Kaggle notebook for lighter GPUs
- `Dockerfile` + `docker-compose.yml`: GPU-ready containerized run
- `scripts/cloud_vm_setup_ubuntu.sh`: one-time Docker + NVIDIA runtime setup for Ubuntu VM

## 3) Cloud GPU VM requirements

Recommended minimum:
- NVIDIA GPU with >=24GB VRAM (A10/A5000/L4/RTX 4090 or better)
- Ubuntu 22.04+
- NVIDIA driver installed and working (`nvidia-smi`)

Recommended VM specs:
- 8 vCPU+
- 32GB RAM+
- 100GB SSD+

Required network access:
- SSH port `22`
- TensorBoard port `6006` if you want to open dashboard directly from browser

## 4) Full workflow: from connecting to VM to running training

### Step 1: Prepare the cloud VM

When creating the VM, make sure:
- OS is Ubuntu 22.04 or newer
- GPU driver is installed by the cloud image, or can be installed after boot
- Security group / firewall allows inbound SSH on port `22`
- If you want browser access to TensorBoard, allow inbound port `6006`

### Step 2: Connect from your Windows machine

If your provider gives you a private key file, connect from PowerShell with:

```powershell
ssh -i C:\path\to\your-key.pem ubuntu@<VM_PUBLIC_IP>
```

If your provider gives you password access, use:

```powershell
ssh <username>@<VM_PUBLIC_IP>
```

After login, verify GPU visibility:

```bash
nvidia-smi
```

If `nvidia-smi` fails, stop here and fix the driver before touching Docker or training.

### Step 3: Install Git on the VM

```bash
sudo apt-get update
sudo apt-get install -y git
```

### Step 4: Get the project onto the VM

Option A: clone from GitHub:

```bash
git clone https://github.com/pluto0203/Qwen2.5B_finetune.git
cd Qwen2.5B_finetune
```

Option B: if your local copy has unpushed changes, upload the folder manually with `scp`, `rsync`, or VS Code Remote SSH.

### Step 5: Put the dataset in the expected path

The training config currently expects:

```text
data/train/train.csv
```

Create the folder if needed:

```bash
mkdir -p data/train
```

Upload your CSV from Windows to the VM:

```powershell
scp -i C:\path\to\your-key.pem .\data\train\train.csv ubuntu@<VM_PUBLIC_IP>:/home/ubuntu/Qwen2.5B_finetune/data/train/train.csv
```

Verify on the VM:

```bash
ls -lh data/train/train.csv
head -n 3 data/train/train.csv
```

### Step 6: Create your runtime environment file

This repo includes a template file:

```bash
cp .env.example .env
```

Edit `.env` and set your Hugging Face token if the model requires gated access:

```bash
nano .env
```

Example:

```env
HF_TOKEN=your_huggingface_token
CUDA_VISIBLE_DEVICES=0
```

Do not put real secrets in `.env.example`.

### Step 7: Install Docker and NVIDIA container runtime

Run the provided setup script once:

```bash
bash scripts/cloud_vm_setup_ubuntu.sh
```

Then reconnect to your shell session (or log out/in).

After reconnecting, verify Docker and GPU integration:

```bash
docker --version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If this Docker GPU check fails, training in container will also fail.

### Step 8: Review and adjust training config

Main config file:
- `configs/train_qwen25_7b_lora.yaml`

Current default dataset path in config:
- `data/train/train.csv`

Important knobs:
- `data.max_seq_length`: start with `1024` if GPU memory is tight
- `training.per_device_train_batch_size`
- `training.gradient_accumulation_steps`
- `training.num_train_epochs`
- `lora.r`, `lora.alpha`, `lora.dropout`

Safe smoke-test settings for first run:
- `data.limit_rows: 200`
- `data.max_seq_length: 1024`
- `training.num_train_epochs: 0.1`

### Step 9: Build the training container

```bash
docker compose build
```

This may take a while the first time because it installs PyTorch, Transformers, PEFT, TRL, and TensorBoard.

### Step 10: Start training

Run with the default command from `docker-compose.yml`:

```bash
docker compose up
```

Or run the training command explicitly:

```bash
docker compose run --rm qwen-finetune bash scripts/train.sh configs/train_qwen25_7b_lora.yaml
```

For 6GB laptop GPUs (QLoRA), use:

```bash
docker compose run --rm qwen-finetune bash scripts/train.sh configs/train_qwen25_7b_laptop6gb_qlora.yaml
```

### Step 11: Monitor logs during training

Outputs are saved in:
- `outputs/qwen25_7b_medqa_lora`

Training logs are saved in:
- `outputs/qwen25_7b_medqa_lora/logs/train.log`
- `outputs/qwen25_7b_medqa_lora/logs/metrics.jsonl`
- `outputs/qwen25_7b_medqa_lora/logs/trainer_log_history.json`
- `outputs/qwen25_7b_medqa_lora/logs/tb`

Watch text log in real time:

```bash
tail -f outputs/qwen25_7b_medqa_lora/logs/train.log
```

### Step 12: Open TensorBoard

Start TensorBoard inside the container:

```bash
docker compose run --rm --service-ports qwen-finetune \
  tensorboard --logdir /workspace/outputs/qwen25_7b_medqa_lora/logs/tb --bind_all --port 6006
```

Open in browser:

```text
http://<VM_PUBLIC_IP>:6006
```

If port `6006` is not publicly opened, tunnel it over SSH from your Windows machine:

```powershell
ssh -L 6006:localhost:6006 -i C:\path\to\your-key.pem ubuntu@<VM_PUBLIC_IP>
```

Then open:

```text
http://localhost:6006
```

### Step 13: Inspect results after training

Main artifacts:
- Adapter checkpoint directory: `outputs/qwen25_7b_medqa_lora`
- Eval metrics: `outputs/qwen25_7b_medqa_lora/eval_metrics.yaml`
- Logs: `outputs/qwen25_7b_medqa_lora/logs/`

You should at minimum inspect:
- training loss trend
- eval loss trend
- whether loss is decreasing smoothly or diverging
- whether generated answers on a few test prompts look medically sane

### Step 14: Optionally merge LoRA into the base model

```bash
docker compose run --rm qwen-finetune bash scripts/merge.sh \
  Qwen/Qwen2.5-7B-Instruct \
  outputs/qwen25_7b_medqa_lora \
  outputs/qwen25_7b_medqa_merged
```

### Step 15: Copy trained artifacts back to your local machine

From Windows PowerShell:

```powershell
scp -r -i C:\path\to\your-key.pem ubuntu@<VM_PUBLIC_IP>:/home/ubuntu/Qwen2.5B_finetune/outputs .\outputs_from_vm
```

## 5) Quick troubleshooting

### GPU not found inside Docker

Check:
- `nvidia-smi` works on host
- `nvidia-container-toolkit` installed
- Docker restarted after toolkit install
- `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi` works

### CUDA out of memory

Lower these first:
- `data.max_seq_length`
- `training.per_device_train_batch_size`

Then increase:
- `training.gradient_accumulation_steps`

### Model download fails

Check:
- `HF_TOKEN` is valid in `.env`
- VM has internet access
- model permission exists on your Hugging Face account

### Training starts but loss looks wrong

Check:
- CSV column names really are `Question` and `Answer`
- CSV encoding is valid UTF-8
- rows are clean and not mostly empty
- system prompt is not overly restrictive

## 6) Local non-Docker run (optional)

```bash
pip install -r requirements.txt
python -m src.train_lora --config configs/train_qwen25_7b_lora.yaml
```

## 7) Kaggle Lite notebook (T4/P100)

Use [notebooks/kaggle_train_qwen25_7b_lite.ipynb](notebooks/kaggle_train_qwen25_7b_lite.ipynb) when running on lighter Kaggle GPUs.

Cell order:
1. Cell 1: overview
2. Cell 2: GPU check (`nvidia-smi`)
3. Cell 3: clone/update repo + install dependencies
4. Cell 4: generate runtime config with lite profile
5. Cell 5: start training
6. Cell 6: inspect metrics/log outputs

The notebook creates `configs/train_qwen25_7b_kaggle_lite_runtime.yaml` automatically, so your base config remains unchanged.

