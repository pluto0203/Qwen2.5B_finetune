from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

from src.data import DataConfig, load_sft_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 with LoRA for medical QA")
    parser.add_argument("--config", type=str, required=True, help="Path to training YAML config")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, metrics_file: str) -> None:
        self.metrics_file = metrics_file

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        payload = {
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "metrics": logs,
        }
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def configure_file_logging(output_dir: str) -> str:
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    train_log_file = os.path.join(logs_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(train_log_file, encoding="utf-8"),
        ],
        force=True,
    )
    return train_log_file


def resolve_attention_backend(requested: str) -> str:
    if requested != "auto":
        return requested

    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["training"]["seed"])
    set_seed(seed)

    model_name = cfg["model"]["name"]
    output_dir = cfg["training"]["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_log_file = configure_file_logging(output_dir)
    logger = logging.getLogger(__name__)
    logger.info("Loading config from %s", args.config)
    logger.info("Artifacts will be saved under %s", output_dir)
    logger.info("Text log file: %s", train_log_file)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    cfg_bf16 = cfg["training"].get("bf16")
    if cfg_bf16 is not None:
        use_bf16 = bool(cfg_bf16) and torch.cuda.is_available()

    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_fp16 = bool(cfg["training"].get("fp16", not use_bf16))

    if torch.cuda.is_available():
        # Hopper benefits from TF32 for matmul-heavy kernels while keeping numerics stable.
        enable_tf32 = bool(cfg["training"].get("tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32
        torch.set_float32_matmul_precision("high")

    bnb_config = None
    if cfg["model"].get("load_in_4bit", True):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_backend = resolve_attention_backend(cfg["model"].get("attn_implementation", "sdpa"))
    device_map = cfg["model"].get("device_map", "auto")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map=device_map,
        attn_implementation=attn_backend,
    )
    logger.info("Using attention backend: %s", attn_backend)

    if cfg["training"].get("gradient_checkpointing", True):
        model.config.use_cache = False

    data_cfg = DataConfig(
        csv_path=cfg["data"]["csv_path"],
        question_col=cfg["data"].get("question_col", "Question"),
        answer_col=cfg["data"].get("answer_col", "Answer"),
        val_size=float(cfg["data"].get("val_size", 0.02)),
        seed=seed,
    )

    dataset = load_sft_dataset(
        tokenizer=tokenizer,
        config=data_cfg,
        system_prompt=cfg["data"]["system_prompt"],
        limit_rows=cfg["data"].get("limit_rows"),
    )

    lora_cfg = LoraConfig(
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"].get("dropout", 0.05)),
        bias=cfg["lora"].get("bias", "none"),
        task_type="CAUSAL_LM",
        target_modules=cfg["lora"]["target_modules"],
    )

    sft_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": int(cfg["training"]["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(cfg["training"]["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(cfg["training"]["gradient_accumulation_steps"]),
        "num_train_epochs": float(cfg["training"]["num_train_epochs"]),
        "learning_rate": float(cfg["training"]["learning_rate"]),
        "warmup_ratio": float(cfg["training"].get("warmup_ratio", 0.03)),
        "logging_strategy": "steps",
        "logging_steps": int(cfg["training"].get("logging_steps", 10)),
        "logging_first_step": True,
        "logging_dir": os.path.join(output_dir, "logs", "tb"),
        "eval_steps": int(cfg["training"].get("eval_steps", 100)),
        "save_strategy": cfg["training"].get("save_strategy", "steps"),
        "save_steps": int(cfg["training"].get("save_steps", 100)),
        "save_total_limit": int(cfg["training"].get("save_total_limit", 3)),
        "lr_scheduler_type": cfg["training"].get("lr_scheduler_type", "cosine"),
        "max_seq_length": int(cfg["data"].get("max_seq_length", 2048)),
        "packing": bool(cfg["data"].get("packing", False)),
        "bf16": use_bf16,
        "fp16": use_fp16,
        "gradient_checkpointing": bool(cfg["training"].get("gradient_checkpointing", True)),
        "report_to": cfg["training"].get("report_to", "none"),
        "remove_unused_columns": False,
        "dataset_text_field": "text",
    }

    optional_training_keys = [
        "optim",
        "tf32",
        "torch_compile",
        "torch_compile_backend",
        "bf16_full_eval",
        "group_by_length",
        "dataloader_num_workers",
        "dataloader_pin_memory",
        "dataloader_persistent_workers",
        "save_safetensors",
        "weight_decay",
        "max_grad_norm",
    ]
    available_fields = getattr(SFTConfig, "__dataclass_fields__", {})
    for key in optional_training_keys:
        if key in cfg["training"] and key in available_fields:
            sft_kwargs[key] = cfg["training"][key]

    eval_strategy = cfg["training"].get("eval_strategy", "steps")
    if "eval_strategy" in getattr(SFTConfig, "__dataclass_fields__", {}):
        sft_kwargs["eval_strategy"] = eval_strategy
    else:
        sft_kwargs["evaluation_strategy"] = eval_strategy

    train_cfg = SFTConfig(**sft_kwargs)

    metrics_file = os.path.join(output_dir, "logs", "metrics.jsonl")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_cfg,
        args=train_cfg,
    )
    trainer.add_callback(MetricsLoggerCallback(metrics_file=metrics_file))

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = trainer.evaluate()
    metrics_path = os.path.join(output_dir, "eval_metrics.yaml")
    with open(metrics_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics, f, sort_keys=True)

    history_path = os.path.join(output_dir, "logs", "trainer_log_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=True, indent=2)

    logger.info("Saved eval metrics to %s", metrics_path)
    logger.info("Saved step metrics jsonl to %s", metrics_file)
    logger.info("Saved trainer log history to %s", history_path)


if __name__ == "__main__":
    main()
