#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import transformers
from qwenvl.train.train_qwen import train
from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments

def main():
    cfg = {
        "deepspeed": "/root/model/Lingshu-7B-Finetuning/qwenvl/scripts/zero3.json",
        "model_name_or_path": "/root/model/Lingshu-7B",
        # "dataset_use": "/root/dataset/skin/Derm1M/Derm1M_train.jsonl",
        "dataset_use": "derm1m",
        "data_flatten": True,
        "tune_mm_vision": False,
        "tune_mm_mlp": True,
        "tune_mm_llm": True,
        "bf16": True,
        "output_dir": "./output",
        "num_train_epochs": 0.5,
        # "per_device_train_batch_size": 4,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 8,
        # "gradient_accumulation_steps": 4,
        "gradient_accumulation_steps": 1,
        "max_pixels": 50176,
        "min_pixels": 784,
        "lora_enable" :True,
        "eval_strategy": "no",
        "save_strategy": "steps",
        "save_steps": 1000,
        "save_total_limit": 1,
        "learning_rate": 2e-7,
        "weight_decay": 0.0,
        "warmup_ratio": 0.03,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "cosine",
        "logging_steps": 1,
        "model_max_length": 8192,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 4,
        "run_name": "lingshu-7b-baseline",
        "report_to": "wandb",
    }


    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(cfg)


    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "23456")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")


    import sys
    sys.argv = [sys.argv[0]]  # 清空命令行
    # 把 cfg 还原成命令行形式喂给 train()
    for k, v in cfg.items():
        key = f"--{k.replace('_','-')}"
        if isinstance(v, bool):
            if v: sys.argv += [key]  # True 就加开关，False 不加
        else:
            sys.argv += [key, str(v)]
    # 调用
    from qwenvl.train.train_qwen import train
    train(attn_implementation="flash_attention_2")

if __name__ == "__main__":
    main()
