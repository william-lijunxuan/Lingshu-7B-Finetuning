import os
import json
import re
import logging
import traceback
from datetime import datetime
from difflib import SequenceMatcher

import torch
from PIL import Image

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# =========================
# 0) Config
# =========================
DATA_PATH = "/root/dataset/skin/SkinCAP/SkinCAP_20250712_121252_close_end_QA.json"
BASE_IMG_DIR = "/root/dataset/skin/SkinCAP/skincap"

CKPT = "/root/model/Qwen3-VL-4B-Instruct"
OUTPUT_DIR = "/root/model/GRPO_qwen3vl4b"

TRAIN_SIZE = 4
EVAL_SIZE = 2

SYSTEM = "SYSTEM INSTRUCTION: think silently if needed."
USER_TEMPLATE = (
    "You are given a clinical image and a question.\n"
    "Return ONLY the disease name in English. No extra words.\n"
    "Image description: {q}\n"
)


# =========================
# Logging
# =========================
def setup_logging():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"RL_GRPO_qwen3vl_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


log_path = setup_logging()


# =========================
# Dataset
# =========================
def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset():
    raw = load_data()

    def process(ex):
        img_path = os.path.join(BASE_IMG_DIR, ex["image_name"])
        image = Image.open(img_path).convert("RGB")

        q = ex.get("caption_zh_polish_en")
        q = "null" if q is None else str(q)

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": USER_TEMPLATE.format(q=q)},
                ],
            },
        ]

        return {
            "prompt": conversation,
            "answer": str(ex["answer"]),
            "image_name": ex["image_name"]
        }

    ds = Dataset.from_list(raw)
    ds = ds.map(process, remove_columns=ds.column_names)
    return ds


# =========================
# Reward
# =========================
def normalize(s):
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    return re.sub(r"\s+", " ", s)


def reward_fn(prompts, completions, answer, **kwargs):
    rewards = []
    for pred, gt in zip(completions, answer):
        if isinstance(pred, list):
            pred = pred[0]
        pred = normalize(str(pred))
        gt = normalize(str(gt))

        if pred == gt:
            rewards.append(1.0)
        else:
            sim = SequenceMatcher(None, pred, gt).ratio()
            rewards.append(0.5 if sim > 0.9 else 0.0)
    return rewards


# =========================
# Training Config
# =========================
def build_args():
    return GRPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=2,
        learning_rate=5e-6,
        max_completion_length=64,
        max_steps=2000,
        logging_steps=10,
        save_steps=200,
        bf16=True,
        use_vllm=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="tensorboard"
    )


def build_lora():
    return LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )


# =========================
# Main
# =========================
def main():
    ds = build_dataset()

    train_ds = ds.select(range(TRAIN_SIZE))
    eval_ds = ds.select(range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE))

    processor = AutoProcessor.from_pretrained(
        CKPT,
        padding_side="left"  # 必须
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        CKPT,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[reward_fn],
        args=build_args(),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=build_lora(),
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("Fatal error: %s", str(e))
        logging.error(traceback.format_exc())