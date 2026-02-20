import os
import sys
import json
import re
import logging
import traceback
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List

import torch
import datasets
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration


# =========================
# 0) Config
# =========================
DATA_PATH = "/root/dataset/skin/SkinCAP/SkinCAP_20250712_121252_close_end_QA.json"
BASE_IMG_DIR = "/root/dataset/skin/SkinCAP/skincap"

CKPT = "/root/model/Qwen3-VL-4B-Instruct"
OUTPUT_DIR = "/root/model/GRPO_qwen3vl4b"

TRAIN_SIZE = 4
EVAL_SIZE = 2

MODEL_TAG = "qwen3vl_4b_instruct"

MAX_Q_CHARS = 800
MAX_A_CHARS = 400

SYSTEM = "SYSTEM INSTRUCTION: think silently if needed."
USER_TEMPLATE = (
    "You are given a clinical image and a question.\n"
    "Return ONLY the disease name in English. No extra words.\n"
    "Image description: {q}\n"
)


# =========================
# 1) Logging
# =========================
def setup_logging(model_tag: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(os.getcwd(), f"RL_GRPO_{model_tag}_{ts}.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(sh)

    logger = logging.getLogger("grpo")
    logger.info("Log file: %s", log_file)
    return logger, log_file


logger, log_path = setup_logging(MODEL_TAG)


def is_rank0() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


# =========================
# 2) Dataset
# =========================
def load_json_list(path: str):
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("JSON root must be a list of examples.")
    return obj


def add_image_path(ex: Dict[str, Any]) -> Dict[str, Any]:
    ex["image_path"] = os.path.join(BASE_IMG_DIR, str(ex.get("image_name", "")))
    return ex


def build_dataset() -> Dataset:
    data = load_json_list(DATA_PATH)
    ds = Dataset.from_list(data)

    ds = ds.map(add_image_path)

    # Keep image as datasets.Image so it can be decoded lazily at __getitem__ time
    ds = ds.cast_column("image_path", datasets.Image())
    ds = ds.rename_column("image_path", "image")

    # Do NOT decode images inside map (slow). Create prompt on-the-fly with set_transform.
    def transform(example: Dict[str, Any]) -> Dict[str, Any]:
        cap = example.get("caption_zh_polish_en")
        cap = "null" if cap is None else str(cap)

        prompt = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},  # PIL image here
                    {"type": "text", "text": USER_TEMPLATE.format(q=cap)},
                ],
            },
        ]

        return {
            "prompt": prompt,
            "image": example["image"],
            "answer": str(example.get("answer", "")),
            "image_name": str(example.get("image_name", "")),
            "question_type": str(example.get("question_type", "")),
        }

    ds.set_transform(transform)
    return ds


# =========================
# 3) Reward helpers
# =========================
ALIAS = {
    "scc": "squamous cell carcinoma",
    "squamous cell ca": "squamous cell carcinoma",
    "squamous cell cancer": "squamous cell carcinoma",
    "bcc": "basal cell carcinoma",
    "basal cell cancer": "basal cell carcinoma",
    "mm": "melanoma",
}


def extract_user_text(prompt_item) -> str:
    if not isinstance(prompt_item, list):
        return ""
    user_msgs = [m for m in prompt_item if isinstance(m, dict) and m.get("role") == "user"]
    if not user_msgs:
        return ""
    content = user_msgs[-1].get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        return "\n".join([t for t in texts if t])
    return ""


def extract_completion_text(comp) -> str:
    if isinstance(comp, str):
        return comp
    if isinstance(comp, dict):
        return comp.get("content") or comp.get("text") or ""
    if isinstance(comp, list) and comp:
        if len(comp) == 1 and isinstance(comp[0], list):
            comp = comp[0]
        if comp and isinstance(comp[0], dict):
            return comp[0].get("content") or comp[0].get("text") or ""
        if comp and isinstance(comp[0], str):
            return comp[0]
    return ""


def normalize_disease(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.strip('"').strip("'")
    s = re.sub(r"^\s*(final\s*answer\s*:?\s*)", "", s)
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonicalize(s: str) -> str:
    s = normalize_disease(s)
    return normalize_disease(ALIAS.get(s, s))


def format_reward(completions, **kwargs):
    rewards = []
    for c in completions:
        text = extract_completion_text(c).strip()
        if not text:
            rewards.append(0.0)
            continue
        if "\n" in text or "{" in text or "}" in text or len(text) > 80:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards


def correctness_reward(
    prompts,
    completions,
    answer,
    image_name=None,
    trainer_state=None,
    **kwargs
) -> List[float]:
    rewards = []
    step = getattr(trainer_state, "global_step", None)

    names = image_name if isinstance(image_name, list) else [None] * len(completions)

    for i, (p, c, gt, nm) in enumerate(zip(prompts, completions, answer, names)):
        q_text = extract_user_text(p)
        pred_raw = extract_completion_text(c)

        pred = canonicalize(pred_raw)
        gt_norm = canonicalize(str(gt))

        if not pred:
            r = 0.0
        elif pred == gt_norm:
            r = 1.0
        elif gt_norm in pred or pred in gt_norm:
            r = 0.5
        else:
            sim = SequenceMatcher(None, pred, gt_norm).ratio()
            r = 0.5 if sim >= 0.92 else 0.0

        rewards.append(float(r))

        if is_rank0():
            q_show = (q_text[:MAX_Q_CHARS] + "...") if len(q_text) > MAX_Q_CHARS else q_text
            pred_show = (pred_raw[:MAX_A_CHARS] + "...") if len(pred_raw) > MAX_A_CHARS else pred_raw
            logger.info(
                "step=%s | idx=%d | image=%s | reward=%.3f | gt='%s' | pred_raw='%s' | q='%s'",
                str(step),
                i,
                str(nm),
                r,
                gt_norm,
                pred_show.replace("\n", "\\n"),
                q_show.replace("\n", "\\n"),
            )

    return rewards


# =========================
# 4) Train config
# =========================
def build_training_args():
    return GRPOConfig(
        output_dir=OUTPUT_DIR,

        learning_rate=2e-5,
        max_steps=2000,

        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=64,

        fp16=True,

        logging_steps=10,
        save_steps=200,

        eval_strategy="steps",
        eval_steps=200,

        report_to="tensorboard",
        log_completions=True,

        use_vllm=False,
        remove_unused_columns=False,

        push_to_hub=False,
    )


def build_lora_config():
    return LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )


# =========================
# 5) Load model / processor
# =========================
def load_model_and_processor():
    processor = AutoProcessor.from_pretrained(CKPT, padding_side="left")

    use_4bit = False
    try:
        import bitsandbytes  # noqa: F401
        use_4bit = True
    except Exception:
        logger.warning("bitsandbytes not found; using bf16 without 4-bit quantization.")

    if use_4bit:
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            CKPT,
            device_map="auto",
            quantization_config=qconfig,
            torch_dtype=torch.float32,
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            CKPT,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )

    return model, processor


# =========================
# 6) Main
# =========================
def run():
    logger.info("Loading model + processor from: %s", CKPT)
    model, processor = load_model_and_processor()

    logger.info("torch.cuda.is_available=%s", torch.cuda.is_available())
    logger.info("cuda_device_count=%s", torch.cuda.device_count())
    try:
        logger.info("model device=%s", str(next(model.parameters()).device))
    except Exception:
        logger.info("model device=unknown (quantized / sharded)")

    logger.info("Loading dataset from: %s", DATA_PATH)
    ds = build_dataset()

    need = TRAIN_SIZE + EVAL_SIZE
    if len(ds) < need:
        raise ValueError(f"Dataset too small: {len(ds)} < {need}")

    train_dataset = ds.select(range(0, TRAIN_SIZE))
    eval_dataset = ds.select(range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE))

    logger.info("Train size: %d | Eval size: %d", len(train_dataset), len(eval_dataset))

    training_args = build_training_args()
    peft_config = build_lora_config()

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[format_reward, correctness_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving model to: %s", training_args.output_dir)
    trainer.save_model(output_dir=training_args.output_dir)
    logger.info("Done. Log file: %s", log_path)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        if is_rank0():
            logger.warning("Interrupted by user (KeyboardInterrupt). Log file: %s", log_path)
        raise
    except Exception as e:
        if is_rank0():
            logger.error("Fatal error occurred. Log file: %s", log_path)
            logger.error("Exception: %s", repr(e))
            logger.error("Traceback:\n%s", traceback.format_exc())
        raise
    finally:
        logging.shutdown()