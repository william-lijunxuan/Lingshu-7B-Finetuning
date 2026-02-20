import os
import sys
import json
import re
import logging
import traceback
from datetime import datetime
from difflib import SequenceMatcher

import torch
import datasets
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

MODEL_TAG = "qwen3vl_4b_instruct"

MAX_Q_CHARS = 800
MAX_A_CHARS = 400


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
# 2) Dataset building
# =========================
SYSTEM = "SYSTEM INSTRUCTION: think silently if needed."
USER_TEMPLATE = (
    "You are given a clinical image and a question.\n"
    "Return ONLY the disease name in English. No extra words.\n"
    "Image description: {q}\n"
)


def load_json_list(path: str):
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("JSON root must be a list of examples.")
    return obj


def add_image_path(ex):
    # Store file path; datasets.Image() will lazy-load as PIL when accessed.
    ex["image"] = os.path.join(BASE_IMG_DIR, ex["image_name"])
    return ex


def make_conversation(ex):
    q = ex.get("caption_zh_polish_en")
    q = "null" if q is None else str(q)

    # IMPORTANT: use ex["image"] AFTER casting to datasets.Image()
    # It will be a PIL.Image.Image when this function runs in map().
    prompt = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ex["image"]},
                {"type": "text", "text": USER_TEMPLATE.format(q=q)},
            ],
        },
    ]

    return {
        "prompt": prompt,
        "answer": str(ex.get("answer", "")),
        "image_name": str(ex.get("image_name", "")),
        "question_type": str(ex.get("question_type", "")),
        # Keep image column (optional). It matches the notebook pattern.
        "image": ex["image"],
    }


def build_dataset():
    data = load_json_list(DATA_PATH)
    ds = Dataset.from_list(data)

    ds = ds.map(add_image_path)

    # Lazy decode image from file path -> PIL when accessed
    ds = ds.cast_column("image", datasets.Image())

    # Now build prompt using ex["image"] (PIL), not a path string
    ds = ds.map(make_conversation, remove_columns=ds.column_names)

    return ds


# =========================
# 3) Reward function + helpers
# =========================
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


ALIAS = {
    "scc": "squamous cell carcinoma",
    "squamous cell ca": "squamous cell carcinoma",
    "squamous cell cancer": "squamous cell carcinoma",
    "bcc": "basal cell carcinoma",
    "basal cell cancer": "basal cell carcinoma",
    "mm": "melanoma",
}


def canonicalize(s: str) -> str:
    s = normalize_disease(s)
    return normalize_disease(ALIAS.get(s, s))


def correctness_reward_func(
    prompts,
    completions,
    answer,
    image_name=None,
    trainer_state=None,
    **kwargs
) -> list[float]:
    rewards = []
    step = getattr(trainer_state, "global_step", None)

    if isinstance(image_name, list):
        names = image_name
    else:
        names = [None] * len(completions)

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
        eval_on_start=False,

        learning_rate=5e-6,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=4,

        max_completion_length=128,

        max_steps=7200,
        logging_steps=20,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=200,

        report_to="tensorboard",

        use_vllm=False,

        bf16=True,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # IMPORTANT: keep columns needed by processor/prompt
        remove_unused_columns=False,

        push_to_hub=False,
    )


def build_lora_config():
    # You can keep all-linear; notebook uses q_proj/v_proj, but all-linear is fine for LoRA if memory allows.
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=64,
        lora_alpha=64,
        target_modules="all-linear",
    )


# =========================
# 5) Model / Processor
# =========================
def load_model_and_processor():
    # IMPORTANT: GRPO expects left padding (same as notebook)
    processor = AutoProcessor.from_pretrained(CKPT, padding_side="left")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        CKPT,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    return model, processor


# =========================
# 6) Main
# =========================
def run():
    logger.info("Loading dataset from: %s", DATA_PATH)
    ds = build_dataset()
    logger.info("Dataset size: %d", len(ds))
    logger.info("Columns: %s", ds.column_names)

    need = TRAIN_SIZE + EVAL_SIZE
    if len(ds) < need:
        raise ValueError(f"Dataset too small: {len(ds)} < {need}")

    train_dataset = ds.select(range(0, TRAIN_SIZE))
    eval_dataset = ds.select(range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE))

    logger.info("Train size: %d | Eval size: %d", len(train_dataset), len(eval_dataset))

    training_args = build_training_args()
    lora_config = build_lora_config()

    logger.info("Loading Qwen3-VL model and processor from: %s", CKPT)
    model, processor = load_model_and_processor()

    logger.info("torch.cuda.is_available=%s", torch.cuda.is_available())
    logger.info("cuda_device_count=%s", torch.cuda.device_count())
    logger.info("model device=%s", str(next(model.parameters()).device))

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[correctness_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
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