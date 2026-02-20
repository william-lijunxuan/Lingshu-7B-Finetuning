import os
import sys
import json
import re
import logging
import traceback
from datetime import datetime
from difflib import SequenceMatcher

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    _HAS_BNB = False


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
# 2) Dataset helpers
# =========================
def load_json_list(path: str):
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("JSON root must be a list of examples.")
    return obj


def pick_caption(ex: dict) -> str:
    for k in ["caption_zh_polish_en", "caption_en", "caption", "question", "query", "text"]:
        if k in ex and ex[k] is not None:
            return str(ex[k])
    return "null"


def make_conversation(ex: dict) -> dict:
    image_name = str(ex.get("image_name", "")).strip()
    image_path = os.path.join(BASE_IMG_DIR, image_name)
    q = pick_caption(ex)
    q = q[:MAX_Q_CHARS]

    prompt = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": USER_TEMPLATE.format(q=q)},
            ],
        },
    ]

    return {
        "prompt": prompt,
        "answer": str(ex.get("answer", "")),
        "image_name": image_name,
        "question_type": str(ex.get("question_type", "")),
    }


def count_images_in_prompt(prompt):
    if not isinstance(prompt, list):
        return 0
    n = 0
    for msg in prompt:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    n += 1
    return n


def safe_apply_chat_template(processor, prompt, image_name_for_log=""):
    """
    Returns (ok: bool, err: str)
    """
    try:
        _ = processor.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        return True, ""
    except Exception as e:
        if is_rank0():
            logger.error("apply_chat_template failed for image=%s | err=%r", image_name_for_log, e)
        return False, repr(e)


def build_dataset(processor):
    data = load_json_list(DATA_PATH)

    need = TRAIN_SIZE + EVAL_SIZE
    if len(data) < need:
        raise ValueError(f"Dataset too small: {len(data)} < {need}")

    data = data[:need]
    ds = Dataset.from_list(data)
    ds = ds.map(make_conversation, remove_columns=ds.column_names)

    # Validate prompts BEFORE training, drop any sample that causes the mismatch/index error.
    good_rows = []
    bad_rows = []

    for i in range(len(ds)):
        row = ds[i]
        prompt = row["prompt"]
        img_count = count_images_in_prompt(prompt)

        ok, err = safe_apply_chat_template(processor, prompt, row.get("image_name", ""))
        if ok:
            good_rows.append(row)
        else:
            bad_rows.append((row.get("image_name", ""), img_count, err))

    if is_rank0() and bad_rows:
        logger.error("Found %d bad samples (dropped). Details:", len(bad_rows))
        for (nm, imgc, err) in bad_rows:
            logger.error("bad_sample image=%s | image_items_in_prompt=%d | err=%s", nm, imgc, err)

    if len(good_rows) < need:
        raise RuntimeError(
            f"After filtering, only {len(good_rows)} usable samples left; need {need}. "
            f"Fix the offending samples listed in the log, or increase the slice size."
        )

    return Dataset.from_list(good_rows)


# =========================
# 3) Rewards
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
            continue
        rewards.append(1.0)
    return rewards


def correctness_reward(prompts, completions, answer, image_name=None, trainer_state=None, **kwargs):
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
        learning_rate=2e-5,
        max_steps=100,
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=64,
        fp16=True,
        output_dir=OUTPUT_DIR,
        logging_steps=1,
        report_to="trackio",
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

    if _HAS_BNB:
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            CKPT,
            dtype="float32",
            device_map="auto",
            quantization_config=qconfig,
        )
    else:
        logger.warning("bitsandbytes not available; loading fp16 without 4-bit quantization.")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            CKPT,
            torch_dtype=torch.float16,
            device_map="auto",
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
        logger.info("model first param device=%s", str(next(model.parameters()).device))
    except Exception:
        logger.info("model device=unknown (quantized/sharded)")

    logger.info("Loading dataset from: %s", DATA_PATH)
    ds = build_dataset(processor)
    logger.info("Dataset size (used, filtered): %d", len(ds))
    logger.info("Columns: %s", ds.column_names)

    need = TRAIN_SIZE + EVAL_SIZE
    train_dataset = ds.select(range(0, TRAIN_SIZE))
    eval_dataset = ds.select(range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE))
    logger.info("Train size: %d | Eval size: %d", len(train_dataset), len(eval_dataset))

    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        max_memory = round(gpu_stats.total_memory / 1024**3, 3)
        logger.info("GPU=%s | Max memory=%.3f GB | Reserved=%.3f GB", gpu_stats.name, max_memory, start_gpu_memory)

    training_args = build_training_args()
    peft_config = build_lora_config()

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, correctness_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=processor,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving model to: %s", training_args.output_dir)
    trainer.save_model(training_args.output_dir)
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