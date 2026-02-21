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
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# =========================
# 0) Config
# =========================
DATA_PATH = "/mnt/d/skinalor/dataset/skin/SkinCAP/SkinCAP_20250712_121252_close_end_QA.json"
BASE_IMG_DIR = "/mnt/d/skinalor/dataset/skin/SkinCAP/skincap"

MODEL_PATH = "/mnt/d/skinalor/model/Hulu-Med-4B"
OUTPUT_DIR = "/mnt/d/skinalor/model/GRPO_hulumed4b"
MODEL_TAG = "GRPO_hulumed4b"

TRAIN_SIZE = 4
EVAL_SIZE = 2

MAX_Q_CHARS = 800
MAX_A_CHARS = 400

IMAGE_MIN_PIXELS = 224 * 224
IMAGE_MAX_PIXELS = 336 * 336
IMAGE_SHORTEST_EDGE = 336

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
    ex["image_path"] = os.path.join(BASE_IMG_DIR, ex["image_name"])
    return ex


def to_prompt(ex):
    q = ex.get("caption_zh_polish_en")
    q = "null" if q is None else str(q)

    # Keep prompt text-only for stability; image comes from dataset "image" column.
    prompt = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(q=q)},
    ]

    return {
        "prompt": prompt,
        "answer": str(ex.get("answer", "")),
        "image_name": str(ex.get("image_name", "")),
        "question_type": str(ex.get("question_type", "")),
        "image": ex["image_path"],
    }


def build_dataset():
    data = load_json_list(DATA_PATH)
    ds = Dataset.from_list(data)

    ds = ds.map(add_image_path)
    ds = ds.cast_column("image_path", datasets.Value("string"))

    ds = ds.map(to_prompt, remove_columns=ds.column_names)
    ds = ds.cast_column("image", datasets.Image())
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
    return content if isinstance(content, str) else ""


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
        eval_on_start=False,
        learning_rate=5e-6,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=4,

        max_prompt_length=256,
        max_completion_length=64,

        max_steps=1700,
        logging_steps=20,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=200,

        report_to="tensorboard",

        use_vllm=False,  # keep off for stability
        bf16=True,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        push_to_hub=False,
    )


def build_lora_config():
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=64,
        lora_alpha=64,
        target_modules="all-linear",
    )


# =========================
# 5) Model / Processor setup
# =========================
def try_configure_image_processor(processor):
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return
    if hasattr(ip, "do_resize"):
        ip.do_resize = True
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = IMAGE_MAX_PIXELS
    if hasattr(ip, "min_pixels"):
        ip.min_pixels = IMAGE_MIN_PIXELS
    if hasattr(ip, "size"):
        try:
            ip.size = {"shortest_edge": IMAGE_SHORTEST_EDGE}
        except Exception:
            pass
    if hasattr(ip, "crop_size"):
        try:
            ip.crop_size = {"height": IMAGE_SHORTEST_EDGE, "width": IMAGE_SHORTEST_EDGE}
        except Exception:
            pass


def enforce_token_ids(model, tokenizer):
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id is None; cannot stabilize generation.")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    gen = model.generation_config
    gen.eos_token_id = tokenizer.eos_token_id
    gen.pad_token_id = tokenizer.pad_token_id
    gen.bos_token_id = tokenizer.bos_token_id

    gen.min_new_tokens = 1
    gen.max_new_tokens = 64
    gen.do_sample = True


def sanity_check_encoding(model, processor, tokenizer, sample):
    # Hulu-Med processor may return Tensor instead of dict; support both.
    enc = processor.apply_chat_template(
        sample["prompt"],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )

    if isinstance(enc, torch.Tensor):
        prompt_len = enc.shape[-1]
    elif isinstance(enc, dict):
        input_ids = enc.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            prompt_len = input_ids.shape[-1]
        else:
            raise RuntimeError(f"Unexpected enc['input_ids'] type: {type(input_ids)}")
    else:
        raise RuntimeError(f"Unexpected encoding return type: {type(enc)}")

    max_ctx = getattr(model.config, "max_position_embeddings", None)
    if max_ctx is None:
        max_ctx = getattr(tokenizer, "model_max_length", None)

    logger.info("Sanity check: prompt_len=%s | model_max_ctx=%s", str(prompt_len), str(max_ctx))
    return prompt_len, max_ctx


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

    logger.info("Loading model: %s", MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.bfloat16,  # use dtype to avoid torch_dtype deprecation warning
        device_map="auto",
        attn_implementation="eager",
    )

    logger.info("Loading processor/tokenizer: %s", MODEL_PATH)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )

    if hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer

    try_configure_image_processor(processor)
    enforce_token_ids(model, tokenizer)

    # Fail-fast check: now compatible with Tensor/dict return types
    _ = sanity_check_encoding(model, processor, tokenizer, train_dataset[0])

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
