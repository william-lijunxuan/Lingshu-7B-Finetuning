from datasets import Dataset
import datasets
import os
import json

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

def load_json_list(path: str):
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("JSON root must be a list of examples.")
    return obj

def add_image_path(ex):
    ex["image_path"] = os.path.join(BASE_IMG_DIR, ex["image_name"])
    return ex


data = load_json_list(DATA_PATH)
train_dataset = Dataset.from_list(data)

train_dataset = train_dataset.map(add_image_path)
train_dataset = train_dataset.cast_column("image_path", datasets.Image())
train_dataset = train_dataset.rename_column("image_path", "image")

train_dataset = train_dataset.select(range(0, TRAIN_SIZE + EVAL_SIZE))


from transformers import AutoProcessor

model_name = CKPT
processor = AutoProcessor.from_pretrained(model_name, padding_side="left")


def make_conversation(example):
    q = example.get("caption_zh_polish_en")
    q = "null" if q is None else str(q)

    prompt = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": USER_TEMPLATE.format(q=q)},
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

train_dataset = train_dataset.map(make_conversation)


from transformers import Qwen3VLForConditionalGeneration, BitsAndBytesConfig
import torch

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="float32",
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    ),
)


from peft import LoraConfig

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)


import re

def format_reward(completions, **kwargs):
    pattern = r"^[A-Za-z0-9][A-Za-z0-9 \-]{0,79}$"
    matches = [re.match(pattern, content.strip()) for content in completions]
    return [1.0 if match else 0.0 for match in matches]


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


def correctness_reward(completions, answer, **kwargs):
    rewards = []
    for pred, gt in zip(completions, answer):
        pred_norm = canonicalize(pred)
        gt_norm = canonicalize(gt)
        if not pred_norm:
            rewards.append(0.0)
        elif pred_norm == gt_norm:
            rewards.append(1.0)
        elif gt_norm in pred_norm or pred_norm in gt_norm:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


from trl import GRPOConfig

output_dir = OUTPUT_DIR

training_args = GRPOConfig(
    learning_rate=2e-5,
    max_steps=100,

    per_device_train_batch_size=2,
    max_completion_length=128,
    num_generations=2,

    fp16=True,

    output_dir=output_dir,
    logging_steps=1,
    report_to="trackio",

    push_to_hub=False,
    log_completions=True,
    remove_unused_columns=False,
)


from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, correctness_reward],
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
)


gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()


used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


trainer.save_model(output_dir)


from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

base_model = model_name
adapter_model = f"{output_dir}"

model = Qwen3VLForConditionalGeneration.from_pretrained(base_model, dtype="float32", device_map="auto")
model = PeftModel.from_pretrained(model, adapter_model)

processor = AutoProcessor.from_pretrained(base_model)


data = load_json_list(DATA_PATH)
infer_ds = Dataset.from_list(data)
infer_ds = infer_ds.map(add_image_path)
infer_ds = infer_ds.cast_column("image_path", datasets.Image())
infer_ds = infer_ds.rename_column("image_path", "image")

example = infer_ds[0]
q = example.get("caption_zh_polish_en")
q = "null" if q is None else str(q)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM}],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": example["image"]},
            {"type": "text", "text": USER_TEMPLATE.format(q=q)},
        ],
    },
]


inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)