import os
from datasets import load_dataset, Image
from transformers import Qwen3VLForConditionalGeneration, BitsAndBytesConfig
import torch
from transformers import AutoProcessor
from peft import LoraConfig
import re
# from math_verify import LatexExtractionConfig, parse, verify
# from latex2sympy2_extended import NormalizationConfig
from trl import GRPOConfig
from trl import GRPOTrainer
import logging
import sys
from datetime import datetime
from  utils import _norm, _canonical,PARENT_MAP

output_dir = "/root/model/Qwen3-VL-4B-Instruct-trl-grpo"
MODEL_TAG = "Qwen3VL_4B"
DATA_PATH = "/root/dataset/skin/SkinCAP/SkinCAP_20250712_121252_close_end_QA.json"
IMAGE_ROOT = "/root/dataset/skin/SkinCAP/skincap"


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

# train_dataset = load_dataset("json", data_files={"train": DATA_PATH}, split="train[:1%]")
train_dataset = load_dataset("json", data_files={"train": DATA_PATH}, split="train")

def to_abs_path(example):
    p = example["image_name"]
    if p and not os.path.isabs(p):
        example["image_name"] = os.path.join(IMAGE_ROOT, p)
    return example

train_dataset = train_dataset.map(to_abs_path)
train_dataset = train_dataset.cast_column("image_name", Image())

print(train_dataset[0]["image_name"])




model_name = "/root/model/Qwen3-VL-4B-Instruct" # "Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_name, padding_side="left")

SYSTEM_PROMPT = (
    "You are given a clinical image and a question.\n Return ONLY the disease name in English. No extra words."
    "You first think about the reasoning process as an internal monologue and then provide the user with the answer. "
    "Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
)


def make_conversation(example):
    prompt = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Image description: "+example["caption_zh_polish_en"]},
            ],
        },
    ]
    return {"prompt": prompt, "image": example["image_name"], "solution": example["answer"] }
train_dataset = train_dataset.map(make_conversation)



train_dataset = train_dataset.remove_columns(['caption_zh', 'caption_zh_polish', 'answer','question_type','image_name','caption_zh_polish_en','image'])



model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name, dtype=torch.bfloat16,
)





def extract_text(completions):
    processed = []
    for c in completions:
        if isinstance(c, list):
            text = " ".join(
                item["text"] if isinstance(item, dict) and "text" in item
                else str(item)
                for item in c
            )
        elif isinstance(c, dict):
            text = c.get("text", str(c))
        else:
            text = c
        processed.append(text)
    return processed

def format_reward(completions, **kwargs):
    pattern = r"<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>"
    contents = extract_text(completions)
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in contents]
    return [1.0 if match else 0.0 for match in matches]

def accuracy_reward(completions, solution, **kwargs):
    contents = extract_text(completions)
    rewards = []
    for i, (content, sol) in enumerate(zip(contents, solution)):
        ans_match = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL)
        pred = ans_match.group(1).strip().lower() if ans_match else ""

        a_norm = _norm(sol)
        p_norm = _norm(pred)

        is_correct = False


        if a_norm == p_norm:
            is_correct = True
        else:
            a_can = _canonical(a_norm)
            p_can = _canonical(p_norm)
            if a_can == p_can:
                is_correct = True
            else:
                children = PARENT_MAP.get(a_can)
                if children and p_can in children:
                    is_correct = True
        reward = 1.0 if is_correct else 0.0
        rewards.append(reward)

        if is_rank0():
            logger.info("idx=%d | reward=%.1f | gt='%s' | pred='%s'", i, reward, a_norm, p_norm)
    return rewards
# def len_reward(completions, solution, **kwargs):
#     contents = extract_text(completions)
#
#     correctness = []
#     for content, sol in zip(contents, solution):
#         ans_match = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL)
#         pred = ans_match.group(1).strip().lower() if ans_match else ""
#         correctness.append(pred == sol.strip().lower())
#
#     lengths = [len(content) for content in contents]
#     min_len = min(lengths)
#     max_len = max(lengths)
#
#     if max_len == min_len:
#         return [0.0] * len(completions)
#
#     rewards = []
#     for length, is_correct in zip(lengths, correctness):
#         lambda_val = 0.5 - (length - min_len) / (max_len - min_len)
#         reward = lambda_val if is_correct else min(0, lambda_val)
#         rewards.append(float(reward))
#
#     return rewards

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(

    learning_rate=2e-5,
    #num_train_epochs=1,
    # max_steps=100,                                        # Number of dataset passes. For full trainings, use `num_train_epochs` instead
    num_train_epochs=3,
    # Parameters that control the data preprocessing
    per_device_train_batch_size=4,
    max_completion_length=256, # default: 256            # Max completion length produced during training
    num_generations=8, # 2, # default: 8                  # Number of generations produced during training for comparison

    fp16=False,
    bf16=True,
    ddp_find_unused_parameters=True,

    # Parameters related to reporting and saving
    output_dir=output_dir,                                # Where to save model checkpoints and logs
    logging_steps=1,                                      # Log training metrics every N steps
    report_to="trackio",                                  # Experiment tracking tool

    # Hub integration
    push_to_hub=True,
    log_completions=True
)


# You may need to update `target_modules` depending on the architecture of your chosen model.
# For example, different VLMs might have different attention/projection layer names.
peft_config = LoraConfig(
    r=64,
    lora_alpha=64,
    lora_dropout=0.1,
    # target_modules=["q_proj", "v_proj"],
    target_modules="all-linear",
)
trainer = GRPOTrainer(
    model=model,
    # reward_funcs=[format_reward, len_reward],
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
logger.info(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()


used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
logger.info(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
logger.info(f"Peak reserved memory = {used_memory} GB.")
logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
logger.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

trainer.save_model(output_dir)
trainer.push_to_hub("williamljx/qwen3vl-skinCap")
logger.info(f"Congratulations! done!")