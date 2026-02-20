import os
from datasets import load_dataset, Image
from transformers import Qwen3VLForConditionalGeneration, BitsAndBytesConfig
import torch
from transformers import AutoProcessor
from peft import LoraConfig
import re
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from trl import GRPOConfig
from trl import GRPOTrainer
import os
os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"
os.environ["TORCH_DTYPE"] = "float16"


output_dir = "/root/model/Qwen3-VL-4B-Instruct-trl-grpo"
DATA_PATH = "/root/dataset/skin/SkinCAP/SkinCAP_20250712_121252_close_end_QA.json"
IMAGE_ROOT = "/root/dataset/skin/SkinCAP/skincap"

train_dataset = load_dataset("json", data_files={"train": DATA_PATH}, split="train[:5%]")

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
                # {"type": "image", "image": example["image_name"]},
                {"type": "text", "text": "Image description: "+example["caption_zh_polish_en"]},
            ],
        },
    ]
    return {"prompt": prompt, "image": example["image_name"], "solution": example["answer"] }
train_dataset = train_dataset.map(make_conversation)

keep = {"prompt", "image", "solution"}
train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in keep])
# train_dataset = train_dataset.map(make_conversation)
#
# train_dataset = train_dataset.remove_columns(['caption_zh', 'caption_zh_polish', 'answer','question_type','image_name','caption_zh_polish_en','image'])
#


model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "22GiB", 1: "22GiB"},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    ),
)

dtypes = {}
for _, p in model.named_parameters():
    dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + 1
print("param dtypes:", dtypes)



# You may need to update `target_modules` depending on the architecture of your chosen model.
# For example, different VLMs might have different attention/projection layer names.
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

def _to_text(x, **kwargs):
    if isinstance(x, str):
        return x

    # token ids
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], int):
        proc = kwargs.get("processing_class", None)
        tok = getattr(proc, "tokenizer", None) if proc is not None else None
        if tok is not None:
            return tok.decode(x, skip_special_tokens=True)
        return " ".join(map(str, x))

    # list of strings / dicts
    if isinstance(x, list):
        return "\n".join(_to_text(i, **kwargs) for i in x)

    if isinstance(x, dict):
        if "text" in x and x["text"] is not None:
            return str(x["text"])
        if "content" in x:
            return _to_text(x["content"], **kwargs)
        return str(x)

    return str(x)


def format_reward(completions, **kwargs):
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    out = []
    for c in completions:
        s = _to_text(c, **kwargs).strip()
        out.append(1.0 if re.match(pattern, s, re.DOTALL) else 0.0)
    return out


# def format_reward(completions, **kwargs):
#     """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
#     pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
#     matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
#     return [1.0 if match else 0.0 for match in matches]


def _extract_answer(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else text).strip().lower()

def len_reward(completions, solution, **kwargs):
    texts = [_to_text(c, **kwargs) for c in completions]
    sols = [str(s).strip().lower() for s in solution]

    ok = [_extract_answer(t) == s for t, s in zip(texts, sols)]
    lens = [len(t) for t in texts]
    mn, mx = min(lens), max(lens)
    if mx == mn:
        return [0.0] * len(texts)

    rewards = []
    for L, correct in zip(lens, ok):
        lam = 0.5 - (L - mn) / (mx - mn)
        rewards.append(float(lam if correct else min(0.0, lam)))
    return rewards

# def len_reward(completions, solution, **kwargs) -> float:
#     """Compute length-based rewards to discourage overthinking and promote token efficiency.
#
#     Taken from the Kimi 1.5 tech report: https://huggingface.co/papers/2501.12599
#
#     Args:
#         completions: List of model completions
#         solution: List of ground truth solutions
#
#     Returns:
#         List of rewards where:
#         - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
#         - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
#     """
#     contents = completions
#
#     # First check correctness of answers
#     correctness = []
#     for content, sol in zip(contents, solution):
#         gold_parsed = parse(
#             sol,
#             extraction_mode="first_match",
#             extraction_config=[LatexExtractionConfig()],
#         )
#         if len(gold_parsed) == 0:
#             # Skip unparsable examples
#             correctness.append(True)  # Treat as correct to avoid penalizing
#             print("Failed to parse gold solution: ", sol)
#             continue
#
#         answer_parsed = parse(
#             content,
#             extraction_config=[
#                 LatexExtractionConfig(
#                     normalization_config=NormalizationConfig(
#                         nits=False,
#                         malformed_operators=False,
#                         basic_latex=True,
#                         equations=True,
#                         boxed=True,
#                         units=True,
#                     ),
#                     boxed_match_priority=0,
#                     try_extract_without_anchor=False,
#                 )
#             ],
#             extraction_mode="first_match",
#         )
#         correctness.append(verify(answer_parsed, gold_parsed))
#
#     # Calculate lengths
#     lengths = [len(content) for content in contents]
#     min_len = min(lengths)
#     max_len = max(lengths)
#
#     # If all responses have the same length, return zero rewards
#     if max_len == min_len:
#         return [0.0] * len(completions)
#
#     rewards = []
#     for length, is_correct in zip(lengths, correctness):
#         lambda_val = 0.5 - (length - min_len) / (max_len - min_len)
#
#         if is_correct:
#             reward = lambda_val
#         else:
#             reward = min(0, lambda_val)
#
#         rewards.append(float(reward))
#
#     return rewards




# Configure training arguments using GRPOConfig
training_args = GRPOConfig(

    learning_rate=2e-5,
    #num_train_epochs=1,
    max_steps=100,                                        # Number of dataset passes. For full trainings, use `num_train_epochs` instead

    # Parameters that control the data preprocessing
    per_device_train_batch_size=2,
    max_completion_length=256, # default: 256            # Max completion length produced during training
    num_generations=2, # 2, # default: 8                  # Number of generations produced during training for comparison

    fp16=True,
    bf16=False,
    max_grad_norm=0.0,
    # Parameters related to reporting and saving
    output_dir=output_dir,                                # Where to save model checkpoints and logs
    logging_steps=1,                                      # Log training metrics every N steps
    report_to="trackio",                                  # Experiment tracking tool

    # Hub integration
    push_to_hub=False,
    log_completions=True
)



trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, len_reward],
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
    processing_class=processor,
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

print(f"Congratulations! done!")