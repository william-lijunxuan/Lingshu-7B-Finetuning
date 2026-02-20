from datasets import load_dataset
from trl import GRPOConfig
from transformers import Qwen3VLForConditionalGeneration, BitsAndBytesConfig
import torch
from peft import LoraConfig
import re

dataset_id = 'lmms-lab/multimodal-open-r1-8k-verified'
train_dataset = load_dataset(dataset_id, split='train[:5%]')
from transformers import AutoProcessor

model_name = "model/Qwen3-VL-4B-Instruct"  # "Qwen/Qwen3-VL-8B-Instruct"
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
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": "Image description: "+example["problem"]},
            ],
        },
    ]
    return {"prompt": prompt, "image": example["image"]}


train_dataset = train_dataset.map(make_conversation)



model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name, dtype="float32",
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    ),
)



# You may need to update `target_modules` depending on the architecture of your chosen model.
# For example, different VLMs might have different attention/projection layer names.
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)




def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    return [1.0 if match else 0.0 for match in matches]


from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig


def len_reward(completions, solution, **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://huggingface.co/papers/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = completions

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparsable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards




output_dir = "Qwen3-VL-4B-Instruct-trl-grpo"

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    learning_rate=2e-5,
    # num_train_epochs=1,
    max_steps=100,  # Number of dataset passes. For full trainings, use `num_train_epochs` instead

    # Parameters that control the data preprocessing
    per_device_train_batch_size=2,
    max_completion_length=1024,  # default: 256            # Max completion length produced during training
    num_generations=2,
    # 2, # default: 8                  # Number of generations produced during training for comparison

    fp16=True,

    # Parameters related to reporting and saving
    output_dir=output_dir,  # Where to save model checkpoints and logs
    logging_steps=1,  # Log training metrics every N steps
    report_to="trackio",  # Experiment tracking tool

    # Hub integration
    push_to_hub=True,
    log_completions=True
)

from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, len_reward],
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
print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

trainer.save_model(output_dir)
trainer.push_to_hub(dataset_name=dataset_id)

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

base_model = model_name
adapter_model = f"{output_dir}"  # Replace with your HF username or organization

model = Qwen3VLForConditionalGeneration.from_pretrained(base_model, dtype="float32", device_map="auto")
model = PeftModel.from_pretrained(model, adapter_model)

processor = AutoProcessor.from_pretrained(base_model)

from datasets import load_dataset

dataset_id = 'lmms-lab/multimodal-open-r1-8k-verified'
train_dataset = load_dataset(dataset_id, split='train[:5%]')

problem = train_dataset[0]['problem']
image = train_dataset[0]['image']

messages = [
    {
        "role": "system", "content": [
        {"type": "text", "text": SYSTEM_PROMPT}
    ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": problem},
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

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)