import torch, json, os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoConfig
from peft import LoraConfig, get_peft_model
from datasets import Dataset, concatenate_datasets
from qwen_vl_utils import process_vision_info
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from tqdm.auto import tqdm
import time
import pyarrow as pa

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:1024"

# 配置常量
LOCAL_SAVE_PATH = "/root/model/Lingshu-7B"
output_dir="./output_dir/Lingshu-7B-ft"
DATA_PATH = "./config/XXX.json"
MAX_LENGTH = 8192
TRAIN_SIZE = 1000
TEST_SIZE = 5
# 加载模型配置并设置参数
config = AutoConfig.from_pretrained(LOCAL_SAVE_PATH)
config.use_cache = False  # 显式禁用use_cache
config.loss_type = "ForCausalLMLoss"  # 显式设置损失类型

# 加载模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    LOCAL_SAVE_PATH,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 加载处理器，显式设置use_fast
processor = AutoProcessor.from_pretrained(LOCAL_SAVE_PATH, use_fast=True)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_SAVE_PATH)

# 允许梯度更新
model.enable_input_require_grads()

# 加载数据
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
    train_data = data[:-10]
    test_data = data[-10:]
with open("./config/XXX.json", "w", encoding='utf-8') as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)
with open("./config/XXX.json", "w", encoding='utf-8') as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)
train_ds = Dataset.from_json("./config/XXX.json")
resize_w = 512
resize_h = 512


def process_func(example):
    """
    预处理输入数据，正确设置张量类型和梯度需求
    """
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    promptStr = conversation[2]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]

    # 构造多模态对话
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"{file_path}", "resized_height": resize_h, "resized_width": resize_w},
                {"type": "text", "text": promptStr},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()}
    # 构造目标输出
    response = tokenizer(f"{output_content}", add_special_tokens=False)
    input_ids = inputs["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = inputs["attention_mask"][0] + response["attention_mask"] + [1]
    labels = [-100] * len(inputs["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]

    # 截断处理
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    # 正确设置张量类型和梯度需求
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),  # 整数类型，不需要梯度
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),  # 整数类型
        "labels": torch.tensor(labels, dtype=torch.long),  # 整数类型，不需要梯度
        "pixel_values": torch.tensor(inputs["pixel_values"], dtype=torch.bfloat16, requires_grad=True),  # 浮点型，需要梯度
        "image_grid_thw": torch.tensor(inputs["image_grid_thw"], dtype=torch.long).squeeze(0)  # 整数类型
    }


# 分批次处理数据
BATCH_SIZE = 500  # 可根据实际情况调整批次大小
train_datasets = []
for i in tqdm(range(0, len(train_ds), BATCH_SIZE)):
    batch_ds = train_ds.select(range(i, min(i + BATCH_SIZE, len(train_ds))))
    processed_batch_ds = batch_ds.map(process_func)
    train_datasets.append(processed_batch_ds)

train_dataset = concatenate_datasets(train_datasets)
print(f"Train dataset size: {len(train_dataset)}")

# LoRA配置
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    inference_mode=False,
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)

# 应用LoRA
peft_model = get_peft_model(model, lora_config)

# 配置训练参数，优化显存使用
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,  # 减小每个设备的批次大小
    gradient_accumulation_steps=8,  # 增加梯度累积步数
    logging_steps=10,
    num_train_epochs=30,  # 正常值3，工业数据集通常需要更多轮训练
    save_steps=100,
    learning_rate=1e-4,  # 正常值1e-4，微调学习率通常比预训练小
    gradient_checkpointing=True,
    report_to="none",
    fp16=False,
    bf16=True,
    max_grad_norm=1.0,
    save_total_limit=3,
)
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()
