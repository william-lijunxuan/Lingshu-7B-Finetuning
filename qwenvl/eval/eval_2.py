from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from peft import PeftModel

BASE = "/home/william/model/Lingshu-7B"
ADAPTER = "/home/william/model/Lingshu-7B-Finetuning/qwenvl/scripts/output"
# ADAPTER = "/home/william/model/Lingshu-7B-Finetuning/qwenvl/train/output"
IMAGE = "/home/william/dataset/skin/Derm1M/IIYI/7_3.png"


# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(BASE,use_fast=True)

# model = PeftModel.from_pretrained(model, ADAPTER)
# model = model.merge_and_unload()
# model.eval()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "The patient has a body temperature of 38.8°C, feels weak and tired, experiences chills and sweating, and complains of a headache and sore throat. What illness could this indicate?"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
