import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

BASE = "/root/model/Lingshu-7B"
ADAPTER = "/root/model/Lingshu-7B-Finetuning/qwenvl/scripts/output"
# ADAPTER = "/root/model/Lingshu-7B-Finetuning/qwenvl/train/output"
IMAGE = "/root/dataset/skin/Derm1M/IIYI/7_3.png"

def load_model(base_path, adapter_path=None):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(base_path,use_fast=True)


    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    return model, processor

@torch.inference_mode()
def chat_vl(model, processor, image_path, prompt, max_new_tokens=256):
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    if model.config.pad_token_id is None and hasattr(processor, "tokenizer"):
        model.config.pad_token_id = processor.tokenizer.pad_token_id

    output_ids = model.generate(
        **inputs,
        do_sample=True, temperature=0.7, top_p=0.9, top_k=50,
        max_new_tokens=128
    )
    out = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return out

if __name__ == "__main__":
    model, processor = load_model(BASE, ADAPTER)
    # prompt = (
    #     "You are a board‐certified dermatology AI specialist. A patient has just uploaded an image of a skin lesion. "
    #     "Carefully examine the lesion’s visual features—color, shape, borders, surface texture, and anatomic location—and "
    #     "then compose a single, fully descriptive diagnostic sentence in English. Mirror the expert style by:\n"
    #     "            1. Opening with a concise description of the key visual finding (e.g. “The red, smooth, exophytic nodule with a slightly narrowed base…”).\n"
    #     "            2. Stating the most likely diagnosis (e.g. “…may indicate squamous cell carcinoma.”).\n"
    #     "            3. Optionally noting any next steps for confirmation (e.g. “Further biopsy is recommended to confirm the diagnosis.”).\n\n"
    #     "            Example output (for a smooth red papule on the lip):\n"
    #     "            “The red, smooth, dome-shaped papule on the lip, with slight keratosis and prominent capillaries, is most consistent with basal cell carcinoma; a skin biopsy is advised for confirmation.“"
    # )
    prompt = "Describe the key dermatological features and likely diagnosis."
    print(chat_vl(model, processor, IMAGE, prompt))
