from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, GenerationConfig
from torch.cuda.amp import autocast
import torch
import time

class MedVLM_R1:
    def __init__(self, model_path, args):
        super().__init__()
        self.llm = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="flash_attention_2")
        self.processor = AutoProcessor.from_pretrained(model_path,use_fast=True)
        self.processor.save_pretrained(model_path)
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens
        self.temp_generation_config = GenerationConfig(
            max_new_tokens=1024,
            do_sample=False,
            temperature=1,
            num_return_sequences=1,
            pad_token_id=151643,
        )

    def process_messages(self, messages):
        new_messages = []
        if "system" in messages:
            new_messages.append({"role": "system", "content": messages["system"]})
        if "image" in messages:
            new_messages.append({"role": "user", "content": [{"type": "image", "image": messages["image"]},
                                                             {"type": "text", "text": messages["prompt"]}]})
        elif "images" in messages:
            content = []
            for i, image in enumerate(messages["images"]):
                content.append({"type": "text", "text": f"<image_{i + 1}>: "})
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": messages["prompt"]})
            new_messages.append({"role": "user", "content": content})
        else:
            new_messages.append({"role": "user", "content": [{"type": "text", "text": messages["prompt"]}]})
        messages = new_messages
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        return inputs

    def generate_output(self, messages):
        inputs = self.process_messages(messages)
        # with torch.no_grad(), autocast():
        generated_ids = self.llm.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False, generation_config=self.temp_generation_config)
        # torch.cuda.empty_cache()
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def generate_outputs(self, messages_list):
        res = []
        sub_total_time= []
        for idx, messages in enumerate(messages_list, start=1):
            start_times = time.perf_counter()
            result = self.generate_output(messages)
            end_times = time.perf_counter()
            delta =end_times - start_times
            print(f"idx:{idx},result-------------:{result},total_time:{delta}")

            res.append(result)
            sub_total_time.append(delta)
        return res, sub_total_time