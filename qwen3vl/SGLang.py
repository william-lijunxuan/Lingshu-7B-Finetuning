import sglang as sgl

# llm = sgl.Engine(model_path="/home/william/model/Qwen3-VL-4B-Instruct")
llm = sgl.Engine(model_path="/home/william/model/Qwen3.5-0.8B")

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}\nGenerated text: {output['text']}\n")

llm.shutdown()


# from openai import OpenAI
# import base64
#
# IMAGE = "/home/william/dataset/skin/Derm1M/IIYI/7_1.png"
# prompt = (
#     "You are given a clinical image and a question.\n Return ONLY the disease name in English. No extra words."
#     "You first think about the reasoning process as an internal monologue and then provide the user with the answer. "
#     "Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
# )
#
# def image_to_base64(image_path):
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")
#
# image_base64 = image_to_base64(IMAGE)
#
# messages = [
#     {
#         "role": "system",
#         "content": (
#             "You are a board-certified dermatology AI specialist. "
#             "A patient has just uploaded an image of a skin lesion."
#         ),
#     },
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": f"data:image/png;base64,{image_base64}"
#                 },
#             },
#             {
#                 "type": "text",
#                 "text": prompt,
#             },
#         ],
#     },
# ]
#
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png"
#                 }
#             },
#             {
#                 "type": "text",
#                 "text": "Where is this?"
#             }
#         ]
#     }
# ]
#
# client = OpenAI(
#     base_url="http://127.0.0.1:30000/v1",
#     api_key="EMPTY",
# )
#
#
# response = client.chat.completions.create(
#     model="/home/william/model/Qwen3.5-0.8B",
#     messages=messages,
#     temperature=0.3,
#     max_tokens=512,
# )
#
# print(response.choices[0].message.content)