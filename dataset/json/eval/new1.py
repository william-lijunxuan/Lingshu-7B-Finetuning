import json
from collections import Counter
from pathlib import Path

FILE_PATH = Path("/home/william/dataset/skin/Derm1M/eval_Derm1M_train_json_1k_clean.jsonl")

images = []

with FILE_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        img = obj.get("image")
        if img:
            images.append(str(img).strip())

counter = Counter(images)
duplicates = [img for img, cnt in counter.items() if cnt > 1]

print("Total records:", len(images))
print("Unique image names:", len(counter))
print("Duplicate image names:", len(duplicates))

if duplicates:
    print("\nDuplicated images:")
    for img in duplicates:
        print(f"- {img} (count={counter[img]})")
else:
    print("\nNo duplicate image names found.")
