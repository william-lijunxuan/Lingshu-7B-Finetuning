#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pandas as pd
from pathlib import Path


CSV_PATH = Path("/home/william/dataset/skin/Derm1M/Derm1M_v2_pretrain_HD.csv")
IN_JSONL = Path("/home/william/dataset/skin/Derm1M/eval_Derm1M_train_json_1k_clean.jsonl")
OUT_JSONL = Path("/home/william/dataset/skin/Derm1M/eval_Derm1M_train_json_1k_clean_new.jsonl")


def norm_path(p: str) -> str:
    return str(p).strip().replace("\\", "/")


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(str(CSV_PATH))
    if not IN_JSONL.exists():
        raise FileNotFoundError(str(IN_JSONL))

    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)

    if "filename" not in df.columns or "body_location" not in df.columns:
        raise RuntimeError(f"CSV must contain 'filename' and 'body_location' columns. Found: {list(df.columns)}")

    df["filename"] = df["filename"].astype(str).map(norm_path)
    df["body_location"] = df["body_location"].astype(str).str.strip()

    body_map = dict(zip(df["filename"], df["body_location"]))

    total = 0
    matched = 0
    missing = 0

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with IN_JSONL.open("r", encoding="utf-8") as fin, OUT_JSONL.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            img = norm_path(obj.get("image", ""))
            body_location = body_map.get(img, "")

            if body_location:
                matched += 1
            else:
                missing += 1

            obj["body_location"] = body_location
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Done. total={total} matched={matched} missing={missing} out={OUT_JSONL}")


if __name__ == "__main__":
    main()
