#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert Derm1M_v2_pretrain(.csv/.xlsx) to JSONL (one record per line).

Uses:
- filename        -> JSON "image"
- caption_format  -> JSON conversations[1].value

Output schema (per line):
{
  "image": "<filename>",
  "conversations": [
    { "from": "human", "value": "<PROMPT>\n<image>" },
    { "from": "gpt",   "value": "<caption_format>" }
  ]
}
"""

import os
import json
from pathlib import Path
import pandas as pd

IN_PATH = r"/mnt/d/skinalor/dataset/skin/Derm1M/Derm1M_v2_pretrain_min20_structured_4ok_multiRegion.xlsx".replace("\\", "/")
OUT_PATH = os.path.join(os.path.dirname(IN_PATH), "Derm1M_train_qwen_prompt_eval.jsonl")

HUMAN_PROMPT = (
    "Write a description of the following skin lesion image, and your description should include the following "
    "information if they are clearly discernible from the image.\n"
    "1. region: The potential area of the body where the lesion or wound has been examined.\n"
    "2. general skin texture and hair growth.\n"
    "3. lesions: size (if scale is available in the image), shape, definition, color, texture.\n"
    "4. elevation: Description of the lesion or wound relative to the skin surface of the patient.\n"
)

def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".xlsx", ".xls"):
        # pandas will use openpyxl for .xlsx if available
        return pd.read_excel(p, dtype=str, keep_default_na=False)
    # CSV fallback with encoding tries
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(p, dtype=str, keep_default_na=False, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("Failed to read file")

def main():
    df = load_table(IN_PATH)

    required = {"filename", "caption_format"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[["filename", "caption_format"]].copy()
    df["filename"] = df["filename"].astype(str).str.strip()
    df["caption_format"] = df["caption_format"].astype(str).str.strip()
    df = df[(df["filename"] != "") & (df["caption_format"] != "")]

    n = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {
                "image": row["filename"],
                "conversations": [
                    {"from": "human", "value": f"{HUMAN_PROMPT}\n<image>"},
                    {"from": "gpt",   "value": row["caption_format"]},
                ],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} records to {OUT_PATH}")

if __name__ == "__main__":
    main()
