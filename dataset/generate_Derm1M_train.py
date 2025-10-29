#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert Derm1M_v2_pretrain.csv to JSONL (one record per line).

CSV columns used:
- filename  -> JSON field "image"
- caption   -> JSON field conversations[1].value (assistant answer)

JSONL output schema (per line):
{
  "image": "<filename>",
  "conversations": [
    { "from": "human", "value": "<PROMPT>\n<image>" },
    { "from": "gpt",   "value": "<caption>" }
  ]
}
"""

import os
import json
import pandas as pd

# Input CSV and output JSONL paths
CSV_PATH = r"/home/william/dataset/skin/Derm1M/Derm1M_v2_pretrain.csv"
CSV_PATH = CSV_PATH.replace("\\", "/")
OUT_PATH = os.path.join(os.path.dirname(CSV_PATH), "Derm1M_train.jsonl")

# Fixed human prompt; the visual placeholder token is appended at runtime
HUMAN_PROMPT = (
    "You are a board‐certified dermatology AI specialist. A patient has just uploaded an image of a skin lesion. "
    "Carefully examine the lesion’s visual features—color, shape, borders, surface texture, and anatomic location—and "
    "then compose a single, fully descriptive diagnostic sentence in English. Mirror the expert style by:\n"
    "            1. Opening with a concise description of the key visual finding (e.g. “The red, smooth, exophytic nodule with a slightly narrowed base…”).\n"
    "            2. Stating the most likely diagnosis (e.g. “…may indicate squamous cell carcinoma.”).\n"
    "            3. Optionally noting any next steps for confirmation (e.g. “Further biopsy is recommended to confirm the diagnosis.”).\n\n"
    "            Example output (for a smooth red papule on the lip):\n"
    "            “The red, smooth, dome-shaped papule on the lip, with slight keratosis and prominent capillaries, is most consistent with basal cell carcinoma; a skin biopsy is advised for confirmation.“"
)

def main():
    # Read CSV as strings; do not convert empty strings to NaN
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)

    # Verify required columns
    required = {"filename", "caption"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    # Keep and clean only the necessary columns
    df = df[["filename", "caption"]].copy()
    df["filename"] = df["filename"].astype(str).str.strip()
    df["caption"] = df["caption"].astype(str).str.strip()

    # Drop rows with empty filename or caption
    df = df[(df["filename"] != "") & (df["caption"] != "")]

    # Stream-write JSONL (no pretty indent; one JSON object per line)
    n = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "image": row["filename"],
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{HUMAN_PROMPT}\n<image>"
                    },
                    {
                        "from": "gpt",
                        "value": row["caption"]
                    }
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} records to {OUT_PATH}")

if __name__ == "__main__":
    main()
