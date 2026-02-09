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
CSV_PATH = r"/mnt/d/skinalor/dataset/skin/Derm1M/Derm1M_balanced_10k.csv"
CSV_PATH = CSV_PATH.replace("\\", "/")
OUT_PATH = os.path.join(os.path.dirname(CSV_PATH), "Derm1M_train_label_10k.jsonl")

# Fixed human prompt; the visual placeholder token is appended at runtime
# HUMAN_PROMPT = (
#     "You are a board‐certified dermatology AI specialist. A patient has just uploaded an image of a skin lesion. "
#     "Carefully examine the lesion’s visual features—color, shape, borders, surface texture, and anatomic location—and "
#     "then compose a single, fully descriptive diagnostic sentence in English. Mirror the expert style by:\n"
#     "            1. Opening with a concise description of the key visual finding (e.g. “The red, smooth, exophytic nodule with a slightly narrowed base…”).\n"
#     "            2. Stating the most likely diagnosis (e.g. “…may indicate squamous cell carcinoma.”).\n"
#     "            3. Optionally noting any next steps for confirmation (e.g. “Further biopsy is recommended to confirm the diagnosis.”).\n\n"
#     "            Example output (for a smooth red papule on the lip):\n"
#     "            “The red, smooth, dome-shaped papule on the lip, with slight keratosis and prominent capillaries, is most consistent with basal cell carcinoma; a skin biopsy is advised for confirmation.“"
# )




def main():
    # Read CSV as strings; do not convert empty strings to NaN
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)

    # Verify required columns
    required = {"filename", "caption"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    # Keep and clean only the necessary columns
    df = df[["filename", "caption","disease_label"]].copy()
    df["filename"] = df["filename"].astype(str).str.strip()
    df["filename"] = df["filename"].astype(str).str.strip()
    df["disease_label"] = df["disease_label"].astype(str).str.strip()

    # Drop rows with empty filename or caption
    df = df[(df["filename"] != "") & (df["caption"] != "")& (df["disease_label"] != "")]



    # Stream-write JSONL (no pretty indent; one JSON object per line)
    n = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            HUMAN_PROMPT = (
                    '''
                **Instruction (system)**
                You are a dermatology vision–language assistant. Given one clinical or dermoscopic image and optional user text, infer the most likely disease name. If the image is not a lesion photo (e.g., poster, icon, cartoon) or is too poor-quality to assess, return “Not applicable”. Use only visual evidence and the user text; do not invent details.

                **image or clinical description**
                '''
                    + row["caption"] +
                    '''
                **Task (user)**
                Answer the question: “What is the name of the disease shown in the image?
                <image>”
                Return a single word or short phrase for the primary answer, and provide top-3 possible diseases with probabilities. Answer in English.

                **Output rules**
                1. Output strict JSON only, no extra text.
                2. `answer` must be one word or a short phrase.
                3. `top3` has exactly 3 items, each item includes fields `disease`, `prob`, and `reason`; the list is sorted by `prob` (0–1) in descending order, and the three `prob` values sum to 1.0 (±0.01). The `reason` is a concise morphological justification (e.g., region, color/shape/border/texture, elevation, perilesional skin).
                4. If the image is not a real lesion or is unreadable, set `answer` to "Not applicable" and return an empty array for `top3`.
                5. Keep reasoning concise and purely morphological (region, color/shape/border/texture, elevation, perilesional skin). No treatment advice.

                **JSON schema**
                {
                  "answer": "<single word or short phrase>",
                  "top3": [
                    {"disease": "<name>", "prob": 0.00, "reason": "<short morphological rationale>"},
                    {"disease": "<name>", "prob": 0.00, "reason": "<short morphological rationale>"},
                    {"disease": "<name>", "prob": 0.00, "reason": "<short morphological rationale>"}
                  ],
                  "reason": {
                    "region": "<if discernible>",
                    "lesion_morphology": {
                      "size_mm": "<if scale visible, else null>",
                      "shape": "<round/oval/irregular>",
                      "border": "<well-defined/ill-defined; smooth/notched>",
                      "colour": "<uniform/variegated + hues>",
                      "surface": "<smooth/scaly/crusted/ulcerated/verrucous>"
                    },
                    "elevation": "<flat/slightly raised/plaque/nodule/depressed/NA>",
                    "perilesional_skin": "<erythema/induration/atrophy/scale/bleeding/NA>"
                  },
                  "quality_flags": {
                    "non_lesion_image": false,
                    "low_resolution_or_glare": false,
                    "occlusion": false
                  }
                }
                '''
            )
            record = {
                "image": row["filename"],
                "disease_label": row["disease_label"],
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
