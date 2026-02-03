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

IN_PATH = r"/home/william/dataset/skin/Derm1M/Derm1M_v2_pretrain_min20_structured_4ok_multiRegion.xlsx".replace("\\", "/")
OUT_PATH = os.path.join(os.path.dirname(IN_PATH), "Derm1M_train_qwen_prompt_eval_964.jsonl")
disease_name =r"/home/william/model/Lingshu-7B-Finetuning/dataset/json/disease_name/derm1m_table10_diseases.txt".replace("\\", "/")

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

    required = {"filename", "caption","disease_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[["filename", "caption","disease_label"]].copy()
    df["filename"] = df["filename"].astype(str).str.strip()
    df["caption"] = df["caption"].astype(str).str.strip()
    df["disease_label"] = df["disease_label"].astype(str).str.strip()
    df = df[(df["filename"] != "") & (df["caption"] != "")& (df["disease_label"] != "")]

    n = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            HUMAN_PROMPT = (
            '''
            **Instruction (system)**
            You are a dermatology vision–language assistant. Given one clinical or dermoscopic image and optional user text, infer the most likely disease name...
        
            **image or clinical description**
            ''' + row["caption"] + '''
            **Task (user)**
            Answer the question: “What is the name of the disease shown in the image?
            <image>”
            Return a single word or short phrase for the primary answer, and provide top-1 possible diseases with probabilities. Answer in English.

            **Output rules**
            1. Output strict JSON only, no extra text.
            2. `answer` must be one word or a short phrase.
            3. `top1` has exactly 1 item, which includes fields `disease`, `prob`, and `reason`; `prob` should be 1.0.
            4. If the image is not a real lesion or is unreadable, set `answer` to "Not applicable" and return an empty array for `top1`.
            5. Keep reasoning concise and morphological only.
            '''
            )

            gpt_payload = {
                "answer": row["disease_label"],
                "top3": [
                    {"disease": row["disease_label"], "reason": row["caption"]},
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
                    "non_lesion_image": False,
                    "low_resolution_or_glare": False,
                    "occlusion": False
                }
            }

            record = {
                "image": row["filename"],
                "conversations": [
                    {"from": "human", "value": f"{HUMAN_PROMPT}"},
                    {"from": "gpt", "value": json.dumps(gpt_payload, ensure_ascii=False)}
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} records to {OUT_PATH}")

if __name__ == "__main__":
    main()
