"""

"""

import os
import json
import pandas as pd

# Input CSV and output JSONL paths
CSV_PATH = r"/mnt/d/skinalor/dataset/skin/Derm1M/Derm1M_v2_pretrain.csv"
CSV_PATH = CSV_PATH.replace("\\", "/")
OUT_PATH = os.path.join(os.path.dirname(CSV_PATH), "eval_Derm1M_train_json_420k.jsonl")


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
    import re



    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)


    required = {"filename", "caption", "disease_label"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"CSV must contain columns: {sorted(required)}; missing: {missing}")

    df = df[["filename", "caption", "disease_label"]].copy()
    for c in ["filename", "caption", "disease_label"]:
        df[c] = df[c].astype(str).str.strip()


    df = df[(df["filename"] != "") & (df["caption"] != "") & (df["disease_label"] != "")]

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
3. `top1` has exactly 1 item, which includes fields `disease`,and `reason`.
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
                    {"from": "gpt", "value": json.dumps(gpt_payload, ensure_ascii=False)}  # 写成字符串
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} records to {OUT_PATH}")


if __name__ == "__main__":
    main()
