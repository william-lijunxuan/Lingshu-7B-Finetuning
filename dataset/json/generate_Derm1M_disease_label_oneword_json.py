import os
import re
import json
import unicodedata
from pathlib import Path
import pandas as pd

IN_PATH = r"/home/root/dataset/skin/Derm1M/Derm1M_v2_pretrain_min20_structured_4ok_multiRegion.xlsx".replace("\\",
                                                                                                                "/")
OUT_PATH = os.path.join(os.path.dirname(IN_PATH), "Derm1M_train_qwen_prompt_eval_964.jsonl")
disease_name = r"/home/root/model/Lingshu-7B-Finetuning/dataset/json/disease_name/derm1m_table10_diseases.txt".replace("\\", "/")

COUNTS_CSV = os.path.join(os.path.dirname(OUT_PATH), "Derm1M_filtered_disease_counts.csv")


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(p, dtype=str, keep_default_na=False)
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(p, dtype=str, keep_default_na=False, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("Failed to read file")


def norm_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def load_allowed_diseases(path: str) -> set:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        names = [norm_text(line) for line in f if line.strip()]
    return set(names)


def main():
    df = load_table(IN_PATH)

    required = {"filename", "caption", "disease_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[["filename", "caption", "disease_label"]].copy()
    df["filename"] = df["filename"].astype(str).str.strip()
    df["caption"] = df["caption"].astype(str).str.strip()
    df["disease_label"] = df["disease_label"].astype(str).str.strip()

    df = df.replace(
        {"filename": r"^\s*$", "caption": r"^\s*$", "disease_label": r"^\s*$"},
        {"filename": "", "caption": "", "disease_label": ""},
        regex=True,
    )
    df = df[(df["filename"] != "") & (df["caption"] != "") & (df["disease_label"] != "")]
    df = df.dropna(subset=["filename", "caption", "disease_label"])

    allowed = load_allowed_diseases(disease_name)
    df["disease_label_norm"] = df["disease_label"].map(norm_text)
    before = len(df)
    df = df[df["disease_label_norm"].isin(allowed)].copy()
    after = len(df)

    rep_name = (
        df.groupby("disease_label_norm")["disease_label"]
        .agg(lambda s: s.value_counts().idxmax())
    )
    counts_norm = df["disease_label_norm"].value_counts().sort_values(ascending=False)
    counts_display = counts_norm.rename(index=rep_name.to_dict())
    counts_display.to_csv(COUNTS_CSV, header=["count"])

    print(f"Filtered rows: {before} -> {after}")
    print(f"Unique diseases: {counts_display.index.nunique()}")
    print(f"Top-10:\n{counts_display.head(75)}")
    print(f"Counts saved: {COUNTS_CSV}")

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
            ).strip()

            gpt_payload = {
                "answer": row["disease_label"],
                "top1": [
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
                    {"from": "human", "value": HUMAN_PROMPT},
                    {"from": "gpt", "value": json.dumps(gpt_payload, ensure_ascii=False)}
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} records to {OUT_PATH}")


if __name__ == "__main__":
    main()
