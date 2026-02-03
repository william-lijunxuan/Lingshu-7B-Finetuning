#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
caption has at least 25 words (tokenized by whitespace).

caption contains no CJK characters (regex range: [\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]).

Normalized disease_label is not “no definitive diagnosis”.

Normalized disease_label is in the whitelist (file: /home/william/model/Lingshu-7B-Finetuning/dataset/json/disease_name/derm1m_table10_diseases.txt; fallback to /mnt/data/derm1m_table10_diseases.txt if missing).

Normalized body_location is not “No body location information”.

Normalized skin_concept is not “No visual concepts”.

Note: Normalization applies NFKC, replaces common punctuation variants, trims, collapses whitespace, and lowercases, making comparisons case/spacing/punctuation-insensitive.
'''
import re
import unicodedata
import pandas as pd
from pathlib import Path

IN_PATH  = "/home/william/dataset/skin/Derm1M/Derm1M_v2_pretrain.csv"
OUT_PATH = "/home/william/dataset/skin/Derm1M/Derm1M_v2_pretrain_HD.csv"
DISEASE_LIST = "/home/william/model/Lingshu-7B-Finetuning/dataset/json/disease_name/derm1m_table10_diseases.txt"
MIN_WORDS = 25
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]")

def norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s.strip())
    return s.lower()

def load_whitelist(path: str) -> set:
    p = Path(path)
    if not p.exists():
        alt = Path("/mnt/data/derm1m_table10_diseases.txt")
        if alt.exists():
            p = alt
    with open(p, "r", encoding="utf-8") as f:
        return {norm(line) for line in f if line.strip()}

def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))

def main():
    df = pd.read_csv(IN_PATH, dtype=str, keep_default_na=False)
    need = {"caption", "disease_label", "body_location", "skin_concept"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing columns: {sorted(miss)}")

    for c in need:
        df[c] = df[c].astype(str).fillna("").str.strip()

    df["caption_words"] = df["caption"].str.split().str.len()
    df["caption_has_cjk"] = df["caption"].apply(has_cjk)

    df["disease_label_norm"] = df["disease_label"].map(norm)
    df["body_location_norm"] = df["body_location"].map(norm)
    df["skin_concept_norm"]  = df["skin_concept"].map(norm)

    wl = load_whitelist(DISEASE_LIST)
    bad_dx  = norm("no definitive diagnosis")
    bad_loc = norm("No body location information")
    bad_con = norm("No visual concepts")

    mask = (
        (df["caption_words"] >= MIN_WORDS) &
        (~df["caption_has_cjk"]) &
        (df["disease_label_norm"] != bad_dx) &
        (df["disease_label_norm"].isin(wl)) &
        (df["body_location_norm"] != bad_loc) &
        (df["skin_concept_norm"]  != bad_con)
    )

    out = df.loc[mask].drop(
        columns=["caption_words", "caption_has_cjk", "disease_label_norm", "body_location_norm", "skin_concept_norm"]
    )
    out.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {len(out)} rows to {OUT_PATH}")

if __name__ == "__main__":
    main()
