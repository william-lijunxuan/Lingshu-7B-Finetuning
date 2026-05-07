#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import pandas as pd
from pathlib import Path

IN_PATH  = "/mnt/d/skinalor/dataset/skin/Derm1M/Derm1M_v2_pretrain.csv"
OUT_PATH = "/mnt/d/skinalor/dataset/skin/Derm1M/Derm1M_v2_pretrain_filtered.xlsx"

REQUIRED_COLS = [
    "disease_label",
    "hierarchical_disease_label",
    "skin_concept",
    "body_location",
    "symptoms",
]

BAD_PHRASES = {
    "disease_label": {"no definitive diagnosis"},
    "hierarchical_disease_label": {"no definitive diagnosis"},
    "skin_concept": {"no visual concepts"},
    "body_location": {"no body location information"},
    "symptoms": {"no symptom information"},
}

CHN_RE   = re.compile(r"[\u4e00-\u9fff]")
# strip leading/trailing non-word and whitespace for robust equality checks
EDGE_PUNCT_RE = re.compile(r"^[^\w\u4e00-\u9fff]+|[^\w\u4e00-\u9fff]+$",
                           flags=re.UNICODE)
# detect punctuation-only (no word chars and no CJK letters)
PUNCT_ONLY_RE = re.compile(r"^(?:[^\w\u4e00-\u9fff]+)$", flags=re.UNICODE)

def normalize(s: str) -> str:
    t = (s or "").strip().lower()
    t = EDGE_PUNCT_RE.sub("", t)
    return t

def is_bad_value(s: str, bad_set: set) -> bool:
    t = (s or "").strip()
    if t == "":
        return True
    # Chinese present -> drop
    if CHN_RE.search(t):
        return True
    # punctuation-only -> drop
    if PUNCT_ONLY_RE.fullmatch(t):
        return True
    # phrase matches (case-insensitive, tolerant to edge punctuation)
    t_norm = normalize(t)
    if t_norm in {normalize(x) for x in bad_set}:
        return True
    return False

def main():
    df = pd.read_csv(IN_PATH, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # ensure string dtype
    for c in REQUIRED_COLS:
        df[c] = df[c].astype(str)

    bad_masks = []
    for col in REQUIRED_COLS:
        phrases = BAD_PHRASES.get(col, set())
        mask = df[col].apply(lambda v: is_bad_value(v, phrases))
        bad_masks.append(mask)

    drop_mask = bad_masks[0]
    for m in bad_masks[1:]:
        drop_mask = drop_mask | m

    out = df.loc[~drop_mask].copy()
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(OUT_PATH, index=False)

if __name__ == "__main__":
    main()
