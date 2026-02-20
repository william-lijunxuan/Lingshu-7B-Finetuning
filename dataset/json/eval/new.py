#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import pandas as pd
from pathlib import Path

CSV_PATH = "/home/william/dataset/skin/Derm1M/Derm1M_v2_pretrain_HD.csv"
BASE_DIR = os.path.dirname(CSV_PATH)

EXISTING_JSONL = os.path.join(BASE_DIR, "eval_Derm1M_train_json_1k.jsonl")
OUT_PATH = os.path.join(BASE_DIR, "eval_Derm1M_train_json_1k2.jsonl")

N_SAMPLES = 150
SEED = 42


def allocate_quota(counts: pd.Series, total: int, seed: int = 42) -> dict:
    rng = random.Random(seed)
    labels = list(counts.index)
    rng.shuffle(labels)

    avail = {lbl: int(counts[lbl]) for lbl in labels}
    alloc = {lbl: 0 for lbl in labels}

    total_available = sum(avail.values())
    if total_available <= total:
        return avail.copy()

    if len(labels) <= total:
        for lbl in labels:
            if avail[lbl] > 0:
                alloc[lbl] = 1
        remaining = total - sum(alloc.values())
    else:
        top = sorted(labels, key=lambda x: avail[x], reverse=True)[:total]
        for lbl in top:
            alloc[lbl] = 1
        return alloc

    k = len(labels)
    base = max(0, (total // k) - 1)
    if base > 0:
        for lbl in labels:
            add = min(base, avail[lbl] - alloc[lbl])
            alloc[lbl] += add
        remaining = total - sum(alloc.values())

    while remaining > 0:
        progressed = False
        labels.sort(key=lambda x: (alloc[x], -avail[x]))
        for lbl in labels:
            if remaining <= 0:
                break
            if alloc[lbl] < avail[lbl]:
                alloc[lbl] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    return alloc


def load_existing_images(jsonl_path: str) -> set:
    existing = set()
    if not os.path.exists(jsonl_path):
        return existing
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            img = obj.get("image", "")
            if img:
                existing.add(str(img).strip())
    return existing


def main():
    existing_images = load_existing_images(EXISTING_JSONL)
    print(f"Loaded existing images: {len(existing_images)} from {EXISTING_JSONL}")

    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)

    need_cols = {"filename", "disease_label", "caption", "body_location"}
    missing = need_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    for c in ["filename", "disease_label", "caption", "body_location"]:
        df[c] = df[c].astype(str).str.strip()

    df = df[(df["filename"] != "") & (df["disease_label"] != "")]
    if df.empty:
        raise RuntimeError("No valid rows after basic cleaning.")

    before = len(df)
    df = df[~df["filename"].isin(existing_images)].copy()
    after = len(df)
    print(f"Filtered overlaps: removed {before - after}, remaining {after}")

    if df.empty:
        raise RuntimeError("No rows left after removing existing images.")

    counts = df["disease_label"].value_counts().sort_index()
    quota = allocate_quota(counts, N_SAMPLES, seed=SEED)

    sampled_parts = []
    for lbl, n in quota.items():
        if n <= 0:
            continue
        sub = df[df["disease_label"] == lbl]
        if sub.empty:
            continue
        take = min(n, len(sub))
        sampled = sub.sample(n=take, random_state=SEED, replace=False)
        sampled_parts.append(sampled)

    if not sampled_parts:
        raise RuntimeError("Sampling produced no data; check CSV and filters.")

    sampled_df = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    if len(sampled_df) < N_SAMPLES:
        print(f"Warning: only {len(sampled_df)} rows available after filtering; requested {N_SAMPLES}.")
    elif len(sampled_df) > N_SAMPLES:
        sampled_df = sampled_df.iloc[:N_SAMPLES].copy()

    sampled_counts = sampled_df["disease_label"].value_counts().sort_index()
    print("Selected diseases and counts:")
    for lbl, cnt in sampled_counts.items():
        print(f"- {lbl}: {cnt}")
    print(f"Total: {len(sampled_df)} rows, {sampled_counts.shape[0]} unique diseases")

    overlap_check = set(sampled_df["filename"]).intersection(existing_images)
    if overlap_check:
        raise RuntimeError(f"Overlap detected after sampling: {len(overlap_check)}")

    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in sampled_df.iterrows():
            rec = {
                "image": row["filename"],
                "caption": row["caption"],
                "answer": row["disease_label"],
                "body_location": row["body_location"],
                "question_type": "close_end_QA",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote JSONL: {out_path} ({len(sampled_df)} lines)")


if __name__ == "__main__":
    main()
