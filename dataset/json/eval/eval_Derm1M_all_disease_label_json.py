#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import pandas as pd
from pathlib import Path

# Inputs/outputs
CSV_PATH = r"/home/root/dataset/skin/Derm1M/Derm1M_v2_pretrain_HD.csv".replace("\\", "/")
OUT_PATH = os.path.join(os.path.dirname(CSV_PATH), "eval_Derm1M_train_json_1k.jsonl")

N_SAMPLES = 1000
SEED = 42


def allocate_quota(counts: pd.Series, total: int, seed: int = 42) -> dict:
    """
    Distribute 'total' samples across classes near-uniformly, without exceeding availability.
    Tries to include every class at least once if feasible.
    """
    rng = random.Random(seed)
    labels = list(counts.index)
    rng.shuffle(labels)  # tie-break randomness

    avail = {lbl: int(counts[lbl]) for lbl in labels}
    alloc = {lbl: 0 for lbl in labels}

    total_available = sum(avail.values())
    if total_available <= total:
        # Not enough data: take everything.
        return avail.copy()

    # If we can include all classes at least once, do so.
    if len(labels) <= total:
        for lbl in labels:
            if avail[lbl] > 0:
                alloc[lbl] = 1
        remaining = total - sum(alloc.values())
    else:
        # More classes than budget: pick classes with the largest availability.
        top = sorted(labels, key=lambda x: avail[x], reverse=True)[:total]
        for lbl in top:
            alloc[lbl] = 1
        return alloc

    # Initial uniform fill up to floor(total / K)
    k = len(labels)
    base = max(0, (total // k) - 1)  # we already gave 1 to each
    if base > 0:
        for lbl in labels:
            add = min(base, avail[lbl] - alloc[lbl])
            alloc[lbl] += add
        remaining = total - sum(alloc.values())

    # Round-robin add remaining one-by-one, always favoring labels with lower current alloc
    # and with spare capacity.
    while remaining > 0:
        progressed = False
        # Sort by current allocation (ascending) to keep near-uniform
        labels.sort(key=lambda x: (alloc[x], -avail[x]))
        for lbl in labels:
            if remaining <= 0:
                break
            if alloc[lbl] < avail[lbl]:
                alloc[lbl] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break  # no more capacity anywhere

    return alloc


def main():
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    need_cols = {"filename", "disease_label"}
    missing = need_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    df["filename"] = df["filename"].astype(str).str.strip()
    df["disease_label"] = df["disease_label"].astype(str).str.strip()
    df = df[(df["filename"] != "") & (df["disease_label"] != "")]
    if df.empty:
        raise RuntimeError("No valid rows after basic cleaning.")

    counts = df["disease_label"].value_counts().sort_index()
    quota = allocate_quota(counts, N_SAMPLES, seed=SEED)

    # Sample per class
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
    if len(sampled_df) > N_SAMPLES:
        sampled_df = sampled_df.iloc[:N_SAMPLES].copy()

    # Print class distribution
    sampled_counts = sampled_df["disease_label"].value_counts().sort_index()
    print("Selected diseases and counts:")
    for lbl, cnt in sampled_counts.items():
        print(f"- {lbl}: {cnt}")
    print(f"Total: {len(sampled_df)} rows, {sampled_counts.shape[0]} unique diseases")

    # Write JSONL
    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in sampled_df.iterrows():
            rec = {
                "image": row["filename"],
                "caption": row["caption"],
                "answer": row["disease_label"],
                "question_type": "close_end_QA"
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote JSONL: {out_path} ({len(sampled_df)} lines)")


if __name__ == "__main__":
    main()
