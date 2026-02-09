#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import math
import numpy as np
import pandas as pd

IN_CSV = "/root/dataset/skin/Derm1M/Derm1M_v2_pretrain.csv"
OUT_CSV = "/root/dataset/skin/Derm1M/Derm1M_balanced_10k.csv"
TARGET_TOTAL = 10_000
RANDOM_SEED = 42

# 1) Load
df = pd.read_csv(IN_CSV)

# 2) Validate required columns (keep ALL columns in the output)
required = ["caption", "disease_label", "body_location"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# 3) Normalize types
df["caption"] = df["caption"].astype(str)
df["disease_label"] = df["disease_label"].astype(str)
df["body_location"] = df["body_location"].astype(str)

# 4) Exclude specific label/body_location
lab_norm = df["disease_label"].str.strip().str.casefold()
loc_norm = df["body_location"].str.strip().str.casefold()
mask_exclude = (lab_norm == "no definitive diagnosis") | (loc_norm == "no body location information")
df = df[~mask_exclude].copy()

# 5) Filter: (a) no CJK characters in caption (b) caption has > 15 words
cjk_pattern = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufa6d]")
no_cjk = ~df["caption"].str.contains(cjk_pattern)
word_counts = df["caption"].str.split().str.len()
long_en = no_cjk & (word_counts > 15)

df_filt = df[long_en & df["caption"].notna() & df["disease_label"].notna()].copy()
if df_filt.empty:
    raise RuntimeError("No rows left after filtering (English-only, >15 words, exclusions applied).")

# 6) Class balancing plan
counts = df_filt["disease_label"].value_counts()
labels = counts.index.tolist()
n_classes = len(labels)
if n_classes == 0:
    raise RuntimeError("No disease_label categories after filtering.")

base = TARGET_TOTAL // n_classes
targets = {lab_name: int(min(base, counts[lab_name])) for lab_name in labels}

# Distribute remainder proportionally to remaining capacity
remainder = TARGET_TOTAL - sum(targets.values())
if remainder > 0:
    extra_capacity = {lab_name: int(max(0, counts[lab_name] - targets[lab_name])) for lab_name in labels}
    total_extra = sum(extra_capacity.values())
    if total_extra > 0:
        # Proportional floor
        provisional = {
            lab_name: int(math.floor(remainder * (extra_capacity[lab_name] / total_extra)))
            for lab_name in labels
        }
        allocated = sum(provisional.values())
        leftover = remainder - allocated

        # Greedy on largest fractional parts (respect capacity)
        if leftover > 0:
            fracs = {
                lab_name: (remainder * (extra_capacity[lab_name] / total_extra) - provisional[lab_name])
                for lab_name in labels
            }
            for lab_name in sorted(fracs, key=fracs.get, reverse=True):
                if leftover <= 0:
                    break
                if provisional[lab_name] < extra_capacity[lab_name]:
                    provisional[lab_name] += 1
                    leftover -= 1

        # Merge
        for lab_name in labels:
            targets[lab_name] += provisional[lab_name]

# Final clamp (safety)
for lab_name in labels:
    targets[lab_name] = int(min(targets[lab_name], counts[lab_name]))

planned_total = sum(targets.values())
if planned_total == 0:
    raise RuntimeError("Sampling plan produced zero rows. Check filters and data availability.")

# 7) Sample per class (preserve ALL original columns)
rng = np.random.RandomState(RANDOM_SEED)
parts = []
for lab_name, n_take in targets.items():
    if n_take <= 0:
        continue
    g = df_filt[df_filt["disease_label"] == lab_name]
    if n_take >= len(g):
        parts.append(g)
    else:
        parts.append(g.sample(n=n_take, random_state=int(rng.randint(0, 2**31 - 1))))

if not parts:
    raise RuntimeError("No samples could be drawn. Check filtering and targets.")

result = pd.concat(parts, axis=0, ignore_index=True)

# 8) Shuffle and trim to the requested total (or available maximum)
max_rows = min(TARGET_TOTAL, len(result))
result = result.sample(frac=1.0, random_state=RANDOM_SEED).head(max_rows).reset_index(drop=True)

# 9) Save
result.to_csv(OUT_CSV, index=False)

# 10) Report
print(f"Wrote: {OUT_CSV}")
print(f"Rows: {len(result)} (target: {TARGET_TOTAL})")
print("Class distribution (top 20):")
print(result["disease_label"].value_counts().head(20))
