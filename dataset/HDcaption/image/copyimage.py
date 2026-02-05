#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import pandas as pd
from pathlib import Path

BASE_DIR = "/root/dataset/skin/Derm1M"
SRC_CSV  = "/root/dataset/skin/Derm1M/Derm1M_v2_pretrain_HD.csv"
DST_DIR  = os.path.join(BASE_DIR, "HDimage")

def main():
    df = pd.read_csv(SRC_CSV, dtype=str, keep_default_na=False)
    if "filename" not in df.columns:
        raise RuntimeError("Column 'filename' not found in CSV")

    os.makedirs(DST_DIR, exist_ok=True)

    copied, missing, skipped = 0, 0, 0
    seen = set()

    for fn in df["filename"].astype(str):
        fn = fn.strip()
        if not fn:
            continue
        if fn in seen:
            skipped += 1
            continue
        seen.add(fn)

        src = os.path.normpath(os.path.join(BASE_DIR, fn))
        # If 'fn' is absolute (rare), use it as-is
        if os.path.isabs(fn):
            src = fn

        if not os.path.isfile(src):
            missing += 1
            continue

        dst = os.path.normpath(os.path.join(DST_DIR, fn))
        Path(dst).parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src, dst)
        copied += 1

    print(f"Copied: {copied}")
    print(f"Missing: {missing}")
    print(f"Duplicates skipped: {skipped}")
    print(f"Destination root: {DST_DIR}")

if __name__ == "__main__":
    main()
