#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path

IN_PATH  = Path("/home/root/dataset/skin/Derm1M/Derm1M_train.jsonl")
OUT_PATH = Path("/home/root/dataset/skin/Derm1M/Derm1M_train_1word.jsonl")


ONE_WORD = re.compile(r"^[A-Za-z]+(?:-[A-Za-z]+)*$")

total = kept = skipped = 0

with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        total += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue

        lab = obj.get("disease_label", "")
        if not isinstance(lab, str):
            lab = str(lab)

        lab_norm = lab.strip()
        if ONE_WORD.match(lab_norm):
            json.dump(obj, fout, ensure_ascii=False)
            fout.write("\n")
            kept += 1
        else:
            skipped += 1

print(f"Input lines : {total}")
print(f"Kept (1 word): {kept}")
print(f"Skipped      : {skipped}")
print(f"Wrote to     : {OUT_PATH}")
