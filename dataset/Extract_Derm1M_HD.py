import os
import json
import pandas as pd
import re

CSV_PATH = r"/mnt/d/skinalor/dataset/skin/Derm1M/Derm1M_v2_pretrain.csv"
CSV_PATH = CSV_PATH.replace("\\", "/")




def main():
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    df[["filename", "caption"]] = df[["filename", "caption"]].fillna("").astype(str)
    df["caption"] = df["caption"].str.strip()
    df["filename"] = df["filename"].str.strip()
    df_no_zh = df[~df["caption"].str.contains(r"[\u4e00-\u9fff]", na=False)]


    # Normalize captions
    s = df_no_zh["caption"].astype(str).fillna("").str.strip()

    # Counts
    word_cnt = s.str.split().str.len()
    char_cnt_with_space = s.str.len()
    char_cnt_no_space = s.str.replace(r"\s+", "", regex=True).str.len()

    # Indices
    w_min_idx, w_max_idx = word_cnt.idxmin(), word_cnt.idxmax()
    c_min_idx, c_max_idx = char_cnt_with_space.idxmin(), char_cnt_with_space.idxmax()

    def maybe_filename(idx):
        return df_no_zh.loc[idx, "filename"] if "filename" in df_no_zh.columns else None

    # Print results
    print("=== WORD COUNT ===")
    print(f"Shortest: {int(word_cnt.loc[w_min_idx])} words | index={w_min_idx} | filename={maybe_filename(w_min_idx)}")
    print(f"Longest : {int(word_cnt.loc[w_max_idx])} words | index={w_max_idx} | filename={maybe_filename(w_max_idx)}")

    print("\n=== CHAR COUNT (with spaces) ===")
    print(
        f"Shortest: {int(char_cnt_with_space.loc[c_min_idx])} chars | index={c_min_idx} | filename={maybe_filename(c_min_idx)}")
    print(
        f"Longest : {int(char_cnt_with_space.loc[c_max_idx])} chars | index={c_max_idx} | filename={maybe_filename(c_max_idx)}")

    print("\n=== CHAR COUNT (no spaces) ===")
    print(f"Shortest: {int(char_cnt_no_space.loc[c_min_idx])} chars_no_space at index={c_min_idx}")
    print(f"Longest : {int(char_cnt_no_space.loc[c_max_idx])} chars_no_space at index={c_max_idx}")


    print("\n=== AVERAGES ===")
    print(f"Avg words per caption: {word_cnt.mean():.2f}")
    print(f"Avg chars per caption (with spaces): {char_cnt_with_space.mean():.2f}")
    print(f"Avg chars per caption (no spaces): {char_cnt_no_space.mean():.2f}")

if __name__ == "__main__":
    main()