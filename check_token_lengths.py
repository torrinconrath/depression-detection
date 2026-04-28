"""
check_token_lengths.py — Analyse token lengths across the DSD dataset.

Usage:
    python check_token_lengths.py
    python check_token_lengths.py --data data/dsd.csv --model meta-llama/Meta-Llama-3.1-8B-Instruct
"""

import argparse
import pandas as pd
from transformers import AutoTokenizer

PERCENTILES   = [50, 75, 90, 95, 99, 100]
DEFAULT_DATA  = "data/dsd.csv"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def analyze(data_path: str, model_name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df = pd.read_csv(data_path)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)

    lengths = df["text"].apply(lambda t: len(tokenizer.encode(t, add_special_tokens=False)))

    print("=" * 48)
    print("  Token length distribution")
    print("=" * 48)
    for p in PERCENTILES:
        val   = int(lengths.quantile(p / 100))
        bar   = "█" * min(40, val // 50)
        label = "max" if p == 100 else f"p{p} "
        print(f"  {label:<4} | {val:>5} tokens  {bar}")

    def ceil128(n): return ((n + 127) // 128) * 128
    print(f"\n  Full coverage → max_length = {ceil128(int(lengths.max()))}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default=DEFAULT_DATA,  help="Path to dsd.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model ID for tokenizer")
    args = parser.parse_args()
    analyze(args.data, args.model)
    