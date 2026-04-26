import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAP = {
    "minimum": "minimal",  # DSD dataset uses "minimum" spelling
    "minimal": "minimal",
    "mild": "mild",
    "moderate": "moderate",
    "severe": "severe",
}

ORDINAL_ORDER = ["minimal", "mild", "moderate", "severe"]


def load_dsd_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads the Depression Severity Dataset (DSD).
    Expected CSV columns: 'text', 'label'
    Normalises the 'minimum' -> 'minimal' label spelling used in this dataset.
    """
    if not os.path.exists(filepath):
        print(f"[Warning] DSD file not found at '{filepath}'. Returning empty DataFrame.")
        return pd.DataFrame(columns=["text", "label"])

    df = pd.read_csv(filepath)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"DSD CSV must have 'text' and 'label' columns. Found: {list(df.columns)}"
        )

    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].str.lower().str.strip().map(LABEL_MAP)

    before = len(df)
    df = df.dropna(subset=["label", "text"])
    df = df[df["text"] != ""]
    after = len(df)

    if before != after:
        print(f"[Info] Dropped {before - after} rows with missing/unknown labels or empty text.")

    return df.reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a stratified train/test split to preserve class proportions in both splits.
    This is critical given the dataset's severe class imbalance
    (minimal ~72.8%, severe ~7.9%).

    Args:
        df:           Full dataset DataFrame.
        test_size:    Fraction reserved for evaluation (default 20%).
        random_state: Seed for reproducibility.

    Returns:
        (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"[Split] Train: {len(train_df)} posts | Test: {len(test_df)} posts (stratified)")
    return train_df, test_df


def load_erisk_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads the eRisk JSON dataset.
    Extracts submission bodies and comments per user.
    Labels are 'unknown' unless provided separately.
    """
    if not os.path.exists(filepath):
        print(f"[Warning] eRisk file not found at '{filepath}'. Returning empty DataFrame.")
        return pd.DataFrame(columns=["author", "text", "label"])

    with open(filepath, "r") as f:
        data = json.load(f)

    records = []
    for entry in data:
        author = entry.get("author", "unknown")
        body = entry.get("body", "").strip()
        if body:
            records.append({"author": author, "text": body, "label": "unknown"})
        for comment in entry.get("comments", []):
            comment_body = comment.get("body", "").strip()
            if comment_body:
                records.append({
                    "author": comment.get("author", author),
                    "text": comment_body,
                    "label": "unknown",
                })

    df = pd.DataFrame(records)
    print(f"[Info] Loaded {len(df)} text entries from eRisk dataset.")
    return df


def get_dummy_data() -> pd.DataFrame:
    """Returns a small labeled dummy dataset for smoke-testing the pipeline."""
    return pd.DataFrame({
        "text": [
            "I've been feeling absolutely terrible and hopeless for weeks. Nothing helps.",
            "Just had a wonderful lunch with old friends, feeling grateful.",
            "Stressed about exams but I have a plan and I'm managing okay.",
            "I can't get out of bed anymore. Everything feels meaningless and dark.",
            "Feeling a bit low today but tomorrow is a new day.",
            "I was diagnosed last month and the medication seems to be helping a little.",
            "Life is good. Went hiking this weekend and it was amazing.",
            "I don't see the point of anything. I've stopped eating and sleeping properly.",
        ],
        "label": ["severe", "minimal", "mild", "severe", "mild", "moderate", "minimal", "severe"],
    })


def print_label_distribution(df: pd.DataFrame, dataset_name: str = "Dataset") -> None:
    """Prints the class distribution of a labeled DataFrame."""
    if "label" not in df.columns:
        print(f"[Warning] No 'label' column in {dataset_name}.")
        return

    counts = df["label"].value_counts()
    total = len(df)
    print(f"\n{dataset_name} Label Distribution ({total} total):")
    for label in ORDINAL_ORDER:
        count = counts.get(label, 0)
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {label:<10} | {count:>4} ({pct:5.1f}%) {bar}")
        