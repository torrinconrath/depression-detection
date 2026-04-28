import os
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAP = {
    "minimum":  "minimal",  # DSD dataset uses "minimum" spelling
    "minimal":  "minimal",
    "mild":     "mild",
    "moderate": "moderate",
    "severe":   "severe",
}

ORDINAL_ORDER = ["minimal", "mild", "moderate", "severe"]


def load_dsd_dataset(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at '{filepath}'.")

    df = pd.read_csv(filepath)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"CSV must have 'text' and 'label' columns. Found: {list(df.columns)}")

    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].str.lower().str.strip().map(LABEL_MAP)

    before = len(df)
    df = df.dropna(subset=["label", "text"])
    df = df[df["text"] != ""].reset_index(drop=True)

    if before - len(df):
        print(f"[Data] Dropped {before - len(df)} rows with missing/unknown labels or empty text.")

    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split — preserves class proportions across train/test.
    Critical given the heavy imbalance (minimal ~72.8%, severe ~7.9%).
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=random_state
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def print_label_distribution(df: pd.DataFrame, name: str = "Dataset") -> None:
    total  = len(df)
    counts = df["label"].value_counts()
    print(f"\n{name} ({total} posts):")
    for label in ORDINAL_ORDER:
        n   = counts.get(label, 0)
        pct = n / total * 100 if total else 0
        print(f"  {label:<10} | {n:>4} ({pct:5.1f}%) {'█' * int(pct / 2)}")
        