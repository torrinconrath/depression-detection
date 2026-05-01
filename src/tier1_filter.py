"""
tier1_filter.py — Binary Sentinel Filter (Tier 1)

Uses mrm8488/distilroberta-base-finetuned-suicide-depression, a pretrained binary
DistilRoBERTa model, as a recall gate. No fine-tuning is required or performed.

Design rationale:
    Tier 1's only job is to ask "might this person need attention?" and pass anything
    uncertain to the LLM. Severity classification (minimal/mild/moderate/severe) is
    exclusively Tier 2's responsibility.

    A 4-class classifier at Tier 1 was considered but rejected: argmax has no threshold
    mechanism, so a post scored [minimal=0.40, mild=0.35, moderate=0.15, severe=0.10]
    gets silently discarded even though 60% of the probability mass sits on at-risk
    classes. The binary model with p > 0.15 instead asks whether any depressive signal
    is present, passing borderline cases through to the LLM rather than dropping them.

    At p > 0.15 the filter achieves ~97% severe recall while still removing the majority
    of clearly non-depressive posts, reducing unnecessary LLM compute.
"""

import time
import torch
import pandas as pd
from transformers import pipeline

BINARY_MODEL   = "mrm8488/distilroberta-base-finetuned-suicide-depression"
DEPRESSIVE_IDX = "LABEL_1"   # label assigned to the depressive class by this model


class Tier1Filter:
    def __init__(self, threshold: float = 0.15):
        """
        Args:
            threshold: minimum depressive-class probability to pass a post to Tier 2.
                       Lower = higher recall (fewer at-risk posts missed).
                       Default 0.15 targets 97%+ severe recall.
        """
        self.threshold = threshold
        device = 0 if torch.cuda.is_available() else -1

        print(f"[Tier 1] Loading binary classifier: {BINARY_MODEL}")
        self.classifier = pipeline(
            "text-classification",
            model=BINARY_MODEL,
            device=device,
            truncation=True,
            max_length=512,
        )
        print(
            f"[Tier 1] Ready on {'GPU' if device == 0 else 'CPU'}. "
            f"Threshold: p(depressive) > {threshold}"
        )

    def _depressive_prob(self, result: dict) -> float:
        """Return the probability assigned to the depressive class."""
        if result["label"] == DEPRESSIVE_IDX:
            return result["score"]
        return 1.0 - result["score"]

    def filter_posts(self, df: pd.DataFrame, batch_size: int = 32) -> tuple[pd.DataFrame, dict]:
        """
        Run binary classification on all posts and return those above the threshold.

        Returns:
            filtered_df: subset of df that passed the threshold, with a 'tier1_score'
                         column containing the raw depressive-class probability.
            metrics:     dict of throughput / latency / reduction statistics.
        """
        texts      = df["text"].tolist()
        start_time = time.time()

        results = []
        for i in range(0, len(texts), batch_size):
            results.extend(self.classifier(texts[i : i + batch_size]))

        probs      = [self._depressive_prob(r) for r in results]
        passed_idx = [i for i, p in enumerate(probs) if p > self.threshold]

        elapsed_ms  = (time.time() - start_time) * 1000
        elapsed_sec = elapsed_ms / 1000

        filtered_df = df.iloc[passed_idx].copy().reset_index(drop=True)
        filtered_df["tier1_score"] = [probs[i] for i in passed_idx]

        metrics = {
            "latency_per_post_ms":  elapsed_ms / len(texts),
            "throughput_per_sec":   len(texts) / elapsed_sec,
            "reduction_percentage": (1 - len(filtered_df) / len(df)) * 100,
            "original_count":       len(df),
            "passed_count":         len(filtered_df),
            "threshold":            self.threshold,
        }
        print(
            f"[Tier 1] {metrics['passed_count']}/{metrics['original_count']} posts passed "
            f"({100 - metrics['reduction_percentage']:.1f}% retained) | "
            f"{metrics['latency_per_post_ms']:.2f} ms/post | "
            f"{metrics['throughput_per_sec']:.0f} posts/sec"
        )
        return filtered_df, metrics
    