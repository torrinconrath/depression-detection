"""
tier1_filter.py — Binary Sentinel Filter (Tier 1)

Loads the fine-tuned DistilBERT binary classifier (trained via tier1_finetune.py)
as a recall gate. The model was fine-tuned on the DSD training split with:
    Negative (label 0): minimal — little to no depressive signal
    Positive (label 1): mild + moderate + severe — any at-risk signal

Design rationale:
    Tier 1's only job is to ask "might this person need attention?" and pass anything
    uncertain to the LLM. Severity classification (minimal/mild/moderate/severe) is
    exclusively Tier 2's responsibility.

    A 4-class classifier at Tier 1 was considered but rejected: argmax has no threshold
    mechanism, so a post scored [minimal=0.40, mild=0.35, moderate=0.15, severe=0.10]
    gets silently discarded even though 60% of the probability mass sits on at-risk
    classes. The binary model with a sub-0.5 threshold instead asks whether any
    depressive signal is present, passing borderline cases through to the LLM.

    Fine-tuning on DSD (same domain, same distribution) rather than using a generic
    pretrained suicide/depression model avoids two failure modes:
      (1) Domain shift — generic models trained on crisis text may not generalise to
          the mild/moderate language that dominates DSD's at-risk classes.
      (2) Label misalignment — a model trained on binary depressive/non-depressive
          labels has no concept of "minimal" as a clinical category; ours does.

    Clinical class split rationale:
      Mild is included in the positive class because mild posts share surface-level
      language with minimal posts but represent early-onset depression where LLM
      reasoning adds the most value. Discarding them at Tier 1 would be clinically
      unsafe — these are exactly the borderline cases the cascade exists to handle.
"""

import os
import time
import torch
import pandas as pd
from transformers import pipeline

FINETUNED_MODEL_DIR = "models/tier1_filter"
AT_RISK_LABEL       = "at_risk"   # id2label set during fine-tuning


class Tier1Filter:
    def __init__(self, threshold: float = 0.3, model_dir: str = FINETUNED_MODEL_DIR):
        """
        Args:
            threshold: minimum at-risk probability to pass a post to Tier 2.
                       Lower = higher recall (fewer at-risk posts missed).
                       Default 0.3 is conservative; fine-tuned model is well-calibrated
                       on DSD so a higher threshold is viable if precision matters more.
            model_dir: path to the fine-tuned binary classifier saved by tier1_finetune.py.
        """
        self.threshold = threshold

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"[Tier 1] Fine-tuned model not found at '{model_dir}'.\n"
                f"[Tier 1] Run: python -m src.tier1_finetune"
            )

        device = 0 if torch.cuda.is_available() else -1
        print(f"[Tier 1] Loading fine-tuned binary classifier from '{model_dir}'")
        self.classifier = pipeline(
            "text-classification",
            model=model_dir,
            device=device,
            truncation=True,
            max_length=256,
        )
        print(
            f"[Tier 1] Ready on {'GPU' if device == 0 else 'CPU'}. "
            f"Threshold: p(at_risk) > {threshold}"
        )

    def _at_risk_prob(self, result: dict) -> float:
        """Return the probability assigned to the at-risk (positive) class."""
        if result["label"] == AT_RISK_LABEL:
            return result["score"]
        return 1.0 - result["score"]

    def filter_posts(self, df: pd.DataFrame, batch_size: int = 32) -> tuple[pd.DataFrame, dict]:
        """
        Run binary classification on all posts and return those above the threshold.

        Returns:
            filtered_df: subset of df that passed the threshold, with a 'tier1_score'
                         column containing the raw at-risk probability.
            metrics:     dict of throughput / latency / reduction statistics.
        """
        texts      = df["text"].tolist()
        start_time = time.time()

        results = []
        for i in range(0, len(texts), batch_size):
            results.extend(self.classifier(texts[i : i + batch_size]))

        probs      = [self._at_risk_prob(r) for r in results]
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
    