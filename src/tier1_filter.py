"""
tier1_filter.py — Sentinel Filter (Tier 1)

Loads the fine-tuned DistilBERT classifier (trained via tier1_finetune.py)
as a recall gate. The model was fine-tuned on the 4-class DSD training split.

Design rationale:
    Tier 1's only job is to ask "might this person need attention?" and pass anything
    uncertain to the LLM. 

    Instead of relying on argmax (which silently drops posts if probability is split 
    across mild/moderate/severe), this module sums the probability of the three 
    "at-risk" classes. If P(mild) + P(moderate) + P(severe) > threshold, it passes 
    to Tier 2.

    Fine-tuning on DSD (same domain, same distribution) rather than using a generic
    pretrained suicide/depression model avoids domain shift and appropriately weights
    the severe class.
"""

import os
import time
import torch
import pandas as pd
from transformers import pipeline

FINETUNED_MODEL_DIR = "models/tier1_filter"


class Tier1Filter:
    def __init__(self, threshold: float = 0.10, model_dir: str = FINETUNED_MODEL_DIR):
        """
        Args:
            threshold: minimum summed at-risk probability to pass a post to Tier 2.
                       Since the model predicts 4 classes, this threshold looks at 
                       P(mild) + P(moderate) + P(severe).
            model_dir: path to the fine-tuned classifier saved by tier1_finetune.py.
        """
        self.threshold = threshold

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"[Tier 1] Fine-tuned model not found at '{model_dir}'.\n"
                f"[Tier 1] Run: python -m src.tier1_finetune"
            )

        device = 0 if torch.cuda.is_available() else -1
        print(f"[Tier 1] Loading fine-tuned classifier from '{model_dir}'")
        self.classifier = pipeline(
            "text-classification",
            model=model_dir,
            device=device,
            truncation=True,
            max_length=256,
            top_k=None,  # CRITICAL: Forces pipeline to return scores for all 4 classes
        )
        print(
            f"[Tier 1] Ready on {'GPU' if device == 0 else 'CPU'}. "
            f"Threshold: P(not minimal) > {threshold}"
        )

    def _at_risk_prob(self, result: list) -> float:
        """Sum the probabilities of mild, moderate, and severe."""
        # result is a list of dicts: [{'label': 'mild', 'score': 0.4}, ...]
        at_risk_prob = 0.0
        for class_score in result:
            if class_score["label"] != "minimal":
                at_risk_prob += class_score["score"]
        return at_risk_prob

    def filter_posts(self, df: pd.DataFrame, batch_size: int = 32) -> tuple[pd.DataFrame, dict]:
        """
        Run classification on all posts and return those above the at-risk threshold.

        Returns:
            filtered_df: subset of df that passed the threshold, with a 'tier1_score'
                         column containing the summed at-risk probability.
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
    