import time
import torch
import pandas as pd
from transformers import pipeline

# DistilRoBERTa fine-tuned for depression/suicide classification on social media.
# LABEL_0 = non-depressive, LABEL_1 = depressive
# https://huggingface.co/mrm8488/distilroberta-base-finetuned-suicide-depression
DEFAULT_MODEL    = "mrm8488/distilroberta-base-finetuned-suicide-depression"
DEPRESSIVE_LABEL = "LABEL_1"


class Tier1Filter:
    def __init__(self, model_name: str = DEFAULT_MODEL, threshold: float = 0.3):
        """
        Lightweight sentinel filter. Scores every post for depressive content and
        passes anything above `threshold` to Tier 2.

        Threshold is kept low (0.3) to maximise System Recall — missing a severe
        case is worse than passing a false positive to Tier 2.
        """
        self.threshold = threshold
        self.device    = 0 if torch.cuda.is_available() else -1

        print(f"[Tier 1] Loading: {model_name}")
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            device=self.device,
            truncation=True,
            max_length=512,
        )
        print(f"[Tier 1] Ready on {'GPU' if self.device == 0 else 'CPU'}. Threshold: p > {threshold}")

    def _depressive_prob(self, result: dict) -> float:
        """Returns the probability of the depressive class regardless of which label was top."""
        if result["label"] == DEPRESSIVE_LABEL:
            return result["score"]
        return 1.0 - result["score"]   # top label was non-depressive

    def filter_posts(self, df: pd.DataFrame, batch_size: int = 32) -> tuple[pd.DataFrame, dict]:
        """
        Score all posts, return those above threshold with Tier 1 metrics.

        Returns:
            filtered_df  — posts that passed, with added 'tier1_score' column
            metrics      — latency_per_post_ms, reduction_percentage, original_count, passed_count
        """
        texts      = df["text"].tolist()
        start_time = time.time()

        results = []
        for i in range(0, len(texts), batch_size):
            results.extend(self.classifier(texts[i : i + batch_size]))

        elapsed_ms      = (time.time() - start_time) * 1000
        probs           = [self._depressive_prob(r) for r in results]
        passed_idx      = [i for i, p in enumerate(probs) if p > self.threshold]

        filtered_df = df.iloc[passed_idx].copy().reset_index(drop=True)
        filtered_df["tier1_score"] = [probs[i] for i in passed_idx]

        metrics = {
            "latency_per_post_ms":  elapsed_ms / len(texts),
            "reduction_percentage": (1 - len(filtered_df) / len(df)) * 100,
            "original_count":       len(df),
            "passed_count":         len(filtered_df),
        }

        print(
            f"[Tier 1] {metrics['passed_count']}/{metrics['original_count']} posts passed "
            f"({100 - metrics['reduction_percentage']:.1f}% retained) | "
            f"{metrics['latency_per_post_ms']:.2f} ms/post"
        )
        return filtered_df, metrics
    