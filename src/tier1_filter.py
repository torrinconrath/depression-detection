import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from src.constants import ORDINAL_ORDER

DEFAULT_PRETRAINED = "mrm8488/distilroberta-base-finetuned-suicide-depression"
DEFAULT_FINETUNED  = "models/tier1_classifier"
DEPRESSIVE_LABEL   = "LABEL_1"


class Tier1Filter:
    def __init__(self, model_path: str = DEFAULT_FINETUNED, threshold: float = 0.15):
        """
        Loads either the fine-tuned 4-class DSD classifier (preferred) or falls
        back to the pretrained binary DistilRoBERTa if the fine-tuned model is
        not found. The 4-class model filters out true minimal posts specifically;
        the binary fallback can only approximate this via threshold.
        """
        self.threshold      = threshold
        self.device         = 0 if torch.cuda.is_available() else -1
        self._is_multiclass = False

        import os
        if os.path.isdir(model_path):
            print(f"[Tier 1] Loading fine-tuned 4-class classifier: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model     = AutoModelForSequenceClassification.from_pretrained(model_path)
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.model.eval()
            self._is_multiclass = True
            print(f"[Tier 1] Ready (4-class) on {'GPU' if self.device == 0 else 'CPU'}.")
        else:
            print(f"[Tier 1] Fine-tuned classifier not found at '{model_path}'.")
            print(f"[Tier 1] Falling back to binary model. Run src/tier1_finetune.py first.")
            self.classifier = pipeline(
                "text-classification", model=DEFAULT_PRETRAINED,
                device=self.device, truncation=True, max_length=512,
            )
            print(f"[Tier 1] Ready (binary fallback) on {'GPU' if self.device == 0 else 'CPU'}. Threshold: p > {threshold}")

    def _predict_multiclass(self, texts: list[str], batch_size: int) -> list[str]:
        id2label = self.model.config.id2label
        preds    = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[i : i + batch_size],
                truncation=True, padding=True,
                max_length=384, return_tensors="pt",
            )
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                logits = self.model(**batch).logits
            preds.extend([id2label[idx] for idx in logits.argmax(dim=-1).cpu().tolist()])
        return preds

    def filter_posts(self, df: pd.DataFrame, batch_size: int = 32) -> tuple[pd.DataFrame, dict]:
        texts      = df["text"].tolist()
        start_time = time.time()

        if self._is_multiclass:
            preds      = self._predict_multiclass(texts, batch_size)
            passed_idx = [i for i, p in enumerate(preds) if p != "minimal"]
            scores     = [1.0 if preds[i] != "minimal" else 0.0 for i in passed_idx]
        else:
            results = []
            for i in range(0, len(texts), batch_size):
                results.extend(self.classifier(texts[i : i + batch_size]))
            probs      = [r["score"] if r["label"] == DEPRESSIVE_LABEL else 1.0 - r["score"] for r in results]
            passed_idx = [i for i, p in enumerate(probs) if p > self.threshold]
            scores     = [probs[i] for i in passed_idx]

        elapsed_ms  = (time.time() - start_time) * 1000
        elapsed_sec = elapsed_ms / 1000
        filtered_df = df.iloc[passed_idx].copy().reset_index(drop=True)
        filtered_df["tier1_score"] = scores

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
    