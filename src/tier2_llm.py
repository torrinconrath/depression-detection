"""
tier2_llm.py — LLM Reasoning Engine (Tier 2)

Loads the fine-tuned Llama 3.1-8B-Instruct + QLoRA adapter and classifies posts
that passed the Tier 1 binary filter into the four DSD severity levels.

The model generates its own clinical reasoning before the label (CoT via system
prompt). Generation uses greedy decoding — deterministic and more consistent than
low-temperature sampling for a classification task.
"""

import re
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.constants import BASE_MODEL_ID, ORDINAL_ORDER, SYSTEM_PROMPT


class Tier2ReasoningEngine:
    def __init__(self, adapter_path: str = "models/tier2_adapter"):
        if not torch.cuda.is_available():
            raise EnvironmentError("[Tier 2] A CUDA GPU is required.")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        print(f"[Tier 2] Loading base model: {BASE_MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        self.tokenizer.pad_token    = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto",
            low_cpu_mem_usage=True, torch_dtype=torch.float16,
        )
        print(f"[Tier 2] Loading adapter: {adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        print("[Tier 2] Ready.")

    def _parse_label(self, response: str) -> str:
        """
        Extract the severity label from the model's response.

        The model generates reasoning before the label, so we look for the explicit
        'Label: <word>' pattern and take the LAST match — any label word mentioned
        during the reasoning section is skipped in favour of the final verdict.
        Falls back to scanning the last few lines, then the full response.
        """
        # Primary: explicit Label: tag — last match skips reasoning mentions
        matches = re.findall(r"label\s*:\s*(\w+)", response, re.IGNORECASE)
        if matches:
            candidate = matches[-1].lower()
            if candidate in ORDINAL_ORDER:
                return candidate

        # Secondary: scan the last 3 lines where the label most likely appears
        tail = "\n".join(response.strip().splitlines()[-3:]).lower()
        for label in reversed(ORDINAL_ORDER):   # severe → minimal (higher severity first)
            if label in tail:
                return label

        # Last resort: full response scan
        return next((l for l in ORDINAL_ORDER if l in response.lower()), "unknown")

    def analyze_post(self, text: str) -> tuple[str, str]:
        """
        Run inference on a single post.

        Returns:
            (reasoning, label) — the full model response and the parsed severity label.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Post: \"{text}\""},
        ]
        inputs = self.tokenizer(
            self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False),
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=300,     # headroom for 2-4 sentence reasoning + label line
                do_sample=False,        # greedy decoding: deterministic, better for classification
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()
        return response, self._parse_label(response)

    def process_filtered_posts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run Tier 2 inference over all posts that passed the Tier 1 filter."""
        df = df.copy()
        labels, reasonings = [], []

        for idx, text in enumerate(df["text"], start=1):
            try:
                reasoning, label = self.analyze_post(text)
                labels.append(label)
                reasonings.append(reasoning)
                print(f"[Tier 2] ({idx}/{len(df)}) → {label}")
            except Exception as e:
                labels.append("error")
                reasonings.append(str(e))
                print(f"[Tier 2] ({idx}/{len(df)}) → ERROR: {e}")

        df["tier2_label"]     = labels
        df["tier2_reasoning"] = reasonings

        n_bad = sum(1 for l in labels if l in ("unknown", "error"))
        if n_bad:
            print(f"[Tier 2] Warning: {n_bad}/{len(df)} posts returned unparseable labels.")
        return df
    