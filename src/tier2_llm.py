"""
tier2_llm.py

Tier 2 LLM Reasoning Engine — inference using the QLoRA fine-tuned adapter.

Loads the fine-tuned LoRA adapter produced by tier2_finetune.py and runs
Chain-of-Thought depression severity classification on Tier 1 filtered posts.
"""

import re

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_ADAPTER_PATH = "models/tier2_adapter"

VALID_LABELS = ["minimal", "mild", "moderate", "severe"]

SYSTEM_PROMPT = """You are an expert clinical psychologist specialising in depression assessment.
Analyse the social media post below and classify the author's depression severity.

First provide step-by-step clinical reasoning citing specific evidence from the post.
Then state the final label.

Severity scale:
- Minimal: Little to no depressive indicators. Normal daily functioning.
- Mild: Occasional low mood or stress. Some negative affect but generally coping.
- Moderate: Persistent low mood, loss of interest, some functional impairment.
- Severe: Intense hopelessness, anhedonia, significant impairment, possible suicidal ideation.

Response format (follow exactly):
Reasoning: <2-4 sentences of clinical reasoning>
Label: <Minimal|Mild|Moderate|Severe>"""


class Tier2ReasoningEngine:
    def __init__(
        self,
        adapter_path: str = DEFAULT_ADAPTER_PATH,
        base_model_id: str = BASE_MODEL_ID,
    ):
        """
        Loads the fine-tuned QLoRA adapter on top of the Llama 3.1-8B base model.

        The adapter is produced by tier2_finetune.py. Run that script first.

        Args:
            adapter_path:   Path to the saved LoRA adapter directory.
            base_model_id:  HuggingFace ID of the base model (must match training).
        """
        if not torch.cuda.is_available():
            raise EnvironmentError(
                "[Tier 2] A CUDA GPU is required for 4-bit inference. "
                "Ensure you are running on a GPU machine."
            )

        print(f"[Tier 2] Loading base model: {base_model_id}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Required for batch generation

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )

        print(f"[Tier 2] Loading fine-tuned adapter from: {adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        print("[Tier 2] Tier 2 engine ready.")

    # ── Inference ──────────────────────────────────────────────────────────────

    def _build_prompt(self, text: str) -> str:
        """Formats a single post into the inference prompt using the Llama 3 chat template."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Post: \"{text}\""},
        ]
        # apply_chat_template adds the correct special tokens for Llama 3 instruct
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    def _parse_label(self, response: str) -> str:
        """
        Extracts the severity label from the model's raw output.
        Searches for 'Label: <value>' first, then falls back to scanning
        the full response for any valid label word.
        """
        match = re.search(r"label\s*:\s*(\w+)", response, re.IGNORECASE)
        if match:
            candidate = match.group(1).lower().strip()
            if candidate in VALID_LABELS:
                return candidate

        # Fallback: last valid label word found in the response
        found = None
        for label in VALID_LABELS:
            if label in response.lower():
                found = label
        return found if found else "unknown"

    def analyze_post(self, text: str) -> tuple[str, str]:
        """
        Runs Chain-of-Thought reasoning on a single post.

        Args:
            text: The social media post to analyse.

        Returns:
            (reasoning_and_label_response, predicted_label)
        """
        prompt = self._build_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,         # Near-deterministic for clinical consistency
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only newly generated tokens (exclude the echoed prompt)
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        predicted_label = self._parse_label(response)
        return response, predicted_label

    # ── Batch processing ───────────────────────────────────────────────────────

    def process_filtered_posts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs Tier 2 analysis on all posts that passed the Tier 1 filter.

        Args:
            df: DataFrame with at least a 'text' column (output of Tier 1).

        Returns:
            DataFrame with added 'tier2_label' and 'tier2_reasoning' columns.
        """
        df = df.copy()
        labels, reasonings = [], []
        total = len(df)

        print(f"[Tier 2] Analysing {total} filtered posts...")

        for idx, text in enumerate(df["text"], start=1):
            try:
                reasoning, label = self.analyze_post(text)
                labels.append(label)
                reasonings.append(reasoning)
                print(f"[Tier 2] ({idx}/{total}) → {label}")
            except Exception as e:
                labels.append("error")
                reasonings.append(str(e))
                print(f"[Tier 2] ({idx}/{total}) → ERROR: {e}")

        df["tier2_label"] = labels
        df["tier2_reasoning"] = reasonings

        unparseable = sum(1 for l in labels if l in ("unknown", "error"))
        if unparseable:
            print(f"[Tier 2] Warning: {unparseable}/{total} posts returned unparseable labels.")

        return df
    