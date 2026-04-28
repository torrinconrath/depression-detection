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
            BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto"
        )
        print(f"[Tier 2] Loading adapter: {adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        print("[Tier 2] Ready.")

    def _parse_label(self, response: str) -> str:
        match = re.search(r"label\s*:\s*(\w+)", response, re.IGNORECASE)
        if match and match.group(1).lower() in ORDINAL_ORDER:
            return match.group(1).lower()
        return next((l for l in ORDINAL_ORDER if l in response.lower()), "unknown")

    def analyze_post(self, text: str) -> tuple[str, str]:
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
                **inputs, max_new_tokens=200, temperature=0.1,
                do_sample=True, repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()
        return response, self._parse_label(response)

    def process_filtered_posts(self, df: pd.DataFrame) -> pd.DataFrame:
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
    