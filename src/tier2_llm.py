import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class Tier2ReasoningEngine:
    def __init__(self, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"):
        """
        Initializes the Tier 2 LLM. 
        Uses 4-bit quantization to fit on consumer GPUs.
        """
        print(f"Loading Tier 2 LLM ({model_id}) in 4-bit precision...")
        
        # 4-bit Quantization Config to optimize computational costs
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        print("Tier 2 LLM loaded successfully.")

    def analyze_post(self, text):
        """
        Prompts the LLM for Chain-of-Thought reasoning and an ordinal severity label.
        """
        prompt = f"""You are an expert clinical psychologist. Analyze the following social media post for signs of depression.
Perform a Chain-of-Thought reasoning, then provide a final severity label. 
The label MUST be one of: [Minimal, Mild, Moderate, Severe].

Post: "{text}"

Response format:
Reasoning: <your step-by-step reasoning>
Label: <Minimal/Mild/Moderate/Severe>
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=150, 
            temperature=0.2, # Low temperature for more deterministic/factual output
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract label from response (simplified parsing)
        predicted_label = "unknown"
        for label in ["Minimal", "Mild", "Moderate", "Severe"]:
            if label.lower() in response.lower().split("label:")[-1]:
                predicted_label = label.lower()
                break
                
        return response, predicted_label

    def process_filtered_posts(self, df):
        """
        Processes a dataframe of Tier 1 filtered texts.
        """
        llm_labels = []
        reasonings = []
        
        for text in df['text']:
            # For testing without a GPU, you might want to mock this response
            try:
                full_response, label = self.analyze_post(text)
                llm_labels.append(label)
                reasonings.append(full_response)
            except Exception as e:
                # Fallback for systems without enough VRAM to run Llama during dev
                llm_labels.append("error")
                reasonings.append(str(e))
                
        df['tier2_label'] = llm_labels
        df['tier2_reasoning'] = reasonings
        return df
    