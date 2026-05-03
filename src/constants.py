BASE_MODEL_ID  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
BASE_FILTER_ID = "distilbert-base-uncased"
# BASE_FILTER_ID = "mrm8488/distilroberta-base-finetuned-suicide-depression"
ORDINAL_ORDER  = ["minimal", "mild", "moderate", "severe"]

SYSTEM_PROMPT = (
    "You are an expert clinical psychologist specialising in depression assessment.\n"
    "Analyse the social media post below and classify the author's depression severity.\n\n"
    "First provide step-by-step clinical reasoning citing specific evidence from the post.\n"
    "Then state the final label.\n\n"
    "Severity scale:\n"
    "- Minimal:  Little to no depressive indicators. Normal daily functioning.\n"
    "- Mild:     Occasional low mood or stress. Some negative affect but generally coping.\n"
    "- Moderate: Persistent low mood, loss of interest, some functional impairment.\n"
    "- Severe:   Intense hopelessness, anhedonia, significant impairment, possible suicidal ideation.\n\n"
    "Response format (follow exactly):\n"
    "Reasoning: <2-4 sentences of clinical reasoning>\n"
    "Label: <Minimal|Mild|Moderate|Severe>"
)
