BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ORDINAL_ORDER = ["minimal", "mild", "moderate", "severe"]

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

COT_STUBS = {
    "Minimal":  "The post contains no prominent depressive language. The author appears to be functioning normally with no marked signs of distress. Affect and tone are broadly neutral or positive.",
    "Mild":     "The post contains some indicators of low mood or stress, but the author appears to be coping. Negative affect is present but not pervasive. There is no evidence of significant functional impairment.",
    "Moderate": "The post shows persistent low mood and reduced interest or energy. There are signs of some functional impairment in daily life. The language suggests ongoing distress beyond typical stress responses.",
    "Severe":   "The post contains strong indicators of hopelessness, anhedonia, or significant functional breakdown. The author's language suggests intense and pervasive distress. There may be implicit or explicit indicators of risk.",
}