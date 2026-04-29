import random

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

# Multiple stubs per class so the model learns the reasoning space rather than
# a single memorised template. format_example picks one at random per example,
# meaning each training pass surfaces different clinical framings for the same label.
COT_STUBS = {
    "Minimal": [
        "The post contains no prominent depressive language. The author appears to be functioning normally with no marked signs of distress. Affect and tone are broadly neutral or positive.",
        "Language is matter-of-fact with no signs of emotional dysregulation. The author engages with daily life without expressing hopelessness or withdrawal. No depressive indicators are present.",
        "The post reflects routine concerns or neutral observations. There is no evidence of anhedonia, persistent low mood, or functional impairment. Overall presentation is consistent with normal baseline functioning.",
        "Tone is stable and grounded. The author shows no signs of cognitive distortion, self-blame, or emotional numbness. Content is consistent with typical daily experience.",
    ],
    "Mild": [
        "The post contains some indicators of low mood or stress, but the author appears to be coping. Negative affect is present but not pervasive. There is no evidence of significant functional impairment.",
        "The author expresses frustration or transient sadness but retains perspective and self-efficacy. Distress appears situational rather than pervasive. No signs of functional breakdown.",
        "Some negative affect is evident, but the author demonstrates resilience and capacity to continue daily activities. The emotional tone is subdued without crossing into persistent hopelessness.",
        "The language suggests mild emotional burden — stress, worry, or low energy — but the author is still engaged with their environment. Impairment, if any, is minimal and likely temporary.",
    ],
    "Moderate": [
        "The post shows persistent low mood and reduced interest or energy. There are signs of some functional impairment in daily life. The language suggests ongoing distress beyond typical stress responses.",
        "The author expresses sustained negative affect with signs of withdrawal or loss of motivation. Functioning appears impaired in at least one domain. Distress is not situational but recurring.",
        "Language reflects cognitive patterns consistent with moderate depression: self-criticism, reduced pleasure, and difficulty maintaining normal routines. The author is struggling but not expressing acute crisis.",
        "There is evidence of anhedonia and persistent emotional heaviness. The author references difficulty coping over time, suggesting the distress is entrenched rather than transient.",
    ],
    "Severe": [
        "The post contains explicit or implicit indicators of hopelessness and loss of will to continue. The author's language reflects profound anhedonia and possible suicidal ideation. Functional breakdown is evident.",
        "The author expresses intense, pervasive despair with no visible coping resources. Language includes markers of suicidal thinking or complete disengagement from life. This presentation warrants urgent attention.",
        "Severe hopelessness and self-worthlessness dominate the post. The author shows signs of crisis-level distress — inability to function, social isolation, and possible risk to self. Immediate support is indicated.",
        "The post reflects a complete collapse of functioning and meaning. There are direct or indirect references to not wanting to exist or seeing no future. Affect is blunted or overwhelmingly negative with no relief.",
        "Language signals acute psychological crisis: the author feels trapped, burdensome, or invisible. Combined with anhedonia and functional impairment, this post is consistent with severe depression requiring intervention.",
    ],
}


def get_cot_stub(label: str) -> str:
    """Returns a randomly sampled stub for the given label capitalised form e.g. 'Severe'."""
    stubs = COT_STUBS.get(label, ["Insufficient information to determine severity."])
    return random.choice(stubs)
