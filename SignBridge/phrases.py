"""Small context-aware phrase layer (rule based, no cloud)."""

PAIR_RULES = [
    ({"help", "emergency"}, "I need help. This is an emergency."),
    ({"hurt", "doctor"}, "I am hurt. I need a doctor."),
]

SINGLE = {
    "hospital": "I need a hospital.",
    "water": "I need water.",
    "food": "I need food.",
    "urgent": "This is urgent.",
    "disagree": "I disagree.",
    "help": "I need help.",
    "doctor": "I need a doctor.",
    "emergency": "This is an emergency.",
    "hurt": "I am hurt.",
}


def build_sentence(sequence):
    remaining = []
    for s in sequence:
        if s not in remaining:
            remaining.append(s)
    parts = []
    for pair, sentence in PAIR_RULES:
        if pair.issubset(remaining):
            parts.append(sentence)
            remaining = [s for s in remaining if s not in pair]
    for s in remaining:
        if s in SINGLE:
            parts.append(SINGLE[s])
    return " ".join(parts)
