"""Fix known mishearings of my vocabulary in the display transcript.

Applied to text only, never to the Word list: metrics count what was spoken,
corrections only change how it's written.
"""
import re

# Lowercase phrase as heard -> correct spelling. Add a line when you spot a new mishearing.
CORRECTIONS: dict[str, str] = {
    "bemo": "BMO",
    "as see 300": "SC-300",
    "s c 300": "SC-300",
    "sc 300": "SC-300",
    "compti": "CompTIA",
    "computeria": "CompTIA",
    "security a i plus": "SecAI+",
    "security ai plus": "SecAI+",
    "security plus": "Security+",
    "still account protector": "Stale Account Detector",
    "anspiring": "aspiring",
    "i till foundation": "ITIL Foundation",
    "batca": "FATCA",
    "socks": "SOX",  # real word, but very unlikely in interview practice (see below)
}

# Acronyms you say. Their letter-by-letter forms are added automatically: "a m l" -> "AML"
ACRONYMS = ["AI", "AML", "IAM", "ITIL", "KYC", "MFA", "QI", "SOX", "SSO"]
CORRECTIONS.update({" ".join(acronym.lower()): acronym for acronym in ACRONYMS})

def apply_corrections(text: str) -> str:
    # Longest phrases first, so "security ai plus" is fixed before "security plus" can match.
    for heard in sorted(CORRECTIONS, key=len, reverse=True):
        pattern = r"\b" + re.escape(heard) + r"\b"
        text = re.sub(pattern, CORRECTIONS[heard], text, flags=re.IGNORECASE)
    return text