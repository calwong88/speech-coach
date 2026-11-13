"""Replace personal details with placeholders before text leaves this machine."""
import re

# Real text -> placeholder. Add anything else you never want to send.
REDACTIONS: dict[str, str] = {
    "Calvin Wong": "[NAME]",
    "Calvin": "[Name]",
    "Bank of Montreal": "[EMPLOYER]",
    "BMO": "[EMPLOYER]",
}


def redact(text: str, redactions: dict[str, str] = REDACTIONS) -> str:
    # Longest first, so "Calvin Wong" becomes on [NAME], not "[Name] Wong".
    for real in sorted(redactions, key=len, reverse=True):
        pattern = r"\b" + re.escape(real) + r"\b"
        text = re.sub(pattern, redactions[real], text, flags=re.IGNORECASE)
    return text
