import re


def clean_text(text: str) -> str:
    """
    Normalize extracted document text while preserving
    semantic content.

    This function intentionally avoids aggressive cleaning
    because punctuation, headings, numbers, and document
    structure may be important during retrieval.
    """

    if not text:
        return ""

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null bytes.
    text = text.replace("\x00", "")

    # Replace non-breaking spaces.
    text = text.replace("\u00a0", " ")

    # Replace repeated horizontal whitespace.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Remove spaces before line breaks.
    text = re.sub(
        r" +\n",
        "\n",
        text,
    )

    # Limit excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()