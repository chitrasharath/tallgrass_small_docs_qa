from __future__ import annotations

import re
import unicodedata

_REPLACEMENTS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u2026": "...",
    }
)

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)
_DATE = (
    r"\d{4}-\d{2}-\d{2}"
    rf"|(?:{_MONTHS})\s+\d{{1,2}},\s+\d{{4}}"
    rf"|(?:{_MONTHS})\s+\d{{4}}"
    r"|(?:Spring|Summer|Fall|Autumn|Winter)\s+\d{4}"
)
_LABELLED_DATE = re.compile(rf"(?:last updated|effective|date)\s*:?\s*({_DATE})", re.IGNORECASE)
_ANY_DATE = re.compile(rf"\b({_DATE})\b")


def normalize_text(text: str) -> str:
    """Plain characters, single spaces, at most one blank line between paragraphs."""
    text = unicodedata.normalize("NFKC", text).translate(_REPLACEMENTS)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def join_lines(lines: list[str]) -> str:
    """Join wrapped lines of one paragraph into a single line.

    A hyphen at a line end is kept and the next line is attached without a space:
    without a dictionary, "in-" + "service" cannot be told apart from a soft hyphen.
    """
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha():
            out += line
        else:
            out += " " + line
    return out


def find_doc_date(text: str) -> str | None:
    """The document date as written: a labelled date first, otherwise the first date seen."""
    match = _LABELLED_DATE.search(text) or _ANY_DATE.search(text)
    return match.group(1) if match else None
