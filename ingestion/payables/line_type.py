"""Conservative classification of a printed invoice row into schema item types.

An invoice table heading such as "Product Reference" describes a column, not
necessarily the accounting nature of every row. Competing signals are left
unresolved for review.
"""

from __future__ import annotations

import re


_SIGNALS = {
    "freight": (
        r"\b(?:freight|postage|carriage|fracht|frete|porto)\b",
        r"\bshipping(?:\s+(?:charge|fee|cost|service))?\s*$",
        r"\b(?:delivery|transport)\s+(?:charge|fee|service)\b",
        r"\b(?:international|local|inland)\s+transport\b",
        r"^transport\b",
    ),
    "service": (
        r"\b(?:consulting|consultancy|maintenance|cleaning|handling|storage)\s+(?:service|fee|charge|work)?\b",
        r"\b(?:management|support|installation|repair)\s+(?:fee|service|work)\b",
        r"\b(?:service|services|servi[cç]o|servicio|technician|tehnik|projektmanagement|arbeit)\b",
        r"\b\d+(?:[.,]\d+)?\s*(?:hours?|hrs?|stunde[n]?|horas?)\b",
    ),
    "goods": (
        r"\b(?:goods|equipment|device|hardware|material|kit|controller|mixer|speaker|subwoofer|cable|kaabel|k[oõ]lar|statiiv|pikendusjuhe|led\s+bar)\b",
        r"\b(?:product|component)\s+(?:unit|item|model)\b",
    ),
    "tax": (r"\b(?:vat|gst|tax|withholding)\s+(?:charge|levy|amount)\b",),
}


def classify_line_type(description: str) -> str | None:
    """Return a supported schema type only when one category is indicated.

    A mixed description such as "transport equipment" has competing signals.
    The caller must preserve it as unresolved rather than choose by priority.
    """
    text = description.casefold()
    signals = {kind for kind, patterns in _SIGNALS.items()
               if any(re.search(pattern, text, re.I) for pattern in patterns)}
    return next(iter(signals)) if len(signals) == 1 else None
