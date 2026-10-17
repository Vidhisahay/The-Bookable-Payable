"""Locale-aware primitive parsing. Ambiguous separators remain unresolved."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


@dataclass(frozen=True)
class ParsedNumber:
    value: Decimal | None
    ambiguous: bool = False


def parse_localized_decimal(raw: str) -> ParsedNumber:
    value = raw.strip().replace("−", "-").replace("'", "").replace("’", "")
    value = re.sub(r"[\s\u00a0\u202f]", "", value)
    value = re.sub(r"[^0-9,.-]", "", value)
    if not value or value in {"-", ".", ","}:
        return ParsedNumber(None)

    negative = value.startswith("-") or (raw.strip().startswith("(") and raw.strip().endswith(")"))
    value = value.lstrip("+-")
    if "," in value and "." in value:
        decimal_mark = "," if value.rfind(",") > value.rfind(".") else "."
        grouping_mark = "." if decimal_mark == "," else ","
        value = value.replace(grouping_mark, "").replace(decimal_mark, ".")
    elif "," in value or "." in value:
        mark = "," if "," in value else "."
        parts = value.split(mark)
        if len(parts) > 2:
            if len(parts[-1]) in (1, 2, 4, 5, 6):
                value = "".join(parts[:-1]) + "." + parts[-1]
            elif len(parts[-1]) == 3:
                value = "".join(parts)
            else:
                return ParsedNumber(None, ambiguous=True)
        elif len(parts) == 2:
            left, right = parts
            if len(right) == 3:
                return ParsedNumber(None, ambiguous=True)
            if not right:
                return ParsedNumber(None, ambiguous=True)
            value = left + "." + right
    try:
        number = Decimal(value)
    except InvalidOperation:
        return ParsedNumber(None, ambiguous=True)
    return ParsedNumber(-number if negative else number)
