"""Helpers for parsing loosely formatted LLM numeric fields."""
import re
from typing import Any


def parse_llm_number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.match(r"^([\d.]+)", value.strip())
        if match:
            return float(match.group(1))
    return default
