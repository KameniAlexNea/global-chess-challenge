import re
from src.config import (
    rationale_tag,
    close_rationale_tag,
    move_tag,
    close_move_tag,
    name_used
)


def extract_xml_answer(text):
    """
    XML extraction - NO participation trophies.
    If the model can't use tags, it gets zero. Forces proper formatting.
    Returns: (rationale, move, has_correct_format)
    """
    # Strategy 1: Strict XML format
    strict_regex = rf"{rationale_tag}([^<]*(?:<(?!/?{name_used}>)[^<]*)*){close_rationale_tag}\s*{move_tag}([^<]+){close_move_tag}"
    match = re.search(strict_regex, text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip(), True

    # Strategy 2: Loose XML - allow newlines/spaces inside tags AND chatter between tags
    loose_regex = rf"{rationale_tag}\s*(.+?)\s*{close_rationale_tag}.*?{move_tag}\s*(.+?)\s*{close_move_tag}"
    match = re.search(loose_regex, text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip(), True

    # No fallbacks - if it can't use XML tags, it fails
    return None, None, False