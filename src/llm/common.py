import json
import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)


def extract_json(response_text: str) -> Dict[str, Any]:
    """
    More robust JSON extraction handling common LLM quirks.
    """
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON found in response")

    json_str = json_match.group(0)
    json_str = re.sub(r"(\\w)\\'(\\w)", r"\1'\2", json_str)
    json_str = re.sub(r"\\([^\"\\nrtbfux/])", r"\1", json_str)
    json_str = re.sub(r'\\""', r'"', json_str)
    json_str = fix_newlines_in_strings(json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error("JSON decode error: %s", e)
        logger.error("Attempted to parse: %s...", json_str[:200])
        try:
            return fallback_extraction(json_str)
        except Exception:
            raise ValueError(f"Unable to parse JSON: {e}")


def fix_newlines_in_strings(json_str: str) -> str:
    """
    Fix unescaped newlines within JSON string values.
    """
    result = []
    in_string = False
    escaped = False
    i = 0

    while i < len(json_str):
        char = json_str[i]

        if char == "\\" and not escaped:
            escaped = True
            result.append(char)
        elif char == '"' and not escaped:
            in_string = not in_string
            result.append(char)
        elif char == "\n" and in_string and not escaped:
            result.append("\\n")
        else:
            if escaped:
                escaped = False
            result.append(char)

        i += 1

    return "".join(result)


def fallback_extraction(json_str: str) -> Dict[str, Any]:
    """Manual extraction when JSON parsing fails"""
    result = {"errors": []}

    errors_match = re.search(r'"errors"\s*:\s*\[\s*\]', json_str)
    if errors_match:
        result["errors"] = []
    else:
        error_pattern = r'\{[^}]*"category"\s*:\s*"([^"]*)"[^}]*"severity"\s*:\s*"([^"]*)"[^}]*\}'
        for match in re.finditer(error_pattern, json_str):
            result["errors"].append({"category": match.group(1), "severity": match.group(2)})

    if '"corrections"' in json_str:
        corrections = {}

        target_match = re.search(r'"edited_target"\s*:\s*"((?:[^"\\]|\\.)*)"', json_str, re.DOTALL)
        if target_match:
            value = target_match.group(1)
            value = value.replace('\\"', '"').replace("\\n", "\n")
            corrections["edited_target"] = value

        rationale_match = re.search(r'"rationale"\s*:\s*"((?:[^"\\]|\\.)*)"', json_str, re.DOTALL)
        if rationale_match:
            value = rationale_match.group(1)
            value = value.replace('\\"', '"').replace("\\n", "\n")
            corrections["rationale"] = value

        if corrections:
            result["corrections"] = corrections

    return result
