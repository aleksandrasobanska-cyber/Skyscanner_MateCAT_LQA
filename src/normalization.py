from typing import Dict, List

CANONICAL_CATEGORIES = {
    "accuracy": "Accuracy",
    "terminology": "Terminology",
    "grammar": "Grammar",
    "style": "Style",
    "locale conventions": "Locale Conventions",
    "locale": "Locale Conventions",
    "local": "Locale Conventions",
    "formatting": "Formatting",
}

CANONICAL_SUBCATEGORIES = {
    "mistrans": "Mistranslation",
    "omission": "Omission",
    "addition": "Addition",
    "term inconsistency": "Term inconsistency",
    "syntax": "Syntax error",
    "morphology": "Morphology error",
    "tone": "Tone/register",
    "register": "Tone/register",
    "redundancy": "Redundancy",
    "ambiguity": "Ambiguity",
    "punctuation": "Punctuation",
    "date": "Date format",
    "time": "Time format",
    "number": "Number format",
    "placeholder": "Placeholder mismatch",
    "tag": "Tag mismatch",
}


def _normalize_single_error(err: Dict) -> Dict:
    if not isinstance(err, dict):
        return err
    out = err.copy()
    cat = str(out.get("category", "") or "").lower()
    sub = str(out.get("subcategory", "") or "").lower()

    for key, canon in CANONICAL_CATEGORIES.items():
        if key in cat:
            out["category"] = canon
            break

    for key, canon in CANONICAL_SUBCATEGORIES.items():
        if key in sub:
            out["subcategory"] = canon
            break

    return out


def normalize_errors_list(err_list) -> List[Dict]:
    if not isinstance(err_list, list):
        return err_list
    return [_normalize_single_error(e) for e in err_list]
