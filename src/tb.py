import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from rapidfuzz import fuzz, process

# ────────────────────────────────────────────────────────────────────────────
#  Configuration & Mapping
# ────────────────────────────────────────────────────────────────────────────
LANG_NAME_TO_CODE: Dict[str, str] = {
    "Arabic": "ar-SA",
    "Catalan": "ca-ES",
    "Chinese": "zh-CN",
    "Chinese Traditional": "zh-TW",
    "Croatian": "hr-HR",
    "Czech": "cs-CZ",
    "Dutch": "nl-NL",
    "French": "fr-FR",
    "German": "de-DE",
    "Greek": "el-GR",
    "Hindi": "hi-IN",
    "Italian": "it-IT",
    "Japanese": "ja-JP",
    "Korean": "ko-KR",
    "Norwegian Bokmal": "nb-NO",
    "Portuguese": "pt-PT",
    "Portuguese Brazil": "pt-BR",
    "Romanian": "ro-RO",
    "Slovak": "sk-SK",
    "Spanish": "es-ES",
    "Spanish Mexico": "es-MX",
    "Swedish": "sv-SE",
    "Thai": "th-TH",
    "Turkish": "tr-TR",
}

_WORD_RE = re.compile(r"\w+")


# ────────────────────────────────────────────────────────────────────────────
#  Build look-up arrays
# ────────────────────────────────────────────────────────────────────────────
def prep_tb_lookup(tb_df: pd.DataFrame, src_col: str, trg_col: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Return three parallel lists:

    * src_terms_lower   – lower-cased for matching
    * src_terms_orig    – untouched original spelling / casing
    * trg_terms         – target-language equivalents
    """
    src_terms_orig = tb_df[src_col].astype(str).tolist()
    src_terms_lower = [t.lower() for t in src_terms_orig]
    trg_terms = tb_df[trg_col].astype(str).tolist()
    return src_terms_lower, src_terms_orig, trg_terms


# ────────────────────────────────────────────────────────────────────────────
#  Word regex & helpers
# ────────────────────────────────────────────────────────────────────────────
def _contains_longer_match(candidate: str, accepted: List[str]) -> bool:
    """
    Return True if *candidate* is a substring (case-insensitive) of any term
    already accepted.  Used to remove 'sampler' after we kept
    'Agilent InfinityLab Sample ID Reader'.
    """
    cand_low = candidate.lower()
    return any(cand_low in acc.lower() for acc in accepted)


# ────────────────────────────────────────────────────────────────────────────
#  Main matcher
# ────────────────────────────────────────────────────────────────────────────
def get_tb_matches(
    source_segment: str,
    src_terms_lower: List[str],
    src_terms_orig: List[str],
    trg_terms: List[str],
    *,
    threshold: int = 95,
    min_len_fuzzy: int = 4,
) -> List[Dict[str, str | int]]:
    """
    Two-stage exact+fuzzy matcher that:

    * returns the original-casing TB term (`'src'` field)
    * drops shorter terms fully covered by longer **exact** matches
    """
    if not isinstance(source_segment, str) or not source_segment:
        return []

    src_low_seg = source_segment.lower()
    words_in_seg = set(_WORD_RE.findall(src_low_seg))

    exact_hits: List[Dict[str, str | int]] = []
    exact_src_terms: List[str] = []
    fuzzy_src: List[str] = []
    fuzzy_orig: List[str] = []
    fuzzy_trg: List[str] = []

    for s_low, s_orig, t_term in zip(src_terms_lower, src_terms_orig, trg_terms):
        if " " in s_orig:
            fuzzy_src.append(s_low)
            fuzzy_orig.append(s_orig)
            fuzzy_trg.append(t_term)
            continue

        if s_low in words_in_seg:
            if not _contains_longer_match(s_orig, exact_src_terms):
                exact_hits.append({"src": s_orig, "trg": t_term, "score": 100})
                exact_src_terms.append(s_orig)
        elif len(s_orig) >= min_len_fuzzy:
            fuzzy_src.append(s_low)
            fuzzy_orig.append(s_orig)
            fuzzy_trg.append(t_term)

    fuzzy_hits: List[Dict[str, str | int]] = []
    if fuzzy_src:
        matches = process.extract(
            src_low_seg,
            fuzzy_src,
            scorer=fuzz.partial_ratio,
            score_cutoff=threshold,
        )
        for _match_low, score, idx in matches:
            s_orig = fuzzy_orig[idx]
            if _contains_longer_match(s_orig, exact_src_terms):
                continue
            fuzzy_hits.append({"src": s_orig, "trg": fuzzy_trg[idx], "score": score})

    return exact_hits + fuzzy_hits


# ────────────────────────────────────────────────────────────────────────────
#  Add column to a DataFrame
# ────────────────────────────────────────────────────────────────────────────
def add_tb_matches(
    df: pd.DataFrame,
    tb_df: pd.DataFrame,
    src_col_tb: str,
    trg_col_tb: str,
    *,
    source_seg_col: str = "Source",
    threshold: int = 85,
    min_len_fuzzy: int = 5,
) -> pd.DataFrame:
    """
    Copy *df* and append the column ``'TB Matches'``.  Uses the revised matcher
    that respects original casing and removes nested shorter terms.
    """
    src_low, src_orig, trg_terms = prep_tb_lookup(tb_df, src_col_tb, trg_col_tb)

    out = df.copy()
    out["TB Matches"] = out[source_seg_col].apply(
        lambda seg: get_tb_matches(
            seg,
            src_low,
            src_orig,
            trg_terms,
            threshold=threshold,
            min_len_fuzzy=min_len_fuzzy,
        )
    )
    return out


def format_tb_matches(matches) -> str:
    """
    Turn the list-of-dicts returned by *get_tb_matches* into a readable string.

    • When *matches* is NaN / None / [] ➜ return "No TB matches found".
    """
    if not isinstance(matches, (list, tuple)) or len(matches) == 0:
        return "No TB matches found"

    lines = ["TB matches:"]
    for m in matches:
        lines.append(f"  • Source term: {m['src']}")
        lines.append(f"    Target term: {m['trg']}\n")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
#  Read TB from Excel files
# ────────────────────────────────────────────────────────────────────────────
def read_excel_tb_columns(
    file_path: str | Path,
    columns: List[str],
    *,
    sheet_name: str | int | None = 0,
    header: int | None = 0,
    **read_excel_kwargs,
) -> pd.DataFrame:
    """
    Load *just the columns you care about* from an Excel workbook.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    df_full = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=header,
        **read_excel_kwargs,
    )

    if isinstance(df_full, dict):
        df_full = pd.concat(df_full.values(), ignore_index=True)

    missing = [col for col in columns if col not in df_full.columns]
    if missing:
        raise ValueError(f"The following columns are missing in '{file_path.name}': {missing}")
    tb = df_full[columns].copy()
    logging.info(f"Source column: {columns[0]} Target column {columns[1]}")
    tb = tb.dropna(subset=[columns[0], columns[1]])
    tb = tb[(tb[columns[0]] != "") & (tb[columns[1]] != "")]
    tb = tb.drop_duplicates(subset=[columns[0], columns[1]])
    return tb


def _detect_english_header(df: pd.DataFrame) -> str:
    """Detects 'en-GB' or 'en-US' in a loaded DataFrame columns."""
    cols = list(df.columns)
    if "en-GB" in cols:
        return "en-GB"
    if "en-US" in cols:
        return "en-US"
    for c in cols:
        if isinstance(c, str) and c.lower().startswith("en-"):
            logging.warning(f"Using fallback English source column: {c}")
            return c
    raise ValueError("Could not find English source column (en-GB/en-US).")


def load_glossary_once(file_path: Path) -> pd.DataFrame:
    """Loads the entire glossary into memory once."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Glossary not found: {file_path}")

    try:
        df = pd.read_excel(file_path, header=0)
        return df
    except Exception as e:
        raise ValueError(f"Failed to load glossary Excel: {e}")


# ────────────────────────────────────────────────────────────────────────────
#  Consolidated DataFrame Wrapper
# ────────────────────────────────────────────────────────────────────────────
def add_tb_matches_to_consolidated(
    df_segments: pd.DataFrame,
    glossary_path: str | Path,
    *,
    source_seg_col: str = "Source",
    lang_col: str = "Language",
    threshold: int = 85,
    min_len_fuzzy: int = 5,
) -> pd.DataFrame:
    """
    Takes the consolidated segments DataFrame, groups by Language,
    slices the relevant columns from the Glossary, and applies matching.
    """
    logging.info(f"Loading Glossary: {glossary_path} ...")
    glossary_full = load_glossary_once(glossary_path)
    en_col = _detect_english_header(glossary_full)

    processed_dfs: List[pd.DataFrame] = []

    for lang_name, lang_group in df_segments.groupby(lang_col):
        glossary_code = LANG_NAME_TO_CODE.get(lang_name)

        if not glossary_code or glossary_code not in glossary_full.columns:
            logging.warning(
                f"[TB] No glossary column found for '{lang_name}' (Code: {glossary_code}). Skipping TB for this lang."
            )
            lang_group = lang_group.copy()
            lang_group["TB Matches"] = [[] for _ in range(len(lang_group))]
            processed_dfs.append(lang_group)
            continue

        tb_subset = glossary_full[[en_col, glossary_code]].dropna()
        tb_subset = tb_subset[(tb_subset[en_col] != "") & (tb_subset[glossary_code] != "")]
        tb_subset = tb_subset.drop_duplicates()

        logging.info(f"[TB] Processing {lang_name} ({len(lang_group)} segments) vs Glossary ({len(tb_subset)} terms)")

        src_low, src_orig, trg_terms = prep_tb_lookup(tb_subset, en_col, glossary_code)
        lang_group = lang_group.copy()

        lang_group["TB Matches"] = lang_group[source_seg_col].apply(
            lambda seg: get_tb_matches(
                seg,
                src_low,
                src_orig,
                trg_terms,
                threshold=threshold,
                min_len_fuzzy=min_len_fuzzy,
            )
        )

        processed_dfs.append(lang_group)

    if not processed_dfs:
        return df_segments.copy()

    final_df = pd.concat(processed_dfs)
    return final_df
