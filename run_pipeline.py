import argparse
import logging
import os
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config_loader import load_config
from src.lqa_pipeline import TaskConfig, run_lqa_first_pass, run_lqa_review_pass
from src.normalization import normalize_errors_list
from src.parser import process_tracker
from src.reporting import generate_lqa_scorecard
from src.tb import LANG_NAME_TO_CODE, add_tb_matches_to_consolidated
from src.utils import load_guidelines_text, load_template, read_excel_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pipeline")


def pick_languages(tracker_df: pd.DataFrame, requested: List[str], mapping_order: List[str] | None = None) -> List[str]:
    available = set(tracker_df["target"].dropna().unique())

    # If ALL requested, respect mapping order when provided, then append any extra tracker languages.
    if not requested or "ALL" in requested:
        if mapping_order:
            ordered = [lang for lang in mapping_order if lang in available]
            extras = sorted([lang for lang in available if lang not in mapping_order])
            return ordered + extras
        return sorted(list(available))

    selected = [l for l in requested if l in available]
    missing = [l for l in requested if l not in available]
    if missing:
        logger.warning("Requested languages not found in tracker and will be skipped: %s", missing)
    if not selected:
        raise ValueError("No valid languages to process after filtering.")
    return selected


def _get_api_key(llm_cfg: Dict, *, agent: str | None = None) -> str:
    """
    Resolve API key with optional agent-specific override.
    Order: agent key -> common key -> env var override.
    """
    key_field = f"{agent}_api_key" if agent else "api_key"
    env_field = f"{agent}_api_key_env" if agent else "api_key_env"
    api_key = llm_cfg.get(key_field) or llm_cfg.get("api_key") or os.getenv(llm_cfg.get(env_field, "") or llm_cfg.get("api_key_env", ""))
    if not api_key:
        raise ValueError(f"API key not provided. Set llm.{key_field} or llm.api_key / api_key_env in config/environment.")
    return api_key


def load_checkpoint(checkpoints_dir: Path, language: str) -> pd.DataFrame | None:
    ckpt_path = checkpoints_dir / language / "df_checkpoint.parquet"
    if ckpt_path.exists():
        return pd.read_parquet(ckpt_path)
    return None


def save_checkpoint(df: pd.DataFrame, checkpoints_dir: Path, language: str) -> None:
    ckpt_lang_dir = checkpoints_dir / language
    ckpt_lang_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ckpt_lang_dir / "df_checkpoint.parquet", index=False)


def _ensure_list(obj):
    try:
        import numpy as np
    except Exception:
        np = None
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj if isinstance(obj, list) else []


def _normalize_collection_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure list-like columns remain Python lists after round-tripping through parquet.
    This prevents np.ndarray values from breaking downstream rendering (e.g., TB Matches in reports).
    """
    list_like_cols = ["TB Matches", "Agent1_Errors", "Final_Errors"]
    df_out = df.copy()
    for col in list_like_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].apply(_ensure_list)
    return df_out


def _skip_reason(source: str, target: str | None) -> str | None:
    """
    Return a string reason to skip LQA for this segment, or None to keep.
    Rules cover empty, URL-only, placeholder-only, punctuation-only, numeric-only, and extremely short strings.
    """
    src = "" if pd.isna(source) else str(source)
    tgt = "" if pd.isna(target) else str(target)
    src_strip = src.strip()
    tgt_strip = tgt.strip()

    if not src_strip or not tgt_strip:
        return "Empty source/target"
    if re.search(r"(https?://|www\\.|\\.com\\b|\\.net\\b|\\.io\\b|\\.org\\b)", src_strip, flags=re.IGNORECASE):
        return "URL-only"
    if re.fullmatch(r"(?:\\s*(?:\\{[^}]+\\}|%s|<\\d+>))+\\s*", src_strip):
        return "Placeholder-only"
    if re.fullmatch(r"[\\W_]+", src_strip):
        return "Punctuation-only"
    if re.fullmatch(r"[\\d\\s.,:/+\\-]+", src_strip):
        return "Numeric-only"
    if len(src_strip) <= 2:
        return "Too-short"
    return None


def _canon_text(text: str) -> str:
    """Lowercase and collapse whitespace for dedupe key construction."""
    if pd.isna(text):
        return ""
    return " ".join(str(text).strip().split()).lower()


def _annotate_skip_and_repeats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Skip_Reason, Is_Repeat, Repeat_Key, Repeat_Of, and _Orig_Index.
    Only non-skipped rows are considered for dedupe; repeats are exact Source+Target matches (normalized).
    """
    df_out = df.copy()
    df_out["_Orig_Index"] = df_out.index
    seen_keys: Dict[str, str] = {}

    skip_reasons: List[str | None] = []
    is_repeat: List[bool] = []
    repeat_keys: List[str] = []
    repeat_of: List[str | None] = []

    for _, row in df_out.iterrows():
        src = row.get("Source")
        tgt = row.get("Target")
        reason = _skip_reason(src, tgt)
        key = f"{_canon_text(src)}||{_canon_text(tgt)}"
        repeat_keys.append(key)
        skip_reasons.append(reason)

        if reason is None:
            if key in seen_keys:
                is_repeat.append(True)
                repeat_of.append(seen_keys[key])
            else:
                is_repeat.append(False)
                repeat_of.append(None)
                # Use Segment_ID as identifier to reference the first occurrence
                seen_keys[key] = str(row.get("Segment_ID", ""))
        else:
            is_repeat.append(False)
            repeat_of.append(None)

    df_out["Skip_Reason"] = skip_reasons
    df_out["Is_Repeat"] = is_repeat
    df_out["Repeat_Key"] = repeat_keys
    df_out["Repeat_Of"] = repeat_of
    return df_out


def _ensure_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure result columns exist with sensible defaults."""
    df_out = df.copy()
    defaults = {
        "Batch_ID": -1,
        "Batch_Input": None,
        "Agent1_Output": None,
        "Agent1_Edited_Target": df_out.get("Target", ""),
        "Agent1_Errors": [[] for _ in range(len(df_out))],
        "Agent2_Status": "",
        "Agent2_Output": None,
        "Final_Target": df_out.get("Target", ""),
        "Final_Errors": [[] for _ in range(len(df_out))],
    }
    for col, default in defaults.items():
        if col not in df_out.columns:
            df_out[col] = default
    # Normalize list-like defaults
    df_out["Agent1_Errors"] = df_out["Agent1_Errors"].apply(_ensure_list)
    df_out["Final_Errors"] = df_out["Final_Errors"].apply(_ensure_list)
    return df_out


def _propagate_repeats(processed_df: pd.DataFrame, repeats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Copy LQA outputs from first occurrences to their repeats.
    """
    if repeats_df.empty:
        return repeats_df

    ref_map = {row["Repeat_Key"]: row for _, row in processed_df.iterrows()} if not processed_df.empty else {}
    rep_out = repeats_df.copy()
    rep_out = _ensure_result_columns(rep_out)

    for idx, row in rep_out.iterrows():
        key = row.get("Repeat_Key")
        ref = ref_map.get(key)
        if ref is None:
            rep_out.at[idx, "Agent2_Status"] = "Repeat (missing base)"
            continue
        rep_out.at[idx, "Agent2_Status"] = "Repeat"
        rep_out.at[idx, "Final_Target"] = ref.get("Final_Target", row.get("Target", ""))
        rep_out.at[idx, "Final_Errors"] = _ensure_list(ref.get("Final_Errors", []))
        rep_out.at[idx, "Agent1_Edited_Target"] = ref.get("Agent1_Edited_Target", row.get("Target", ""))
        rep_out.at[idx, "Agent1_Errors"] = _ensure_list(ref.get("Agent1_Errors", []))
        rep_out.at[idx, "Agent1_Output"] = ref.get("Agent1_Output")
        rep_out.at[idx, "Agent2_Output"] = ref.get("Agent2_Output")
        rep_out.at[idx, "Batch_ID"] = ref.get("Batch_ID", -1)
        rep_out.at[idx, "Batch_Input"] = ref.get("Batch_Input")
    return rep_out


def _mark_skipped(df: pd.DataFrame) -> pd.DataFrame:
    """Set LQA columns for skipped rows."""
    if df.empty:
        return df
    df_out = _ensure_result_columns(df)
    df_out["Agent2_Status"] = "Skipped"
    return df_out


def _fill_repeats_from_bases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all repeat rows inherit LQA outputs and status from their base rows
    even if a previous save skipped propagation.
    """
    if "Is_Repeat" not in df.columns or "Repeat_Key" not in df.columns:
        return df
    df_out = _ensure_result_columns(df)
    bases = df_out[df_out["Is_Repeat"] == False]
    base_map = {row["Repeat_Key"]: row for _, row in bases.iterrows()}

    for idx, row in df_out[df_out["Is_Repeat"] == True].iterrows():
        base = base_map.get(row.get("Repeat_Key"))
        if base is None:
            df_out.at[idx, "Agent2_Status"] = "Repeat (missing base)"
            continue
        df_out.at[idx, "Agent2_Status"] = "Repeat"
        df_out.at[idx, "Final_Target"] = base.get("Final_Target", row.get("Target", ""))
        df_out.at[idx, "Final_Errors"] = _ensure_list(base.get("Final_Errors", []))
        df_out.at[idx, "Agent1_Edited_Target"] = base.get("Agent1_Edited_Target", row.get("Target", ""))
        df_out.at[idx, "Agent1_Errors"] = _ensure_list(base.get("Agent1_Errors", []))
        df_out.at[idx, "Agent1_Output"] = base.get("Agent1_Output")
        df_out.at[idx, "Agent2_Output"] = base.get("Agent2_Output")
        df_out.at[idx, "Batch_ID"] = base.get("Batch_ID", -1)
        df_out.at[idx, "Batch_Input"] = base.get("Batch_Input")
    return df_out


def main():
    parser = argparse.ArgumentParser(description="Run full LQA pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--language",
        action="append",
        dest="languages",
        help="Target language(s) to process (can be passed multiple times). Defaults to config languages or ALL.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    llm_cfg = cfg.get("llm", {})
    tb_cfg = cfg.get("tb", {})
    parser_cfg = cfg.get("parser", {})
    languages_cfg = cfg.get("languages", {})

    LANG_NAME_TO_CODE.update({k: v.get("code") for k, v in languages_cfg.get("mapping", {}).items() if v.get("code")})

    tracker_path = Path(paths.get("tracker", "")).expanduser()
    glossary_path = Path(paths.get("glossary", "")).expanduser()
    prompts_dir = Path(paths.get("prompts_dir", "prompts")).expanduser()
    instructions_dir = Path(paths.get("instructions_dir", "Langs_Instructions")).expanduser()
    xliff_dir = Path(paths.get("xliff_download_dir", "XLIFF_Downloads")).expanduser()
    output_dir = Path(paths.get("output_dir", "Output")).expanduser()
    checkpoints_dir = Path(paths.get("checkpoints_dir", "checkpoints")).expanduser()

    tracker_df = read_excel_file(tracker_path)
    requested_langs = args.languages or languages_cfg.get("process", ["ALL"])
    mapping_order = list(languages_cfg.get("mapping", {}).keys())
    languages = pick_languages(tracker_df, requested_langs, mapping_order=mapping_order)

    tracker_filtered = tracker_df[tracker_df["target"].isin(languages)]

    logger.info("Running XLIFF parser for languages: %s", languages)
    df_segments, df_audit = process_tracker(
        tracker_filtered,
        str(xliff_dir),
        context_size=int(parser_cfg.get("context_size", 2)),
    )

    df_final = add_tb_matches_to_consolidated(
        df_segments,
        glossary_path,
        threshold=int(tb_cfg.get("threshold", 85)),
        min_len_fuzzy=int(tb_cfg.get("min_len_fuzzy", 5)),
    )

    system_1_path = prompts_dir / cfg.get("prompts", {}).get("system_1", "system_1.txt")
    user_1_path = prompts_dir / cfg.get("prompts", {}).get("user_agent_1", "user_agent_1.txt")
    system_2_path = prompts_dir / cfg.get("prompts", {}).get("system_2", "system_2.txt")
    user_2_path = prompts_dir / cfg.get("prompts", {}).get("user_agent_2", "user_agent_2.txt")

    src_lang_label = llm_cfg.get("source_lang_label", "English (UK)")

    for lang in languages:
        # Checkpoint: load existing if present
        ckpt_df = load_checkpoint(checkpoints_dir, lang)
        if ckpt_df is not None:
            logger.info("Loaded checkpoint for %s", lang)
            lang_df = _normalize_collection_columns(ckpt_df)
        else:
            lang_df = df_final[df_final["Language"] == lang].copy()
            if lang_df.empty:
                logger.warning("No rows for language %s after TB matching; skipping.", lang)
                continue

        # Tag skip/repeats early to avoid unnecessary LQA
        lang_df = _annotate_skip_and_repeats(lang_df)

        lang_cfg_entry = languages_cfg.get("mapping", {}).get(lang, {})
        guideline_file = lang_cfg_entry.get("guidelines")
        try:
            guidelines_text = load_guidelines_text(instructions_dir, lang, guideline_file)
        except FileNotFoundError as exc:
            logger.warning("Guidelines missing for %s: %s. Continuing with empty guidelines.", lang, exc)
            guidelines_text = ""

        system_tpl_agent1 = load_template(system_1_path)
        system_tpl_agent2 = load_template(system_2_path)
        system_agent1 = system_tpl_agent1.safe_substitute(
            source_lang=src_lang_label,
            target_lang=lang,
            lang_specific_guidelines=guidelines_text,
        )
        system_agent2 = system_tpl_agent2.safe_substitute(
            source_lang=src_lang_label,
            target_lang=lang,
            lang_specific_guidelines=guidelines_text,
        )

        agent1_model = llm_cfg.get("agent1_model") or llm_cfg.get("model", "gemini-3-pro-preview")
        agent2_model = llm_cfg.get("agent2_model") or llm_cfg.get("model", "gemini-3-pro-preview")
        agent1_api_key = _get_api_key(llm_cfg, agent="agent1")
        agent2_api_key = _get_api_key(llm_cfg, agent="agent2")

        agent1_cfg = TaskConfig(
            source_lang=src_lang_label,
            target_lang=lang,
            model=agent1_model,
            api_key=agent1_api_key,
            temp=float(llm_cfg.get("temp", 1.0)),
            prompt=load_template(user_1_path),
            system=system_agent1,
        )

        agent2_cfg = TaskConfig(
            source_lang=src_lang_label,
            target_lang=lang,
            model=agent2_model,
            api_key=agent2_api_key,
            temp=float(llm_cfg.get("temp", 1.0)),
            prompt=load_template(user_2_path),
            system=system_agent2,
        )

        # Split rows
        to_process = lang_df[(lang_df["Skip_Reason"].isna()) & (~lang_df["Is_Repeat"])].copy()
        skipped_rows = lang_df[lang_df["Skip_Reason"].notna()].copy()
        repeat_rows = lang_df[lang_df["Is_Repeat"]].copy()

        agent2_complete = "Agent2_Status" in lang_df.columns and (lang_df["Agent2_Status"].fillna("") != "").all()

        if agent2_complete:
            logger.info("Agent 2 already completed for %s. Skipping LLM calls.", lang)
            df_result = lang_df.copy()
        else:
            if to_process.empty:
                processed_result = _ensure_result_columns(to_process)
            else:
                if "Batch_ID" in to_process.columns:
                    df_step1 = to_process.copy()
                    logger.info("Resuming from Agent1-complete checkpoint for %s", lang)
                else:
                    logger.info("Running Agent 1 for %s (%d segments)", lang, len(to_process))
                    df_step1 = run_lqa_first_pass(
                        to_process,
                        agent1_cfg,
                        batch_segments=int(llm_cfg.get("batch_segments", 1)),
                        max_concurrency=int(llm_cfg.get("max_concurrency", 25)),
                        wait_seconds=int(llm_cfg.get("wait_seconds", 5)),
                        include_context=True,
                    )
                    save_checkpoint(pd.concat([df_step1, repeat_rows, skipped_rows]), checkpoints_dir, lang)

                if "Agent2_Status" in df_step1.columns and (df_step1["Agent2_Status"].fillna("") != "").all():
                    processed_result = df_step1.copy()
                else:
                    logger.info("Running Agent 2 for %s", lang)
                    processed_result = run_lqa_review_pass(
                        df_step1,
                        agent2_cfg,
                        max_concurrency=int(llm_cfg.get("max_concurrency", 25)),
                        wait_seconds=int(llm_cfg.get("wait_seconds", 5)),
                    )
                    save_checkpoint(pd.concat([processed_result, repeat_rows, skipped_rows]), checkpoints_dir, lang)

            # Propagate LQA outputs to repeats and mark skipped rows
            processed_result = _ensure_result_columns(processed_result)
            repeat_filled = _propagate_repeats(processed_result, repeat_rows)
            skipped_filled = _mark_skipped(skipped_rows)

            df_result = pd.concat([processed_result, repeat_filled, skipped_filled], ignore_index=True)
            if "_Orig_Index" in df_result.columns:
                df_result = df_result.sort_values("_Orig_Index")
                df_result = df_result.drop(columns=["_Orig_Index"], errors="ignore")

        if "_Orig_Index" in df_result.columns:
            df_result = df_result.sort_values("_Orig_Index")
            df_result = df_result.drop(columns=["_Orig_Index"], errors="ignore")

        # Normalize category/subcategory to the allowed set
        df_result["Final_Errors"] = df_result["Final_Errors"].apply(_ensure_list).apply(normalize_errors_list)
        if "Agent1_Errors" in df_result.columns:
            df_result["Agent1_Errors"] = df_result["Agent1_Errors"].apply(_ensure_list).apply(normalize_errors_list)

        # Final safety pass to ensure repeats inherit base LQA outputs
        df_result = _fill_repeats_from_bases(df_result)

        output_dir.mkdir(parents=True, exist_ok=True)
        generate_lqa_scorecard(df_result, str(output_dir), lang)


if __name__ == "__main__":
    main()
