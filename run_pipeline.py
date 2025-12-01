import argparse
import logging
import os
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


def pick_languages(tracker_df: pd.DataFrame, requested: List[str]) -> List[str]:
    if not requested or "ALL" in requested:
        return sorted([lang for lang in tracker_df["target"].dropna().unique()])
    available = set(tracker_df["target"].dropna().unique())
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

    tracker_df = read_excel_file(tracker_path)
    requested_langs = args.languages or languages_cfg.get("process", ["ALL"])
    languages = pick_languages(tracker_df, requested_langs)

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
        lang_df = df_final[df_final["Language"] == lang].copy()
        if lang_df.empty:
            logger.warning("No rows for language %s after TB matching; skipping.", lang)
            continue

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

        logger.info("Running Agent 1 for %s (%d segments)", lang, len(lang_df))
        df_step1 = run_lqa_first_pass(
            lang_df,
            agent1_cfg,
            batch_segments=int(llm_cfg.get("batch_segments", 1)),
            max_concurrency=int(llm_cfg.get("max_concurrency", 25)),
            wait_seconds=int(llm_cfg.get("wait_seconds", 5)),
            include_context=True,
        )

        logger.info("Running Agent 2 for %s", lang)
        df_result = run_lqa_review_pass(
            df_step1,
            agent2_cfg,
            max_concurrency=int(llm_cfg.get("max_concurrency", 25)),
            wait_seconds=int(llm_cfg.get("wait_seconds", 5)),
        )

        # Normalize category/subcategory to the allowed set
        df_result["Final_Errors"] = df_result["Final_Errors"].apply(normalize_errors_list)
        if "Agent1_Errors" in df_result.columns:
            df_result["Agent1_Errors"] = df_result["Agent1_Errors"].apply(normalize_errors_list)

        output_dir.mkdir(parents=True, exist_ok=True)
        generate_lqa_scorecard(df_result, str(output_dir), lang)


if __name__ == "__main__":
    main()
