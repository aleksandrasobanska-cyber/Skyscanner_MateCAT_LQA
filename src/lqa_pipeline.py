import asyncio
import json
import logging
from dataclasses import dataclass
from string import Template
from typing import Any, Callable, Dict, List, Optional, Sequence

import nest_asyncio
import pandas as pd

from .llm.gemini import get_gemini_json
from .llm.claude import get_claude_json
from .llm.gpt import get_gpt_json
from .utils import load_template

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    # Languages
    source_lang: str
    target_lang: str

    # LLM selection
    model: str
    api_key: str = ""
    client: Any = None
    temp: float = 0.0

    # Prompt templates
    #   • `system` is rendered *once* at start-up
    #   • `prompt` is rendered per-row during batch processing
    prompt: Template | str = ""
    system: Template | str = ""

    # Batch control
    batch_size: int = 8
    wait_seconds: int = 10  # pause between batches to respect rate limits

    # Provider-specific override (optional)
    max_tokens: Optional[int] = None


def dedupe_tb_matches(tb_lists) -> list[dict]:
    """
    Flatten a collection of TB match lists and dedupe by (source, target).
    Only `source` and `target` keys are preserved in the output.
    """
    seen = set()
    merged: list[dict] = []
    if tb_lists is None:
        return merged

    for matches in tb_lists:
        if not isinstance(matches, (list, tuple)):
            continue
        for m in matches:
            if not isinstance(m, dict):
                continue
            src = m.get("src") or m.get("source")
            trg = m.get("trg") or m.get("target")
            if src is None or trg is None:
                continue
            key = (str(src), str(trg))
            if key in seen:
                continue
            seen.add(key)
            merged.append({"source": key[0], "target": key[1]})
    return merged


MODEL_ROUTER: Dict[str, Callable] = {
    "gemini": get_gemini_json,
    "claude": get_claude_json,
    "gpt": get_gpt_json,
    "openai": get_gpt_json,
    "anthropic": get_claude_json,
}


def _llm_runner(model_name: str) -> Callable:
    key = next((k for k in MODEL_ROUTER if k in model_name.lower()), None)
    if not key:
        raise ValueError(f"No LLM handler for model '{model_name}'")
    return MODEL_ROUTER[key]


def _render_tmpl(tpl, **kw) -> str:
    return tpl.safe_substitute(**kw) if hasattr(tpl, "substitute") else tpl


def _get_seg_id(obj: dict) -> str:
    return str(obj.get("seg_id") or obj.get("id") or "")


def _init_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Batch_ID"] = -1
    out["Batch_Input"] = None
    out["Agent1_Output"] = None
    out["Agent1_Edited_Target"] = out["Target"]
    out["Agent1_Errors"] = [[] for _ in range(len(out))]
    out["Agent2_Status"] = ""
    out["Agent2_Output"] = None
    out["Final_Target"] = out["Target"]
    out["Final_Errors"] = [[] for _ in range(len(out))]
    return out


def build_batch_payload(
    batch_df: pd.DataFrame,
    *,
    include_context: bool = True,
    tb_column: str = "TB Matches",
) -> dict:
    """
    Construct the JSON payload expected by the new system prompts.
    """
    segments = []
    for _, row in batch_df.iterrows():
        seg_obj = {
            "id": str(row["Segment_ID"]),
            "source": str(row["Source"]),
            "target": str(row["Target"]),
        }
        char_lim = row.get("Character_Limit")
        if pd.notna(char_lim) and str(char_lim).strip():
            try:
                seg_obj["character_limit"] = int(float(char_lim))
            except (ValueError, TypeError):
                pass
        segments.append(seg_obj)

    context_str = ""
    if include_context:
        before = str(batch_df.iloc[0].get("Context_Before", "") or "").strip()
        after = str(batch_df.iloc[-1].get("Context_After", "") or "").strip()
        current_sources = batch_df["Source"].fillna("").astype(str).tolist()
        middle = " || ".join(current_sources)
        parts = [p for p in (before, middle, after) if p]
        context_str = " || ".join(parts)

    tb_lists = batch_df.get(tb_column, []).tolist()
    unique_tb = dedupe_tb_matches(tb_lists)

    return {
        "segments": segments,
        "context": context_str if include_context else None,
        "tb_matches": unique_tb,
    }


async def _call_runner(cfg, prompt_parts: dict, runner: Callable):
    responses = await runner(
        api_key=cfg.api_key,
        llm_model=cfg.model,
        temp=cfg.temp,
        prompt_parts_list=[prompt_parts],
    )
    return responses[0] if responses else {"error": "Empty response"}


def generate_batches(df: pd.DataFrame, target_size: int) -> List[List[int]]:
    """
    Generates batches of indices while respecting Main Segment affinity.
    Rules:
    1. All segments with the same 'Main_Segment' ID must stay in the same batch.
    2. Try to fill batch up to 'target_size'.
    3. If a single Main Segment group > target_size, it becomes a batch alone.
    """
    if df.empty:
        return []

    groups: Dict[str, List[int]] = {}
    for idx, row in df.iterrows():
        m_id = row.get("Main_Segment", f"unknown_{idx}")
        if m_id not in groups:
            groups[m_id] = []
        groups[m_id].append(idx)

    batches: List[List[int]] = []
    current_batch: List[int] = []

    for _, indices in groups.items():
        group_len = len(indices)
        curr_len = len(current_batch)

        if curr_len == 0:
            current_batch.extend(indices)
            continue

        if curr_len + group_len > target_size:
            batches.append(current_batch)
            current_batch = list(indices)
        else:
            current_batch.extend(indices)

    if current_batch:
        batches.append(current_batch)

    return batches


def _apply_agent1_batch(
    df: pd.DataFrame,
    batch_indices: Sequence[int],
    batch_id: int,
    payload: dict,
    resp: dict | None,
) -> None:
    """
    Persist Agent-1 results onto the dataframe row-level columns.
    """
    idx_list = list(batch_indices)
    df.loc[idx_list, "Batch_ID"] = batch_id
    df.loc[idx_list, "Batch_Input"] = json.dumps(payload, ensure_ascii=False)
    df.loc[idx_list, "Agent1_Output"] = json.dumps(resp or {}, ensure_ascii=False)

    id_lookup = {str(df.loc[i, "Segment_ID"]): i for i in idx_list}

    if not resp or "error" in resp:
        df.loc[idx_list, "Feedback Error"] = resp.get("error", "APIError") if isinstance(resp, dict) else "APIError"
        return

    returned = resp.get("segments", []) or []
    for seg in returned:
        seg_id = _get_seg_id(seg)
        if seg_id not in id_lookup:
            continue
        row_idx = id_lookup[seg_id]
        df.at[row_idx, "Agent1_Edited_Target"] = seg.get("edited_target", df.at[row_idx, "Target"])
        df.at[row_idx, "Agent1_Errors"] = seg.get("errors", [])


async def run_lqa_first_pass_async(
    df: pd.DataFrame,
    cfg: TaskConfig,
    *,
    batch_segments=10,
    max_concurrency=5,
    wait_seconds=0,
    include_context=True,
) -> pd.DataFrame:
    """
    Process Agent-1 using affinity-aware batching.
    """
    df_work = _init_columns(df)
    runner = _llm_runner(cfg.model)
    system_msg = _render_tmpl(cfg.system)

    batch_indices_list = generate_batches(df_work, batch_segments)
    if not batch_indices_list:
        return df_work

    async def _process_batch(indices: List[int], batch_id: int):
        batch_df = df_work.loc[indices]
        payload = build_batch_payload(batch_df, include_context=include_context)

        prompt_parts = {
            "system": system_msg,
            "prompt": _render_tmpl(cfg.prompt, payload=json.dumps(payload, ensure_ascii=False)),
        }

        if batch_id == 1:
            logger.info("Agent1 system prompt:\n%s", prompt_parts["system"])
            logger.info("Agent1 user prompt:\n%s", prompt_parts["prompt"])

        resp = await _call_runner(cfg, prompt_parts, runner)
        _apply_agent1_batch(df_work, indices, batch_id, payload, resp)

    await _process_batch(batch_indices_list[0], 1)

    if wait_seconds:
        await asyncio.sleep(wait_seconds)

    if len(batch_indices_list) > 1:
        remaining = [(b_idx, internal_id) for internal_id, b_idx in enumerate(batch_indices_list[1:], start=2)]

        for i in range(0, len(remaining), max_concurrency):
            window = remaining[i : i + max_concurrency]
            await asyncio.gather(*(_process_batch(b_idxs, bid) for b_idxs, bid in window))

            if wait_seconds:
                await asyncio.sleep(wait_seconds)

    return df_work


def _apply_agent2_batch(
    df: pd.DataFrame,
    batch_indices: Sequence[int],
    resp: dict | None,
) -> None:
    idx_list = list(batch_indices)
    status = (resp or {}).get("status", "Error") if isinstance(resp, dict) else "Error"
    df.loc[idx_list, "Agent2_Status"] = status
    df.loc[idx_list, "Agent2_Output"] = json.dumps(resp or {}, ensure_ascii=False)

    id_lookup = {str(df.loc[i, "Segment_ID"]): i for i in idx_list}

    if status == "Accepted":
        for i in idx_list:
            df.at[i, "Final_Target"] = df.at[i, "Agent1_Edited_Target"]
            df.at[i, "Final_Errors"] = df.at[i, "Agent1_Errors"]
        return

    if status == "Rejected" or status == "Error":
        for i in idx_list:
            df.at[i, "Final_Target"] = df.at[i, "Target"]
            df.at[i, "Final_Errors"] = []
        return

    segs = (resp or {}).get("segments", []) or []
    seg_map = {_get_seg_id(seg): seg for seg in segs}

    for seg_id, row_idx in id_lookup.items():
        seg_resp = seg_map.get(seg_id, {})
        df.at[row_idx, "Final_Target"] = seg_resp.get("edited_target", df.at[row_idx, "Target"])
        df.at[row_idx, "Final_Errors"] = seg_resp.get("errors", [])


async def run_lqa_review_pass_async(
    df: pd.DataFrame,
    cfg: TaskConfig,
    *,
    max_concurrency=5,
    wait_seconds=0,
) -> pd.DataFrame:
    df_work = df.copy()
    runner = _llm_runner(cfg.model)
    system_msg = _render_tmpl(cfg.system)

    if "Batch_ID" not in df_work.columns:
        logger.warning("Batch_ID column missing; skipping Agent-2.")
        return df_work

    valid_batches = df_work[df_work["Batch_ID"] != -1]
    batch_ids = sorted(valid_batches["Batch_ID"].unique())

    if not batch_ids:
        return df_work

    async def _process_batch(batch_id: int):
        batch_df = df_work[df_work["Batch_ID"] == batch_id]
        if batch_df.empty:
            return

        input_payload = batch_df["Batch_Input"].iloc[0] or "{}"
        agent1_payload = batch_df["Agent1_Output"].iloc[0] or "{}"

        prompt_parts = {
            "system": system_msg,
            "prompt": _render_tmpl(
                cfg.prompt,
                input_payload=input_payload,
                agent1_payload=agent1_payload,
            ),
        }

        if batch_id == min(batch_ids):
            logger.info("Agent2 system prompt:\n%s", prompt_parts["system"])
            logger.info("Agent2 user prompt:\n%s", prompt_parts["prompt"])

        resp = await _call_runner(cfg, prompt_parts, runner)
        _apply_agent2_batch(df_work, batch_df.index, resp)

    first_id = batch_ids[0]
    await _process_batch(first_id)

    if wait_seconds:
        await asyncio.sleep(wait_seconds)

    remaining_ids = batch_ids[1:]

    for i in range(0, len(remaining_ids), max_concurrency):
        window = remaining_ids[i : i + max_concurrency]
        await asyncio.gather(*(_process_batch(bid) for bid in window))
        if wait_seconds and i + max_concurrency < len(remaining_ids):
            await asyncio.sleep(wait_seconds)

    return df_work


def run_lqa_first_pass(df, cfg, **kw):
    nest_asyncio.apply()
    return asyncio.run(run_lqa_first_pass_async(df, cfg, **kw))


def run_lqa_review_pass(df, cfg, **kw):
    nest_asyncio.apply()
    return asyncio.run(run_lqa_review_pass_async(df, cfg, **kw))
