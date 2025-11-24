import asyncio
import logging
from typing import Any, Dict, List

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .common import extract_json

logger = logging.getLogger(__name__)


def _response_output_text(resp) -> str:
    """
    Robustly extract string output from Responses API result.
    Prefers .output_text; falls back to concatenating text parts.
    """
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt:
        return txt

    out = []
    try:
        for block in getattr(resp, "output", []) or []:
            for c in getattr(block, "content", []) or []:
                t = getattr(c, "text", None)
                if isinstance(t, str):
                    out.append(t)
    except Exception:
        pass
    return "".join(out) if out else str(resp)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=60),
    reraise=True,
)
async def _call_gpt_once_json(
    api_key: str,
    llm_model: str,
    temp: float,
    parts: Dict[str, str],
    client: AsyncOpenAI,
) -> Dict[str, Any]:
    """
    One Responses API request; Tenacity handles per-call retry.
    """
    kwargs = dict(
        model=llm_model,
        input=[
            {"role": "developer", "content": [{"type": "input_text", "text": parts["system"]}]},
            {"role": "user", "content": [{"type": "input_text", "text": parts["prompt"]}]},
        ],
        text={"format": {"type": "text"}, "verbosity": "medium"},
        reasoning={"effort": "high"},
    )

    resp = await client.responses.create(**kwargs)
    raw_text = _response_output_text(resp)
    return extract_json(raw_text)


async def get_gpt_json(
    api_key: str,
    llm_model: str,
    temp: float,
    prompt_parts_list: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """
    Returns list aligned with *prompt_parts_list*:
      • parsed dict on success
      • {"error": "..."} on final failure / JSON parse error
    """
    client = AsyncOpenAI(api_key=api_key)

    async def _safe_call(parts):
        try:
            return await _call_gpt_once_json(api_key, llm_model, temp, parts, client=client)
        except Exception as exc:  # noqa: BLE001
            logger.error("GPT task failed: %s", exc)
            return {"error": str(exc)}

    try:
        tasks = [_safe_call(p) for p in prompt_parts_list]
        return await asyncio.gather(*tasks, return_exceptions=False)
    finally:
        await client.close()
