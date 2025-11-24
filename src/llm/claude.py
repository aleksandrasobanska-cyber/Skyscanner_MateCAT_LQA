import asyncio
import logging
from typing import Any, Dict, List

from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .common import extract_json

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=60),
    reraise=True,
)
async def _call_claude_once_json(
    api_key: str,
    llm_model: str,
    temp: float,
    parts: Dict[str, str],
    client: AsyncAnthropic,
) -> Dict[str, Any]:
    """
    One Anthropic call. Tenacity handles retries externally.
    """
    msg = await client.messages.create(
        model=llm_model,
        max_tokens=18_000,
        temperature=temp,
        system=[
            {
                "type": "text",
                "text": parts["system"],
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": parts["prompt"]}],
            }
        ],
        thinking={"type": "enabled", "budget_tokens": 16_000},
    )

    raw = msg.content[1].text
    return extract_json(raw)


async def get_claude_json(
    api_key: str,
    llm_model: str,
    temp: float,
    prompt_parts_list: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """
    Returns list aligned with *prompt_parts_list*.
    On success    -> parsed dict
    On final fail -> {"error": "..."} (keeps position)
    """
    client = AsyncAnthropic(api_key=api_key)

    async def _safe_call(parts):
        try:
            return await _call_claude_once_json(api_key, llm_model, temp, parts, client=client)
        except Exception as exc:  # noqa: BLE001
            logger.error("Claude task failed: %s", exc)
            return {"error": str(exc)}

    try:
        return await asyncio.gather(*(_safe_call(p) for p in prompt_parts_list))
    finally:
        await client.close()
