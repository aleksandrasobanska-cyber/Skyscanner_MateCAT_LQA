import asyncio
import logging
from typing import Any, Dict, List

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .common import extract_json

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=60), reraise=True)
async def _call_gemini_once_json(
    llm_model: str,
    temp: float,
    parts: Dict[str, str],
    client: genai.Client,
) -> Dict[str, Any]:
    """
    One Gemini API call. Tenacity handles retries externally.
    """
    config_params = {
        "system_instruction": parts["system"],
        "temperature": temp,
        "max_output_tokens": 24_000,
        "response_mime_type": "text/plain",
    }
    if "2.5" in llm_model:
        config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=20_000)

    response = await client.aio.models.generate_content(
        model=llm_model,
        contents=parts["prompt"],
        config=types.GenerateContentConfig(**config_params),
    )
    if not response.text:
        raise ValueError("Empty response from Gemini")

    raw = response.text
    return extract_json(raw)


async def get_gemini_json(
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

    client = genai.Client(api_key=api_key)

    async def _safe_call(parts):
        try:
            return await _call_gemini_once_json(llm_model, temp, parts, client)
        except Exception as exc:  # noqa: BLE001
            logging.error("Gemini task failed: %s", exc)
            return {"error": str(exc)}

    return await asyncio.gather(*(_safe_call(p) for p in prompt_parts_list))
