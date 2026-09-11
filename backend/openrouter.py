"""LLM client for making requests to council members.

Despite the filename (kept for zero-diff compatibility with council.py's
imports), this now goes through litellm instead of a raw OpenRouter httpx
call — so models can be routed via OpenRouter OR a direct provider key,
per council.yaml. Function signatures are unchanged from the original.
"""

import asyncio
from typing import Any, Dict, List, Optional

import litellm

from .config import FALLBACK_ON_ERROR, MODEL_LABELS, OPENROUTER_API_KEY

# litellm needs this env var name for openrouter/... model strings
import os
if OPENROUTER_API_KEY:
    os.environ.setdefault("OPENROUTER_API_KEY", OPENROUTER_API_KEY)


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via litellm (OpenRouter or a direct provider,
    depending on how the model string is prefixed in council.yaml).

    Args:
        model: litellm model identifier (e.g. "openrouter/openai/gpt-5.1")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content', 'reasoning_details', and 'cost_usd',
        or None if the call failed and fallback.on_error is skip_and_flag.

    Raises:
        The underlying exception if fallback.on_error is fail_fast.
    """
    label = MODEL_LABELS.get(model, model)

    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            timeout=timeout,
        )

        message = response.choices[0].message

        try:
            cost_usd = litellm.completion_cost(completion_response=response)
        except Exception:
            cost_usd = None

        return {
            "content": message.content,
            "reasoning_details": getattr(message, "reasoning_details", None),
            "cost_usd": cost_usd,
        }

    except Exception as e:
        print(f"[council] advisor '{label}' ({model}) failed: {e}")
        if FALLBACK_ON_ERROR:
            return None
        raise


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of litellm model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed
        and fallback.on_error is skip_and_flag).
    """
    tasks = [query_model(model, messages) for model in models]

    # skip_and_flag: query_model already catches internally and returns None,
    # so gather never sees an exception here.
    # fail_fast: query_model re-raises, and gather (return_exceptions=False,
    # the default) propagates that exception up and kills the whole request —
    # matching the original all-or-nothing behavior.
    responses = await asyncio.gather(*tasks)

    return {model: response for model, response in zip(models, responses)}
