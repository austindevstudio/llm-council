"""Configuration for the LLM Council.

Council roster lives in council.yaml at the repo root — edit that file,
not this one. This module just loads it.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key (still used by default — council.yaml can also route
# individual advisors through direct provider keys, see council.yaml comments)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "council.yaml"


def _load_council_config() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"council.yaml not found at {_CONFIG_PATH}. "
            "This file defines your council roster and is required."
        )
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


_raw = _load_council_config()

if not _raw.get("council"):
    raise ValueError("council.yaml must define at least one advisor under 'council:'")

# List of litellm model strings, e.g. "openrouter/openai/gpt-5.1".
# Kept as a flat list (same shape as the original COUNCIL_MODELS) so
# council.py and main.py don't need to change.
COUNCIL_MODELS = [m["litellm_model"] for m in _raw["council"]]

# Friendly id -> litellm model string, and the reverse, for display purposes.
MODEL_LABELS = {m["litellm_model"]: m["id"] for m in _raw["council"]}

CHAIRMAN_MODEL = _raw["chairman"]["litellm_model"]

TITLE_MODEL = _raw.get("title_model", {}).get(
    "litellm_model", "openrouter/google/gemini-2.5-flash"
)

# skip_and_flag (default): a dead/failing model is dropped from the round,
# not fatal to the whole request. fail_fast: old all-or-nothing behavior.
FALLBACK_ON_ERROR = _raw.get("fallback", {}).get("on_error", "skip_and_flag") == "skip_and_flag"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
