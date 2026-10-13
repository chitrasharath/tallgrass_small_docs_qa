"""Settings for the 4Geeks LiteLLM proxy, read from the environment or a local .env file.

Variable names and defaults match the healthcore_ft_ai_1 repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class LlmSettings:
    base_url: str
    api_key: str
    ssl_verify: bool
    embedding_model: str
    generation_model: str


def llm_settings() -> LlmSettings:
    load_dotenv()
    return LlmSettings(
        base_url=os.getenv("LLM_BASE_URL", "https://llm.4geeks.ai"),
        api_key=os.getenv("LLM_API_KEY", ""),
        # The proxy's certificate chain is sometimes expired; set true when it is valid.
        ssl_verify=os.getenv("LLM_SSL_VERIFY", "false").lower() in {"1", "true", "yes"},
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "downtown-miami/openrouter/perplexity/pplx-embed-v1-0.6b"
        ),
        generation_model=os.getenv(
            "GENERATION_MODEL", "downtown-miami/openrouter/deepseek/deepseek-v4-flash"
        ),
    )
