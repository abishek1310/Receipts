"""Configuration (CLAUDE.md §3, §14).

`pydantic-settings` + `.env`. Keys are never logged and never committed.

On Streamlit Community Cloud there is no `.env`, so `st.secrets` is folded into the
environment before settings are read — see `load_streamlit_secrets()`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- provider -----------------------------------------------------------------
    # §3 names Anthropic and OpenAI. Gemini is here because the team has neither
    # key and §3 also requires a deployed Streamlit Community Cloud URL, which
    # rules out a locally hosted model. "One adapter, swappable" is the part of
    # §3 that matters, and it held — only src/llm.py and this file changed.
    llm_provider: Literal["anthropic", "openai", "gemini"] = "gemini"

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None

    anthropic_model: str = "claude-opus-5"
    openai_model: str = "gpt-4o"
    gemini_model: str = "gemini-3.6-flash"

    # Comma-separated models to fail over to. Each Gemini model has its own
    # 20-requests-per-day free-tier bucket and its own overload profile, so a
    # second and third name is the difference between a 503 ending the demo and
    # nobody noticing.
    gemini_fallback_models: str = "gemini-3.7-flash,gemini-3.5-flash,gemini-3-flash-preview"

    # Gemini 2.5 thinks by default and thinking tokens come out of the output
    # allowance, which can return an empty body. 0 disables it; raise it if
    # attribution quality slips.
    gemini_thinking_budget: int = 0

    # §7 asks for ~0.7. Current Claude models removed sampling parameters and
    # return a 400 if sent, so `src.llm` only forwards this to models that accept
    # it. Either way the point of §7 stands: the validator provides safety, not a
    # low temperature — do not turn this down hoping for fewer rejections.
    temperature: float = 0.7

    # The replacement knob on current models. "medium" keeps the demo responsive;
    # raise to "high" if attribution quality slips.
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"

    max_tokens: int = 8000

    # Transient-failure retries at the provider layer (503 / 429). Distinct from
    # max_retries, which is the §6.3 citation-regeneration budget.
    provider_retries: int = Field(default=3, ge=0, le=6)

    # --- retrieval ----------------------------------------------------------------
    retrieval_k: int = Field(default=40, ge=5, le=200)

    # --- validation ---------------------------------------------------------------
    # Two retries after the first attempt, then the block is surfaced as
    # UNSUPPORTED rather than dropped (§6.3).
    max_retries: int = Field(default=2, ge=0, le=5)

    @property
    def gemini_fallback_list(self) -> list[str]:
        return [m.strip() for m in self.gemini_fallback_models.split(",") if m.strip()]

    @property
    def api_key(self) -> str | None:
        return {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "gemini": self.gemini_api_key,
        }[self.llm_provider]

    @property
    def model(self) -> str:
        return {
            "anthropic": self.anthropic_model,
            "openai": self.openai_model,
            "gemini": self.gemini_model,
        }[self.llm_provider]

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


def load_streamlit_secrets() -> None:
    """Copy `st.secrets` into the environment so Settings picks them up.

    No-op outside Streamlit, or when no secrets file exists. Called by `app.py`
    before the first `get_settings()`.
    """
    try:
        import streamlit as st

        secrets = dict(st.secrets)
    except Exception:  # noqa: BLE001 - any failure here just means "no secrets"
        return
    for key, value in secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key.upper(), value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
