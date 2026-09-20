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
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    anthropic_model: str = "claude-sonnet-5"
    openai_model: str = "gpt-4o"

    # 0.7 for copy quality. The validator is what provides safety, not a low
    # temperature (§7) — do not turn this down hoping for fewer rejections.
    temperature: float = 0.7
    max_tokens: int = 1400

    # --- retrieval ----------------------------------------------------------------
    retrieval_k: int = Field(default=40, ge=5, le=200)

    # --- validation ---------------------------------------------------------------
    # Two retries after the first attempt, then the block is surfaced as
    # UNSUPPORTED rather than dropped (§6.3).
    max_retries: int = Field(default=2, ge=0, le=5)

    @property
    def api_key(self) -> str | None:
        return (
            self.anthropic_api_key
            if self.llm_provider == "anthropic"
            else self.openai_api_key
        )

    @property
    def model(self) -> str:
        return (
            self.anthropic_model
            if self.llm_provider == "anthropic"
            else self.openai_model
        )

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
