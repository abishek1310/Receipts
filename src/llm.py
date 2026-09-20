"""Provider adapter (CLAUDE.md §3: "Keep behind one adapter, swappable").

One `complete()` call, three implementations: Anthropic, OpenAI, and a scripted
stub used by the eval harness so §11 can measure the validator without spending
tokens or needing a key.

Two notes on the stack, both deliberate departures worth knowing about:

* **The Anthropic call uses the official `anthropic` SDK, not LangChain.** §3 asks
  for "Anthropic Claude (or OpenAI) via LangChain", but the requirement it states
  in the same row — one adapter, swappable — is what this file provides. Going
  direct keeps us off a wrapper that lags new model parameters, and LangGraph (the
  part of the stack §3 actually cares about) is unaffected.
* **`temperature` is not sent to current Claude models.** §7 asks for ~0.7.
  Sampling parameters were removed from Claude Opus 5, Sonnet 5 and the 4.7/4.8
  family — sending `temperature` returns a 400. The knob that replaced it is
  `output_config.effort`. §7's actual point still holds and is honoured: safety
  comes from the validator, not from clamping the sampler. Temperature is still
  sent to models that accept it (Haiku 4.5, OpenAI).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from src.config import Settings, get_settings

# Sampling parameters were removed on these families and return a 400.
_NO_SAMPLING_PREFIXES = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
)


def accepts_temperature(model: str) -> bool:
    return not model.startswith(_NO_SAMPLING_PREFIXES)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMError(RuntimeError):
    """A provider call failed. Distinct from a validation failure, which is a value."""


class LLMClient(Protocol):
    def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse: ...


# --------------------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------------------


class AnthropicClient:
    """Claude via the official SDK."""

    def __init__(self, settings: Settings):
        import anthropic

        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._settings = settings

    def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        model = self._settings.anthropic_model
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self._settings.max_tokens,
            # The evidence block is identical across the retries in one turn, so
            # caching the system prompt makes regeneration cheap (§6.3).
            "system": [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": self._settings.effort},
        }
        if json_schema is not None:
            kwargs["output_config"]["format"] = {
                "type": "json_schema",
                "schema": json_schema,
            }
        if accepts_temperature(model):
            kwargs["temperature"] = self._settings.temperature

        try:
            response = self._client.messages.create(**kwargs)
        except self._anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic rejected the API key") from exc
        except self._anthropic.RateLimitError as exc:
            raise LLMError("Anthropic rate limit hit — retry in a moment") from exc
        except self._anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Anthropic API") from exc

        if response.stop_reason == "refusal":
            raise LLMError("The model declined this request")

        text = "".join(b.text for b in response.content if b.type == "text")
        return LLMResponse(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


# --------------------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------------------


class OpenAIClient:
    """The swappable alternative §3 allows for. Same contract, different vendor."""

    def __init__(self, settings: Settings):
        from openai import OpenAI

        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._settings = settings

    def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        import openai

        kwargs: dict[str, Any] = {
            "model": self._settings.openai_model,
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "copy_blocks",
                    "strict": True,
                    "schema": json_schema,
                },
            }
        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as exc:
            raise LLMError("OpenAI rejected the API key") from exc
        except openai.RateLimitError as exc:
            raise LLMError("OpenAI rate limit hit — retry in a moment") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"OpenAI API error {exc.status_code}") from exc
        except openai.APIConnectionError as exc:
            raise LLMError("Could not reach the OpenAI API") from exc

        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=response.model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


# --------------------------------------------------------------------------------------
# Scripted stub — no network
# --------------------------------------------------------------------------------------


class ScriptedClient:
    """Replays canned responses in order. For eval fixtures and offline demos.

    §14 forbids LLM calls in tests; this is how the graph and the eval harness get
    exercised end to end without one.
    """

    def __init__(self, responses: Sequence[str]):
        self._responses = list(responses)
        self._i = 0
        self.calls: list[tuple[str, str]] = []

    def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append((system, user))
        if self._i >= len(self._responses):
            raise LLMError(
                f"ScriptedClient exhausted after {len(self._responses)} response(s)"
            )
        text = self._responses[self._i]
        self._i += 1
        if not isinstance(text, str):
            text = json.dumps(text)
        return LLMResponse(text=text, model="scripted")


def get_client(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    if settings.llm_provider == "openai":
        return OpenAIClient(settings)
    return AnthropicClient(settings)
