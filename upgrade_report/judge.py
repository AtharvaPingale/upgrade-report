"""The pinned judge model.

Judges are treated as fixed instruments: the model is pinned in config, is the
same object for every arm, is part of the score cache key, and (by default)
every response is checked against the pinned model id so a provider-side
alias change or fallback fails loudly instead of skewing scores.

A judge client is a callable:

    def complete(*, model: str, prompt: str, system: str | None,
                 temperature: float | None, max_tokens: int) -> str | JudgeResponse | dict

`judge.client: anthropic` uses the built-in Anthropic adapter below.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .cache import stable_hash
from .config import JudgeConfig
from .errors import ConfigError, JudgeDriftError, JudgeError
from .loader import load_object


@dataclass
class JudgeResponse:
    text: str
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


def _coerce(resp: Any) -> JudgeResponse:
    if isinstance(resp, JudgeResponse):
        return resp
    if isinstance(resp, str):
        return JudgeResponse(text=resp)
    if isinstance(resp, dict) and isinstance(resp.get("text"), str):
        return JudgeResponse(text=resp["text"], model=resp.get("model"),
                             input_tokens=int(resp.get("input_tokens") or 0),
                             output_tokens=int(resp.get("output_tokens") or 0))
    raise JudgeError(f"judge client returned {type(resp).__name__}; expected str, dict or JudgeResponse")


class Judge:
    def __init__(self, cfg: JudgeConfig, client: Callable[..., Any] | None):
        self.cfg = cfg
        self._client = client
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    @classmethod
    def from_config(cls, cfg: JudgeConfig, base_dir: Path) -> "Judge":
        if cfg.client is None:
            return cls(cfg, None)
        if cfg.client == "anthropic":
            return cls(cfg, anthropic_client())
        fn = load_object(cfg.client, base_dir)
        if not callable(fn):
            raise ConfigError(f"judge.client {cfg.client!r} is not callable")
        return cls(cfg, fn)

    @property
    def model(self) -> str:
        return self.cfg.model

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def key(self) -> str:
        return stable_hash(self.cfg.describe(), 16)

    def complete(self, prompt: str, system: str | None = None) -> str:
        if self._client is None:
            raise ConfigError("an evaluator needs the judge, but `judge.client` is not configured")
        resp = _coerce(self._client(model=self.cfg.model, prompt=prompt, system=system,
                                    temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens))
        if self.cfg.verify_response_model and resp.model and resp.model != self.cfg.model:
            raise JudgeDriftError(
                f"judge is pinned to {self.cfg.model!r} but the provider answered with {resp.model!r}"
            )
        with self._lock:
            self.calls += 1
            self.input_tokens += resp.input_tokens
            self.output_tokens += resp.output_tokens
        return resp.text

    def usage(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


def anthropic_client() -> Callable[..., JudgeResponse]:
    try:
        import anthropic
    except ImportError as exc:
        raise ConfigError("judge.client 'anthropic' needs the anthropic package: "
                          "pip install 'upgrade-report[anthropic]'") from exc
    client = anthropic.Anthropic()

    def complete(*, model: str, prompt: str, system: str | None, temperature: float | None,
                 max_tokens: int) -> JudgeResponse:
        kwargs: dict[str, Any] = {"model": model, "max_tokens": max_tokens,
                                  "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        # No server-side model fallbacks here: a fallback would swap the judge.
        response = client.messages.create(**kwargs)
        if response.stop_reason == "refusal":
            raise JudgeError(f"judge {model} refused to grade")
        if response.stop_reason == "max_tokens":
            raise JudgeError(f"judge {model} hit max_tokens={max_tokens}; raise judge.max_tokens")
        text = "".join(block.text for block in response.content if block.type == "text")
        return JudgeResponse(text=text, model=response.model, input_tokens=response.usage.input_tokens,
                             output_tokens=response.usage.output_tokens)

    return complete


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_reply(text: str) -> dict:
    """Pull the JSON object out of a judge reply (tolerates code fences and prose)."""
    match = _JSON_OBJECT.search(text)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    raise JudgeError(f"judge reply is not a JSON object: {text[:200]!r}")
