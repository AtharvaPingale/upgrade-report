"""Configuration models (config.yaml and pricing.yaml)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, field_validator, model_validator

from .errors import ConfigError

DEFAULT_REFUSAL_PATTERNS = [
    r"\bI(?:'m| am) (?:sorry|afraid),? but\b",
    r"\bI can(?:not|'t) (?:help|assist|answer|provide|do that)\b",
    r"\bI(?:'m| am) (?:not able|unable) to\b",
    r"\bI won't be able to\b",
    r"\bI don't have (?:enough )?information to answer\b",
]

ARM_NAMES = ("baseline", "candidate", "adapted_candidate")
CANDIDATE_ARMS = ("candidate", "adapted_candidate")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArmConfig(_Model):
    model: str
    prompt: str

    @field_validator("model", "prompt")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class JudgeConfig(_Model):
    model: str
    # null for judge models that do not accept sampling parameters.
    temperature: float | None = 0.0
    # "anthropic" for the built-in adapter, or "module:function" for your own.
    client: str | None = None
    max_tokens: int = Field(4096, ge=16)
    # Fail when the provider reports a different model than the pinned one.
    verify_response_model: bool = True

    @field_validator("model")
    @classmethod
    def _pinned(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("judge model must be set")
        if re.search(r"(^|[-_:@/])latest($|[-_:@/])", v):
            raise ValueError(f"judge model {v!r} is an alias; pin an exact model version")
        return v

    def describe(self) -> dict:
        return {"model": self.model, "temperature": self.temperature, "client": self.client,
                "max_tokens": self.max_tokens}


class MetricsConfig(_Model):
    primary: str
    guardrails: list[str] = []
    tracked: list[str] = []
    # Override auto-detection (evaluators returning bools are pass/fail).
    kinds: dict[str, Literal["binary", "continuous"]] = {}
    # Plain-language explanations shown in the client report.
    descriptions: dict[str, str] = {}

    @model_validator(mode="after")
    def _unique(self) -> "MetricsConfig":
        names = [self.primary, *self.guardrails, *self.tracked]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"metrics listed in more than one role: {sorted(dupes)}")
        return self

    def roles(self) -> dict[str, str]:
        out = {self.primary: "primary"}
        out.update({m: "guardrail" for m in self.guardrails})
        out.update({m: "tracked" for m in self.tracked})
        return out


class DecisionConfig(_Model):
    primary_margin: float = -0.02
    guardrail_tolerance: Literal["noise"] | float = "noise"
    min_slice_size: int = Field(20, ge=1)
    bootstrap_resamples: int = Field(10_000, ge=200)
    seed: int = 20_240_601

    @field_validator("primary_margin")
    @classmethod
    def _margin(cls, v: float) -> float:
        if v > 0:
            raise ValueError("primary_margin is a non-inferiority margin and must be <= 0")
        return v

    @field_validator("guardrail_tolerance")
    @classmethod
    def _tolerance(cls, v):
        if isinstance(v, float) and v < 0:
            raise ValueError("guardrail_tolerance must be 'noise' or a non-negative number")
        return v


class BehaviorFlags(_Model):
    length_change_pct: float = 25
    refusal_rate_change_pts: float = 3
    cost_increase_pct: float = 15
    format_failure_change_pts: float = 2
    error_rate_change_pts: float = 1
    latency_p95_increase_pct: float | None = None


class PairwiseConfig(_Model):
    enabled: bool = False
    sample_size: int = Field(75, ge=10, le=500)


class BehaviorConfig(_Model):
    refusal_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_REFUSAL_PATTERNS))
    # module:function(answer: str) -> bool, replaces the pattern list.
    refusal_classifier: str | None = None
    # module:function(output: dict) -> bool, True when the output is well-formed.
    format_validator: str | None = None
    pairwise: PairwiseConfig = PairwiseConfig()

    @field_validator("refusal_patterns")
    @classmethod
    def _compile(cls, v: list[str]) -> list[str]:
        for p in v:
            try:
                re.compile(p)
            except re.error as exc:
                raise ValueError(f"invalid refusal pattern {p!r}: {exc}") from exc
        return v


class TrafficConfig(_Model):
    monthly_queries: int = Field(ge=0)


class ReportConfig(_Model):
    formats: list[Literal["md", "json", "html"]] = ["md", "json"]
    redact_inputs: bool = False
    client_title: str | None = None
    # Example IDs to show in the client report; auto-picked when empty.
    client_examples: list[str] = []
    max_examples: int = Field(10, ge=0)


class CacheConfig(_Model):
    enabled: bool = True
    path: str = ".upgrade-report/cache.sqlite"


class LangSmithConfig(_Model):
    # None = on when the dataset comes from LangSmith and an API key is set.
    enabled: bool | None = None
    experiment_prefix: str | None = None


class EstimateConfig(_Model):
    """Fallback token estimates for --dry-run when nothing is cached yet."""

    input_tokens_per_query: int = 1500
    output_tokens_per_query: int = 300
    judge_input_tokens_per_call: int = 1500
    judge_output_tokens_per_call: int = 300


class ScheduleConfig(_Model):
    # "anthropic" or "module:function" returning a list of model ids.
    model_source: str
    model_filter: str | None = None
    state_file: str = ".upgrade-report/seen_models.json"


class Config(_Model):
    project: str
    dataset: str
    dataset_version: str | None = None
    dataset_splits: list[str] | None = None
    target: str
    target_paths: list[str] | None = None

    baseline: ArmConfig
    candidate: ArmConfig
    adapted_candidate: ArmConfig | None = None

    evaluators: list[str] = []
    repeats: int = Field(3, ge=1, le=20)
    concurrency: int = Field(8, ge=1, le=256)
    retries: int = Field(2, ge=0, le=10)

    judge: JudgeConfig
    metrics: MetricsConfig
    decision: DecisionConfig = DecisionConfig()
    behavior_flags: BehaviorFlags = BehaviorFlags()
    behavior: BehaviorConfig = BehaviorConfig()
    traffic: TrafficConfig
    pricing_file: str = "pricing.yaml"
    report: ReportConfig = ReportConfig()
    cache: CacheConfig = CacheConfig()
    langsmith: LangSmithConfig = LangSmithConfig()
    slice_by: str = "tags"
    estimate: EstimateConfig = EstimateConfig()
    schedule: ScheduleConfig | None = None

    _base_dir: Path = PrivateAttr(default=Path("."))
    _source_path: Path | None = PrivateAttr(default=None)

    @field_validator("target")
    @classmethod
    def _target_ref(cls, v: str) -> str:
        if ":" not in v:
            raise ValueError("target must be 'module:function'")
        return v

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def source_path(self) -> Path | None:
        return self._source_path

    def arms(self) -> dict[str, ArmConfig]:
        out = {"baseline": self.baseline, "candidate": self.candidate}
        if self.adapted_candidate is not None:
            out["adapted_candidate"] = self.adapted_candidate
        return out

    def resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self._base_dir / p

    def canonical_hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def with_base_dir(self, base_dir: Path) -> "Config":
        self._base_dir = base_dir
        return self


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  {loc}: {err['msg']}")
    return "\n".join(lines)


def parse_config(data: dict, base_dir: Path, source: Path | None = None) -> Config:
    try:
        cfg = Config.model_validate(data)
    except ValidationError as exc:
        where = f" in {source}" if source else ""
        raise ConfigError(f"invalid config{where}:\n{_format_validation_error(exc)}") from None
    cfg._base_dir = base_dir.resolve()
    cfg._source_path = source
    return cfg


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return parse_config(data, path.parent, path)


class Price(_Model):
    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)

    def cost(self, input_tokens: float, output_tokens: float) -> float:
        return input_tokens / 1e6 * self.input_per_mtok + output_tokens / 1e6 * self.output_per_mtok


def load_pricing(path: Path) -> dict[str, Price]:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must map model ids to prices")
    out: dict[str, Price] = {}
    for model, entry in data.items():
        try:
            out[str(model)] = Price.model_validate(entry)
        except ValidationError as exc:
            raise ConfigError(f"invalid price for {model!r} in {path}:\n{_format_validation_error(exc)}") from None
    return out
