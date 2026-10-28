"""Pydantic v2 contracts for every agent message."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Proposal(StrictModel):
    agent: str = Field(min_length=1, max_length=60)
    asset: str = Field(min_length=1, max_length=20)
    direction: Literal["increase", "decrease", "hold"]
    target_weight_delta: float = Field(ge=-1.0, le=1.0)
    conviction: float = Field(ge=0.0, le=1.0)
    horizon_days: int = Field(ge=1, le=252)
    thesis: str = Field(min_length=1, max_length=600)
    evidence: list[str] = Field(min_length=1, max_length=12)
    what_would_change_my_mind: str = Field(min_length=1, max_length=400)
    recognizes_asset: bool


class ProposalBundle(StrictModel):
    proposals: list[Proposal] = Field(default_factory=list, max_length=5)
    summary: str = Field(default="", max_length=600)
    recognizes_asset: bool = False


class Critique(StrictModel):
    target: str = Field(min_length=1, max_length=20)
    objections: list[str] = Field(min_length=1, max_length=8)
    severity: Literal["low", "medium", "high"]
    suggested_cap: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[str] = Field(min_length=1, max_length=12)
    recognizes_asset: bool


class CritiqueBundle(StrictModel):
    critiques: list[Critique] = Field(default_factory=list, max_length=12)
    summary: str = Field(default="", max_length=600)
    recognizes_asset: bool = False


class AnalystResponse(StrictModel):
    target: str = Field(min_length=1, max_length=20)
    accept_critique: bool
    revised_delta: float | None = Field(default=None, ge=-1.0, le=1.0)
    argument: str = Field(min_length=1, max_length=500)
    evidence: list[str] = Field(min_length=1, max_length=12)
    recognizes_asset: bool


class ResponseBundle(StrictModel):
    responses: list[AnalystResponse] = Field(default_factory=list, max_length=12)
    recognizes_asset: bool = False


class Decision(StrictModel):
    target_weights: dict[str, float] = Field(default_factory=dict)
    rationale: str = Field(min_length=1, max_length=1200)
    dissent_noted: bool
    evidence: list[str] = Field(min_length=1, max_length=12)
    recognizes_asset: bool

    @field_validator("target_weights")
    @classmethod
    def _weights_valid(cls, value: dict[str, float]) -> dict[str, float]:
        for ticker, weight in value.items():
            if not isinstance(ticker, str) or not ticker:
                raise ValueError("target_weights keys must be non-empty asset codes")
            if weight < 0 or weight > 1:
                raise ValueError(f"target weight for {ticker} must be in [0, 1]")
        if sum(value.values()) > 1.0 + 1e-6:
            raise ValueError("target weights must sum to at most 1.0 (no leverage)")
        return value

    @model_validator(mode="after")
    def _no_nan(self) -> "Decision":
        for ticker, weight in self.target_weights.items():
            if weight != weight:  # NaN
                raise ValueError(f"NaN weight for {ticker}")
        return self


class IdentificationGuess(StrictModel):
    recognizes_asset: bool
    guessed_asset: str = "unknown"
    guessed_period: str = "unknown"
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1, max_length=500)


class ProbePrediction(StrictModel):
    next_move: Literal["up", "down", "flat", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=500)
    recognizes_asset: bool


SCHEMAS = {
    "proposal_bundle": ProposalBundle,
    "critique_bundle": CritiqueBundle,
    "response_bundle": ResponseBundle,
    "decision": Decision,
    "identification_guess": IdentificationGuess,
    "probe_prediction": ProbePrediction,
}
