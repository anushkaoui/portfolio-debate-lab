"""Trader role: combine debate into target weights."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

from agents.llm import CachedLLM
from agents.prompts import build_prompt
from agents.schemas import (
    AnalystResponse,
    Critique,
    Decision,
    Proposal,
)
from core.context import ContextPacket


@dataclass
class TraderResult:
    decision: Decision
    prompt_sha: str
    cache_key: str
    cached: bool
    usage: dict
    flags: list[str]


def run_trader(
    client: CachedLLM,
    *,
    packet: ContextPacket,
    proposals: list[Proposal],
    critiques: list[Critique],
    responses: list[AnalystResponse],
    params: Mapping[str, Any],
    schema_json: str,
    as_of: date,
) -> TraderResult:
    payload = {
        "proposals": [p.model_dump() for p in proposals],
        "critiques": [c.model_dump() for c in critiques],
        "responses": [r.model_dump() for r in responses],
    }
    prompt = build_prompt(
        "trader",
        schema_name="Decision",
        context=packet.context,
        payload=payload,
        params=params,
        schema_json=schema_json,
    )
    result = client.call(
        role="trader",
        system=prompt.system,
        user=prompt.user,
        schema=Decision,
        params=params,
        as_of=as_of,
        context_hash=packet.context_hash,
        payload=prompt.payload,
    )
    decision: Decision = result.value  # type: ignore[assignment]
    codes = set(packet.context["universe"])
    flags: list[str] = []
    cleaned = {}
    for code, weight in decision.target_weights.items():
        if code not in codes:
            flags.append(f"unknown_weight:{code}")
            continue
        cleaned[code] = max(0.0, min(1.0, float(weight)))
    total = sum(cleaned.values())
    if total > 1.0:
        flags.append("weights_scaled_to_gross_one")
        cleaned = {k: v / total for k, v in cleaned.items()}
    if decision.recognizes_asset:
        flags.append("self_reported_recognition:trader")
    decision = decision.model_copy(update={"target_weights": cleaned})
    prompt_sha = hashlib.sha256((prompt.system + "\n" + prompt.user).encode()).hexdigest()
    return TraderResult(
        decision=decision,
        prompt_sha=prompt_sha,
        cache_key=result.key,
        cached=result.cached,
        usage=result.usage.to_dict(),
        flags=flags,
    )
