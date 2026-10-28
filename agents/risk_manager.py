"""Risk manager role (advisory only; the hard layer is code)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

from agents.llm import CachedLLM
from agents.prompts import build_prompt
from agents.schemas import Critique, CritiqueBundle, Proposal
from core.context import ContextPacket


@dataclass
class RiskManagerResult:
    bundle: CritiqueBundle
    prompt_sha: str
    cache_key: str
    cached: bool
    usage: dict
    flags: list[str]


def run_risk_manager(
    client: CachedLLM,
    *,
    packet: ContextPacket,
    proposals: list[Proposal],
    params: Mapping[str, Any],
    schema_json: str,
    as_of: date,
) -> RiskManagerResult:
    payload = {"proposals": [p.model_dump() for p in proposals]}
    prompt = build_prompt(
        "risk_manager",
        schema_name="CritiqueBundle",
        context=packet.context,
        payload=payload,
        params=params,
        schema_json=schema_json,
    )
    result = client.call(
        role="risk_manager",
        system=prompt.system,
        user=prompt.user,
        schema=CritiqueBundle,
        params=params,
        as_of=as_of,
        context_hash=packet.context_hash,
        payload=prompt.payload,
    )
    bundle: CritiqueBundle = result.value  # type: ignore[assignment]
    codes = {p.asset for p in proposals}
    flags: list[str] = []
    clean: list[Critique] = []
    for critique in bundle.critiques:
        if critique.target not in codes:
            flags.append(f"critique_unknown_target:{critique.target}")
            continue
        if critique.recognizes_asset:
            flags.append("self_reported_recognition:risk_manager")
        clean.append(critique)
    bundle = bundle.model_copy(update={"critiques": clean})
    prompt_sha = hashlib.sha256((prompt.system + "\n" + prompt.user).encode()).hexdigest()
    return RiskManagerResult(
        bundle=bundle,
        prompt_sha=prompt_sha,
        cache_key=result.key,
        cached=result.cached,
        usage=result.usage.to_dict(),
        flags=flags,
    )
