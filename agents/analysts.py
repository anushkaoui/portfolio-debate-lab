"""Analyst roles: build prompts, call the cached LLM, sanitize proposals."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

from agents.llm import CachedLLM
from agents.prompts import build_prompt
from agents.schemas import Proposal, ProposalBundle
from core.context import ContextPacket


@dataclass
class AnalystResult:
    role: str
    bundle: ProposalBundle
    prompt_sha: str
    cache_key: str
    cached: bool
    usage: dict
    flags: list[str]


def _grounded(context: Mapping[str, Any], key: str) -> bool:
    cur: Any = context
    for part in key.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return False
    return True


def run_analyst(
    client: CachedLLM,
    *,
    style: str,
    packet: ContextPacket,
    params: Mapping[str, Any],
    max_analyst_delta: float,
    schema_json: str,
    as_of: date,
) -> AnalystResult:
    role = f"analyst_{style}"
    prompt = build_prompt(
        role,
        schema_name="ProposalBundle",
        context=packet.context,
        params=params,
        schema_json=schema_json,
    )
    result = client.call(
        role=role,
        system=prompt.system,
        user=prompt.user,
        schema=ProposalBundle,
        params=params,
        as_of=as_of,
        context_hash=packet.context_hash,
        payload=prompt.payload,
    )
    bundle: ProposalBundle = result.value  # type: ignore[assignment]
    codes = set(packet.context["universe"])
    flags: list[str] = []
    clean: list[Proposal] = []
    for prop in bundle.proposals:
        if prop.asset not in codes:
            flags.append(f"unknown_asset:{prop.asset}")
            continue
        if abs(prop.target_weight_delta) > max_analyst_delta + 1e-9:
            flags.append(f"delta_clamped:{prop.asset}")
            prop = prop.model_copy(
                update={
                    "target_weight_delta": max(
                        -max_analyst_delta, min(max_analyst_delta, prop.target_weight_delta)
                    )
                }
            )
        for ev in prop.evidence:
            if not _grounded(packet.context, ev):
                flags.append(f"ungrounded_evidence:{prop.asset}:{ev}")
        clean.append(prop)
    if bundle.recognizes_asset:
        flags.append(f"self_reported_recognition:{role}")
    bundle = bundle.model_copy(update={"proposals": clean})
    prompt_sha = hashlib.sha256((prompt.system + "\n" + prompt.user).encode()).hexdigest()
    return AnalystResult(
        role=role,
        bundle=bundle,
        prompt_sha=prompt_sha,
        cache_key=result.key,
        cached=result.cached,
        usage=result.usage.to_dict(),
        flags=flags,
    )
