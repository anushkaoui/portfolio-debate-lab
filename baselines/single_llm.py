"""Single LLM analyst, no debate, then the same hard risk layer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

from agents.analysts import run_analyst
from agents.llm import CachedLLM
from agents.schemas import SCHEMAS
from core.context import ContextPacket


@dataclass
class LLMBaselineResult:
    target_weights: dict[str, float]
    messages: list[dict]
    flags: list[str]
    calls: int
    provider_calls: int
    cost_usd: float


def _schema_json(name: str) -> str:
    return json.dumps(SCHEMAS[name].model_json_schema(), sort_keys=True)


def single_llm_weights(
    client: CachedLLM,
    *,
    packet: ContextPacket,
    params: Mapping[str, Any],
    as_of: date,
    max_delta: float,
) -> LLMBaselineResult:
    start_calls = client.n_calls
    start_provider = client.n_provider_calls
    start_cost = client.total_cost
    current = {
        str(k): float(v)
        for k, v in (packet.context.get("portfolio", {}).get("weights") or {}).items()
    }
    codes = list(packet.context["universe"])
    weights = {c: max(0.0, current.get(c, 0.0)) for c in codes}
    messages: list[dict] = []
    flags: list[str] = ["no_debate", "single_llm"]
    try:
        res = run_analyst(
            client,
            style="trend",
            packet=packet,
            params=params,
            max_analyst_delta=max_delta,
            schema_json=_schema_json("proposal_bundle"),
            as_of=as_of,
        )
        for prop in res.bundle.proposals:
            weights[prop.asset] = max(0.0, weights.get(prop.asset, 0.0) + prop.target_weight_delta)
        flags.extend(res.flags)
        messages.append(
            {
                "role": res.role,
                "prompt_sha": res.prompt_sha,
                "cache_key": res.cache_key,
                "cached": res.cached,
                "usage": res.usage,
                "response": res.bundle.model_dump(),
            }
        )
    except Exception as exc:
        flags.append(f"single_llm_failure:{type(exc).__name__}")
        messages.append({"role": "analyst_trend", "error": str(exc)[:800]})
    total = sum(weights.values())
    if total > 1.0:
        weights = {k: v / total for k, v in weights.items()}
        flags.append("weights_scaled_to_gross_one")
    real = {packet.code_to_ticker.get(c, c): w for c, w in weights.items()}
    return LLMBaselineResult(
        target_weights=real,
        messages=messages,
        flags=sorted(set(flags)),
        calls=client.n_calls - start_calls,
        provider_calls=client.n_provider_calls - start_provider,
        cost_usd=client.total_cost - start_cost,
    )


def ensemble_weights(
    client: CachedLLM,
    *,
    packet: ContextPacket,
    params: Mapping[str, Any],
    as_of: date,
    max_delta: float,
    n_samples: int = 3,
) -> LLMBaselineResult:
    start_calls = client.n_calls
    start_provider = client.n_provider_calls
    start_cost = client.total_cost
    current = {
        str(k): float(v)
        for k, v in (packet.context.get("portfolio", {}).get("weights") or {}).items()
    }
    codes = list(packet.context["universe"])
    aggregate = {c: 0.0 for c in codes}
    messages: list[dict] = []
    flags: list[str] = ["no_discussion", "ensemble_vote"]
    successes = 0
    for i in range(max(1, int(n_samples))):
        sample_params = dict(params)
        sample_params["ensemble_sample"] = i
        try:
            res = run_analyst(
                client,
                style="trend" if i % 2 == 0 else "contrarian",
                packet=packet,
                params=sample_params,
                max_analyst_delta=max_delta,
                schema_json=_schema_json("proposal_bundle"),
                as_of=as_of,
            )
            successes += 1
            for prop in res.bundle.proposals:
                aggregate[prop.asset] += prop.target_weight_delta
            flags.extend(res.flags)
            messages.append(
                {
                    "role": res.role,
                    "sample": i,
                    "prompt_sha": res.prompt_sha,
                    "cache_key": res.cache_key,
                    "cached": res.cached,
                    "usage": res.usage,
                    "response": res.bundle.model_dump(),
                }
            )
        except Exception as exc:
            flags.append(f"ensemble_sample_failure:{i}:{type(exc).__name__}")
            messages.append({"role": "ensemble_sample", "sample": i, "error": str(exc)[:500]})
    if successes:
        for c in codes:
            aggregate[c] /= successes
    weights = {c: max(0.0, current.get(c, 0.0) + aggregate[c]) for c in codes}
    total = sum(weights.values())
    if total > 1.0:
        weights = {k: v / total for k, v in weights.items()}
        flags.append("weights_scaled_to_gross_one")
    real = {packet.code_to_ticker.get(c, c): w for c, w in weights.items()}
    return LLMBaselineResult(
        target_weights=real,
        messages=messages,
        flags=sorted(set(flags)),
        calls=client.n_calls - start_calls,
        provider_calls=client.n_provider_calls - start_provider,
        cost_usd=client.total_cost - start_cost,
    )
