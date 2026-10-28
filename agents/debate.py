"""Bounded debate state machine: analysts -> risk manager -> response -> trader.

Termination is guaranteed by a fixed sequence and a hard cap on calls per
decision. Every failed call is logged and falls back to holding the current
portfolio rather than crashing the backtest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping

from agents.analysts import AnalystResult, run_analyst
from agents.llm import CachedLLM
from agents.risk_manager import RiskManagerResult, run_risk_manager
from agents.schemas import (
    AnalystResponse,
    Critique,
    Decision,
    Proposal,
    ProposalBundle,
    ResponseBundle,
    SCHEMAS,
)
from agents.trader import TraderResult, run_trader
from core.context import ContextPacket


def _schema_json(name: str) -> str:
    return json.dumps(SCHEMAS[name].model_json_schema(), sort_keys=True)


@dataclass
class DebateResult:
    as_of: date
    context_hash: str
    proposals: list[Proposal]
    critiques: list[Critique]
    responses: list[AnalystResponse]
    decision: Decision
    target_weights_real: dict[str, float]
    messages: list[dict]
    flags: list[str]
    calls: int
    provider_calls: int
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    analyst_results: list[AnalystResult] = field(default_factory=list)
    risk_manager_result: RiskManagerResult | None = None
    trader_result: TraderResult | None = None

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "context_hash": self.context_hash,
            "proposals": [p.model_dump() for p in self.proposals],
            "critiques": [c.model_dump() for c in self.critiques],
            "responses": [r.model_dump() for r in self.responses],
            "decision": self.decision.model_dump(),
            "target_weights_real": {k: round(v, 8) for k, v in self.target_weights_real.items()},
            "flags": self.flags,
            "calls": self.calls,
            "provider_calls": self.provider_calls,
            "cost_usd": round(self.cost_usd, 8),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "messages": self.messages,
        }


class DebateEngine:
    def __init__(
        self,
        client: CachedLLM,
        params: Mapping[str, Any],
        *,
        analyst_styles: tuple[str, ...] = ("trend", "contrarian"),
        enable_risk_manager: bool = True,
        enable_rebuttal: bool = True,
        max_calls_per_decision: int = 8,
    ) -> None:
        self.client = client
        self.params = dict(params)
        self.analyst_styles = tuple(analyst_styles)
        self.enable_risk_manager = bool(enable_risk_manager)
        self.enable_rebuttal = (
            bool(enable_rebuttal)
            and bool(self.params.get("allow_analyst_rebuttal", True))
            and int(self.params.get("max_debate_rounds", 1)) > 0
        )
        self.max_calls_per_decision = int(max_calls_per_decision)
        self._schemas = {
            "proposal": _schema_json("proposal_bundle"),
            "critique": _schema_json("critique_bundle"),
            "response": _schema_json("response_bundle"),
            "decision": _schema_json("decision"),
        }

    def _current_codes(self, packet: ContextPacket) -> dict[str, float]:
        return {
            str(k): float(v)
            for k, v in (packet.context.get("portfolio", {}).get("weights") or {}).items()
        }

    def decide(
        self,
        packet: ContextPacket,
        *,
        as_of: date,
    ) -> DebateResult:
        start_calls = self.client.n_calls
        start_provider_calls = self.client.n_provider_calls
        start_cost = self.client.total_cost
        start_pin = self.client.total_prompt_tokens
        start_pout = self.client.total_completion_tokens

        current_codes = self._current_codes(packet)
        codes = set(packet.context["universe"])
        messages: list[dict] = []
        flags: list[str] = []
        analyst_results: list[AnalystResult] = []
        proposals: list[Proposal] = []

        def calls_left() -> int:
            return self.max_calls_per_decision - (self.client.n_calls - start_calls)

        for style in self.analyst_styles:
            if calls_left() <= 0:
                flags.append("call_budget_per_decision_reached")
                break
            try:
                res = run_analyst(
                    self.client,
                    style=style,
                    packet=packet,
                    params=self.params,
                    max_analyst_delta=float(self.params.get("max_analyst_delta", 0.08)),
                    schema_json=self._schemas["proposal"],
                    as_of=as_of,
                )
                analyst_results.append(res)
                proposals.extend(res.bundle.proposals)
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
            except Exception as exc:  # fall back, keep the run alive
                flags.append(f"analyst_failure:{style}:{type(exc).__name__}")
                messages.append({"role": f"analyst_{style}", "error": str(exc)[:800]})

        proposal_dump = [p.model_dump() for p in proposals]
        critiques: list[Critique] = []
        risk_result: RiskManagerResult | None = None
        if self.enable_risk_manager and proposals and calls_left() > 0:
            try:
                risk_result = run_risk_manager(
                    self.client,
                    packet=packet,
                    proposals=proposals,
                    params=self.params,
                    schema_json=self._schemas["critique"],
                    as_of=as_of,
                )
                critiques = list(risk_result.bundle.critiques)
                flags.extend(risk_result.flags)
                messages.append(
                    {
                        "role": "risk_manager",
                        "prompt_sha": risk_result.prompt_sha,
                        "cache_key": risk_result.cache_key,
                        "cached": risk_result.cached,
                        "usage": risk_result.usage,
                        "response": risk_result.bundle.model_dump(),
                    }
                )
            except Exception as exc:
                flags.append(f"risk_manager_failure:{type(exc).__name__}")
                messages.append({"role": "risk_manager", "error": str(exc)[:800]})
        elif not self.enable_risk_manager:
            flags.append("risk_manager_disabled")

        responses: list[AnalystResponse] = []
        if self.enable_rebuttal and critiques and proposals and calls_left() > 0:
            try:
                from agents.prompts import build_prompt

                prompt = build_prompt(
                    "analyst_response",
                    schema_name="ResponseBundle",
                    context=packet.context,
                    payload={
                        "proposals": proposal_dump,
                        "critiques": [c.model_dump() for c in critiques],
                    },
                    params=self.params,
                    schema_json=self._schemas["response"],
                )
                result = self.client.call(
                    role="analyst_response",
                    system=prompt.system,
                    user=prompt.user,
                    schema=ResponseBundle,
                    params=self.params,
                    as_of=as_of,
                    context_hash=packet.context_hash,
                    payload=prompt.payload,
                )
                bundle: ResponseBundle = result.value  # type: ignore[assignment]
                valid_targets = {p.asset for p in proposals}
                responses = [r for r in bundle.responses if r.target in valid_targets]
                for response in bundle.responses:
                    if response.target not in valid_targets:
                        flags.append(f"response_unknown_target:{response.target}")
                    if response.recognizes_asset:
                        flags.append("self_reported_recognition:analyst_response")
                messages.append(
                    {
                        "role": "analyst_response",
                        "cache_key": result.key,
                        "cached": result.cached,
                        "usage": result.usage.to_dict(),
                        "response": bundle.model_dump(),
                    }
                )
            except Exception as exc:
                flags.append(f"response_failure:{type(exc).__name__}")
                messages.append({"role": "analyst_response", "error": str(exc)[:800]})

        decision: Decision | None = None
        trader_result: TraderResult | None = None
        if calls_left() > 0:
            try:
                trader_result = run_trader(
                    self.client,
                    packet=packet,
                    proposals=proposals,
                    critiques=critiques,
                    responses=responses,
                    params=self.params,
                    schema_json=self._schemas["decision"],
                    as_of=as_of,
                )
                decision = trader_result.decision
                flags.extend(trader_result.flags)
                messages.append(
                    {
                        "role": "trader",
                        "prompt_sha": trader_result.prompt_sha,
                        "cache_key": trader_result.cache_key,
                        "cached": trader_result.cached,
                        "usage": trader_result.usage,
                        "response": trader_result.decision.model_dump(),
                    }
                )
            except Exception as exc:
                flags.append(f"trader_failure:{type(exc).__name__}")
                messages.append({"role": "trader", "error": str(exc)[:800]})

        if decision is None:
            flags.append("fallback_to_current_weights")
            decision = Decision(
                target_weights={code: max(0.0, current_codes.get(code, 0.0)) for code in codes},
                rationale="LLM decision unavailable; holding current portfolio.",
                dissent_noted=bool(critiques),
                evidence=["portfolio.weights"],
                recognizes_asset=False,
            )

        decision_codes = {
            code: max(0.0, float(weight))
            for code, weight in decision.target_weights.items()
            if code in codes
        }
        for code in codes:
            decision_codes.setdefault(code, max(0.0, current_codes.get(code, 0.0)))
        total = sum(decision_codes.values())
        if total > 1.0:
            decision_codes = {k: v / total for k, v in decision_codes.items()}
            flags.append("decision_scaled_to_gross_one")

        # High-severity objection handling check (risk layer still has final say).
        for critique in critiques:
            if critique.severity != "high":
                continue
            accepted = any(
                r.target == critique.target and r.accept_critique for r in responses
            )
            final = decision_codes.get(critique.target, 0.0)
            current = current_codes.get(critique.target, 0.0)
            if final > current + 1e-9 and not accepted:
                rationale = decision.rationale.lower()
                if "override" not in rationale and "objection" not in rationale:
                    flags.append(f"unjustified_high_severity_override:{critique.target}")

        target_weights_real = {
            packet.code_to_ticker.get(code, code): weight
            for code, weight in decision_codes.items()
        }
        for result in analyst_results:
            if result.bundle.recognizes_asset:
                flags.append(f"recognition_flag:{result.role}")
        if risk_result and risk_result.bundle.recognizes_asset:
            flags.append("recognition_flag:risk_manager")
        if trader_result and trader_result.decision.recognizes_asset:
            flags.append("recognition_flag:trader")

        return DebateResult(
            as_of=as_of,
            context_hash=packet.context_hash,
            proposals=proposals,
            critiques=critiques,
            responses=responses,
            decision=decision,
            target_weights_real=target_weights_real,
            messages=messages,
            flags=sorted(set(flags)),
            calls=self.client.n_calls - start_calls,
            provider_calls=self.client.n_provider_calls - start_provider_calls,
            cost_usd=self.client.total_cost - start_cost,
            prompt_tokens=self.client.total_prompt_tokens - start_pin,
            completion_tokens=self.client.total_completion_tokens - start_pout,
            analyst_results=analyst_results,
            risk_manager_result=risk_result,
            trader_result=trader_result,
        )
