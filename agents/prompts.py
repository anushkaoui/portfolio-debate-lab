"""Static prompt construction. Prompts are frozen and hashed."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from core.config import read_prompt


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user: str
    payload: dict[str, Any]


def _json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def build_prompt(
    role: str,
    *,
    schema_name: str,
    context: Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
    schema_json: str | None = None,
) -> PromptBundle:
    shared = read_prompt("shared_rules")
    params = dict(params or {})
    if role == "analyst_trend":
        role_text = read_prompt("analyst_trend").format(
            max_proposals=int(params.get("proposals_per_analyst", 3))
        )
        user_text = (
            "CONTEXT_JSON:\n"
            + _json(context or {})
            + "\nEND_CONTEXT_JSON\n\n"
            "Return a JSON object matching the ProposalBundle schema. "
            "Only use asset codes present in context.universe."
        )
    elif role == "analyst_contrarian":
        role_text = read_prompt("analyst_contrarian").format(
            max_proposals=int(params.get("proposals_per_analyst", 3))
        )
        user_text = (
            "CONTEXT_JSON:\n"
            + _json(context or {})
            + "\nEND_CONTEXT_JSON\n\n"
            "Return a JSON object matching the ProposalBundle schema. "
            "Only use asset codes present in context.universe."
        )
    elif role == "risk_manager":
        role_text = read_prompt("risk_manager")
        user_text = (
            "CONTEXT_JSON:\n"
            + _json(context or {})
            + "\nEND_CONTEXT_JSON\n\nPROPOSALS_JSON:\n"
            + _json((payload or {}).get("proposals") or [])
            + "\nEND_PROPOSALS_JSON\n\n"
            "Return a JSON object matching the CritiqueBundle schema. "
            "For each proposal, use its asset code as `target`."
        )
    elif role == "analyst_response":
        role_text = read_prompt("analyst_response")
        user_text = (
            "CONTEXT_JSON:\n"
            + _json(context or {})
            + "\nEND_CONTEXT_JSON\n\nPROPOSALS_JSON:\n"
            + _json((payload or {}).get("proposals") or [])
            + "\nEND_PROPOSALS_JSON\n\nCRITIQUES_JSON:\n"
            + _json((payload or {}).get("critiques") or [])
            + "\nEND_CRITIQUES_JSON\n\n"
            "Return a JSON object matching the ResponseBundle schema."
        )
    elif role == "trader":
        role_text = read_prompt("trader")
        user_text = (
            "CONTEXT_JSON:\n"
            + _json(context or {})
            + "\nEND_CONTEXT_JSON\n\nPROPOSALS_JSON:\n"
            + _json((payload or {}).get("proposals") or [])
            + "\nEND_PROPOSALS_JSON\n\nCRITIQUES_JSON:\n"
            + _json((payload or {}).get("critiques") or [])
            + "\nEND_CRITIQUES_JSON\n\nRESPONSES_JSON:\n"
            + _json((payload or {}).get("responses") or [])
            + "\nEND_RESPONSES_JSON\n\n"
            "HIGH_SEVERITY_PRESENT: "
            + ("true" if any(
                c.get("severity") == "high"
                for c in ((payload or {}).get("critiques") or [])
            ) else "false")
            + "\n\nReturn a JSON object matching the Decision schema. "
            "Include every asset code in context.universe exactly once."
        )
    elif role in {"identification_guess", "probe_prediction"}:
        role_text = (
            "ROLE: MEMORIZATION PROBE\n"
            "Use only the supplied context. Try to identify the asset or period, "
            "or predict the next move, and state your confidence. "
            "Do not use outside knowledge; mark unknown where appropriate."
        )
        user_text = (
            "CONTEXT_JSON:\n"
            + _json(context or {})
            + "\nEND_CONTEXT_JSON\n\n"
            "Return the requested probe JSON schema."
        )
    else:
        raise ValueError(f"Unknown role {role!r}")

    system = (
        shared
        + "\n"
        + role_text.strip()
        + "\n\nOUTPUT JSON SCHEMA:\n"
        + (schema_json or "")
        + "\n"
    )
    full_payload = {
        "context": dict(context or {}),
        "role": role,
        "schema_name": schema_name,
        **(dict(payload or {})),
    }
    return PromptBundle(system=system, user=user_text, payload=full_payload)
