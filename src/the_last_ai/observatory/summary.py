"""Generate cautious computational life summaries from recorded data only."""

from __future__ import annotations

from the_last_ai.observatory.affect import affect_label, relationship_label
from the_last_ai.observatory.event import LifeEventType
from the_last_ai.observatory.history import AgentLifeHistory
from the_last_ai.types import JSONDict


def generate_life_summary(
    history: AgentLifeHistory,
    *,
    active_ids: set[str] | None = None,
    memories: JSONDict | None = None,
) -> str:
    """
    Data-driven biography paragraph(s). No scripted emotional claims.
    """
    active_ids = active_ids or set()
    aid = history.agent_id
    lines: list[str] = [f"{aid.upper()} — COMPUTATIONAL LIFE SUMMARY", ""]

    # Strongest relationships from trajectories
    ranked = sorted(
        history.relationship_trajectories.values(),
        key=lambda t: (
            -(t.strength_history[-1][1] if t.strength_history else 0.0),
            -(t.interaction_count),
        ),
    )
    if ranked:
        top = ranked[0]
        strength = top.strength_history[-1][1] if top.strength_history else 0.0
        trust = top.trust_history[-1][1] if top.trust_history else 0.5
        label = relationship_label(
            interactions=top.interaction_count, strength=strength, trust=trust
        )
        status = "historical" if (top.historical or top.other_id not in active_ids) else "active"
        lines.append(
            f"{aid} recorded {top.interaction_count} interactions with {top.other_id}, "
            f"one of its strongest measured relationships "
            f"(label: {label}; strength {strength:.2f}; trust {trust:.2f}; status: {status})."
        )
        lines.append("")

    # Communication
    sent = [m for m in history.messages if m.get("role_for_agent") == "sent"]
    recv = [m for m in history.messages if m.get("role_for_agent") == "received"]
    if history.messages:
        token_counts: dict[str, int] = {}
        for m in history.messages:
            for t in m.get("tokens") or []:
                token_counts[t] = token_counts.get(t, 0) + 1
        top_tokens = sorted(token_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        lines.append(
            f"Communication memory contains {len(history.messages)} message records "
            f"({len(sent)} sent / {len(recv)} received). "
            f"Frequent tokens: "
            + ", ".join(f"{t}×{c}" for t, c in top_tokens)
            + "."
        )
        lines.append("")

    # Disappearances / collapse
    absences = [
        e for e in history.events if e.event_type == LifeEventType.DISAPPEARANCE_OTHER
    ]
    collapses = [e for e in history.events if e.event_type == LifeEventType.COLLAPSE_STAGE]
    if absences:
        others = sorted({e.context.get("other_id") for e in absences if e.context.get("other_id")})
        lines.append(
            f"After counterpart disappearance events involving "
            f"{', '.join(str(o) for o in others[:8])}"
            f"{'…' if len(others) > 8 else ''}, "
            f"{aid} retained historical relationship and/or communication records "
            f"rather than deleting them."
        )
        lines.append("")
    if collapses:
        lines.append(
            f"Population collapse stages recorded: "
            + "; ".join(
                f"tick {e.tick}→pop {e.context.get('population')}" for e in collapses[-6:]
            )
            + "."
        )
        lines.append("")

    # Affect trajectory
    if history.affect_series:
        first = history.affect_series[0][1]
        last = history.affect_series[-1][1]
        last_label, reasons = affect_label(last)
        lines.append(
            f"Affective dimensions shifted from valence {first.valence:.2f} "
            f"to {last.valence:.2f}; final descriptive label: {last_label} "
            f"(reasons: {', '.join(reasons)})."
        )
        lines.append("")

    # Memories
    if memories:
        count = memories.get("entity_count") or len(memories.get("familiarities") or {})
        lines.append(
            f"Entity memory retained representations of {count} agents "
            f"at the inspected snapshot."
        )
        lines.append("")

    lines.append(
        "These observations describe computational state changes and behavioural "
        "associations. They do not establish subjective emotion or consciousness."
    )
    return "\n".join(lines)
