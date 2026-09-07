"""Human-readable layers for computational loss (derived only from recorded data)."""

from __future__ import annotations

from the_last_ai.types import JSONDict


DISCLAIMER = (
    "These labels describe patterns in the agent's computational state and behaviour. "
    "They do not demonstrate subjective feelings, consciousness, or sentience."
)


def _rel_strength_phrase(strength: float, interactions: int, trust: float) -> str:
    # Local copy of observatory thresholds to avoid import cycles.
    if interactions <= 0:
        return "stranger"
    if interactions >= 15 and trust >= 0.7 and strength >= 0.75:
        return "strongly connected"
    if trust >= 0.6 and strength >= 0.55:
        return "trusted"
    if interactions >= 5 and strength >= 0.35:
        return "familiar"
    if interactions >= 1:
        return "acquaintance"
    return "stranger"


def _behaviour_deltas(windows: list[JSONDict] | None) -> list[str]:
    if not windows or len(windows) < 2:
        return []
    base = (windows[0].get("rates") or {}) if windows[0].get("label") == "baseline" else {}
    # Prefer named baseline
    for w in windows:
        if w.get("label") == "baseline":
            base = w.get("rates") or {}
            break
    notes: list[str] = []
    post = None
    for w in windows:
        if w.get("label") in {"immediate", "short_term", "post"}:
            post = w
            break
    if not post:
        post = windows[-1] if windows else None
    if not post or not base:
        return notes
    rates = post.get("rates") or {}
    mapping = [
        ("exploration_cells_per_tick", "Exploration"),
        ("social_actions_per_tick", "Social interaction"),
        ("seek_per_tick", "Search / memory-guided movement"),
        ("investigation_per_tick", "Investigation"),
        ("communicate_per_tick", "Communication"),
    ]
    for key, label in mapping:
        b = float(base.get(key, 0.0) or 0.0)
        a = float(rates.get(key, 0.0) or 0.0)
        if abs(a - b) < 0.01 and b == 0:
            continue
        if a > b * 1.25 + 0.005:
            notes.append(f"{label} ↑")
        elif a < b * 0.75 - 0.005:
            notes.append(f"{label} ↓")
    return notes


def narrate_entity_loss(rec: JSONDict, *, agent_id: str = "agent") -> JSONDict:
    """Build Level 1 / Level 2 / Level 3 views from one EntityLossRecord dict."""
    eid = rec.get("entity_id", "entity")
    tick = rec.get("disappearance_tick")
    before = rec.get("relationship_before") or {}
    mem_before = rec.get("memory_before") or {}
    strength = float(before.get("strength", 0.0) or 0.0)
    trust = float(before.get("trust", 0.5) or 0.5)
    interactions = int(before.get("interaction_count", 0) or 0)
    label = _rel_strength_phrase(strength, interactions, trust)
    pe_before = float((rec.get("prediction_before") or {}).get("confidence", 0.0) or 0.0)
    pe_peak = float(rec.get("peak_prediction_disruption", 0.0) or 0.0)
    loss0 = 0.0
    traj = rec.get("loss_trajectory") or []
    if traj:
        loss0 = float(traj[0].get("social_loss", 0.0) or 0.0)
    loss_peak = float(rec.get("peak_social_loss", 0.0) or 0.0)
    loss_now = float(rec.get("social_loss", 0.0) or 0.0)
    mem_now = float(rec.get("memory_strength_current", 0.0) or 0.0)
    mem0 = float(mem_before.get("strength", 0.0) or 0.0)

    level1 = (
        f"{eid} disappeared at tick {tick} after a period of interaction with {agent_id} "
        f"({interactions} recorded interactions; relationship labelled {label}). "
        f"{agent_id} retained memory and relationship representations of {eid}. "
        f"After disappearance, prediction disruption rose "
        f"(peak {pe_peak:.2f}) and social_loss rose "
        f"({loss0:.2f} → peak {loss_peak:.2f}). "
        f"Search attempts: {rec.get('search_attempts', 0)}; "
        f"memory retrievals after loss: {rec.get('memory_retrievals_after', 0)}; "
        f"communication attempts toward absent entity: {rec.get('communication_attempts_after', 0)}. "
        f"Current memory strength: {mem_now:.2f} (was {mem0:.2f}). "
        f"Status: {rec.get('status')}."
    )

    beh = _behaviour_deltas(rec.get("behaviour_windows"))
    beh_note = (" — observed: " + ", ".join(beh)) if beh else ""
    level2 = (
        f"{eid} had a learned relationship with {agent_id} "
        f"(strength {strength:.2f}, trust {trust:.2f}, significance {float(rec.get('significance', 0)):.2f}). "
        f"Disappearance removed an entity that internal models could still expect "
        f"(presence expectation at loss {float(rec.get('presence_expectation_at_loss', 0)):.2f}). "
        f"The mismatch between expectation and non-observation is associated with elevated "
        f"prediction disruption and, when memory was retrieved, search toward last-known locations. "
        f"Behavioural change is measured, not assumed{beh_note}. "
        f"Adaptation (computational criterion): "
        f"{'tick ' + str(rec['adaptation_tick']) if rec.get('adapted') else 'not yet met'}."
    )

    analogy = None
    if loss_peak >= 0.35 and pe_peak >= 0.2:
        analogy = (
            "Behaviourally, this resembles aspects of a sadness-like / social-loss response "
            "in the computational sense (elevated social_loss, prediction mismatch, residual search). "
            "This is an analogy to measurable patterns — not evidence of felt grief."
        )

    return {
        "entity_id": eid,
        "level1_human_summary": level1,
        "level2_why": level2,
        "level3_technical": rec,
        "behaviour_notes": _behaviour_deltas(rec.get("behaviour_windows")),
        "analogy_note": analogy,
        "disclaimer": DISCLAIMER,
        "display": {
            "entity": eid,
            "relationship_before": label,
            "interaction_count": interactions,
            "trust": round(trust, 3),
            "memory_strength_before": round(mem0, 3),
            "disappeared_tick": tick,
            "prediction_disruption": f"{pe_before:.2f} → peak {pe_peak:.2f}",
            "social_loss": f"{loss0:.2f} → peak {loss_peak:.2f} (now {loss_now:.2f})",
            "search_attempts": rec.get("search_attempts", 0),
            "communication_attempts": rec.get("communication_attempts_after", 0),
            "memory_retrievals": rec.get("memory_retrievals_after", 0),
            "adaptation": rec.get("adaptation_tick"),
            "current_memory": round(mem_now, 3),
            "status": rec.get("status"),
        },
    }


def build_loss_observatory_section(
    loss_summary: JSONDict | None,
    *,
    agent_id: str,
    affect_series: list | None = None,
) -> JSONDict:
    """Assemble observatory Loss tab payload from tracker summary + optional affect."""
    loss_summary = loss_summary or {}
    records = loss_summary.get("records") or {}
    narratives = [
        narrate_entity_loss(rec, agent_id=agent_id)
        for rec in records.values()
        if rec.get("disappearance_tick") is not None
        or rec.get("status") in {"HISTORICAL", "FORGOTTEN"}
    ]
    narratives.sort(key=lambda n: (n["display"].get("disappeared_tick") or 0))

    # Simple series for charts from first/primary historical entity or aggregate
    charts: JSONDict = {
        "social_loss_over_time": [],
        "prediction_error_over_time": [],
        "memory_strength_over_time": [],
    }
    for narr in narratives:
        tech = narr["level3_technical"]
        eid = tech.get("entity_id")
        for pt in tech.get("loss_trajectory") or []:
            charts["social_loss_over_time"].append(
                {"tick": pt.get("tick"), "value": pt.get("social_loss"), "entity_id": eid}
            )
        for pt in tech.get("prediction_error_trajectory") or []:
            charts["prediction_error_over_time"].append(
                {
                    "tick": pt.get("tick"),
                    "value": pt.get("prediction_disruption"),
                    "entity_id": eid,
                }
            )
        for pt in tech.get("memory_trajectory") or []:
            charts["memory_strength_over_time"].append(
                {"tick": pt.get("tick"), "value": pt.get("memory_strength"), "entity_id": eid}
            )

    if affect_series:
        charts["affect_social_loss"] = [
            {"tick": t, "value": (s.get("social_loss") if isinstance(s, dict) else s.social_loss)}
            for t, s in affect_series
        ]

    return {
        "disclaimer": DISCLAIMER,
        "aggregate_social_loss": loss_summary.get("aggregate_social_loss", 0.0),
        "counts": loss_summary.get("counts") or {},
        "by_status": loss_summary.get("by_status") or {},
        "entities": narratives,
        "dimension_changes": loss_summary.get("dimension_changes") or [],
        "charts": charts,
        "focus_prompt": (
            "Watch what the agent did after each disappearance: "
            "did prediction error rise, did memory stay active, did search continue?"
        ),
    }
