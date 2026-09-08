"""Spectator experience helpers — causal chains, deltas, patterns, cinema moments.

All values derive from real simulation / loss / observatory state.
No fabricated emotions.
"""

from __future__ import annotations

from the_last_ai.loss.model import EntityStatus
from the_last_ai.observatory.event import LifeEventType
from the_last_ai.types import JSONDict


DISCLAIMER = (
    "These labels describe computational state and behaviour patterns. "
    "They do not demonstrate subjective feelings, consciousness, or sentience."
)


def metric_deltas(current: JSONDict, baseline: JSONDict | None) -> JSONDict:
    """Compare live metrics to a captured baseline (e.g. pre-disappearance)."""
    if not baseline or not current:
        return {}
    keys = (
        "social_loss",
        "prediction_error",
        "uncertainty",
        "security",
        "social_drive",
        "valence",
        "seek_attempts",
        "search_pressure",
        "exploration",
    )
    out: JSONDict = {}
    for key in keys:
        if key not in current or key not in baseline:
            continue
        try:
            cur = float(current[key])
            base = float(baseline[key])
        except (TypeError, ValueError):
            continue
        delta = cur - base
        direction = "up" if delta > 0.03 else ("down" if delta < -0.03 else "flat")
        # Counters: absolute growth
        if key in {"seek_attempts"}:
            direction = "up" if delta > 0 else "flat"
        out[key] = {
            "baseline": base,
            "current": cur,
            "delta": delta,
            "direction": direction,
        }
    return out


def build_spotlight(agent, *, active_ids: set[str]) -> list[JSONDict]:
    """Last-known / expected loci for historical entities (search focus)."""
    spots: list[JSONDict] = []
    if not hasattr(agent, "loss_tracker"):
        return spots
    for eid, rec in agent.loss_tracker.records.items():
        if rec.status != EntityStatus.HISTORICAL:
            continue
        if eid in active_ids:
            continue
        loc = None
        mem = agent.ltm.get(eid) if hasattr(agent, "ltm") else None
        if mem and mem.expected_location:
            loc = list(mem.expected_location)
        elif (rec.memory_before or {}).get("expected_location"):
            loc = list(rec.memory_before["expected_location"])
        elif (rec.prediction_before or {}).get("last_position"):
            loc = list(rec.prediction_before["last_position"])
        if not loc:
            continue
        spots.append(
            {
                "entity_id": eid,
                "x": int(loc[0]),
                "y": int(loc[1]),
                "significance": float(rec.significance),
                "social_loss": float(rec.social_loss),
                "search_pressure": float(rec.search_pressure),
                "label": "last known / expected location",
            }
        )
    spots.sort(key=lambda s: -s["significance"])
    return spots[:3]


def build_causal_chain(agent, *, tick: int) -> JSONDict | None:
    """Human-readable computational chain for the strongest historical loss."""
    if not hasattr(agent, "loss_tracker"):
        return None
    hist = [
        r
        for r in agent.loss_tracker.records.values()
        if r.status == EntityStatus.HISTORICAL and r.disappearance_tick is not None
    ]
    if not hist:
        return None
    rec = max(hist, key=lambda r: (r.significance, r.peak_social_loss))
    steps = [
        {
            "id": "relationship",
            "label": "Learned relationship",
            "detail": (
                f"significance {rec.significance:.2f} · "
                f"interactions {(rec.relationship_before or {}).get('interaction_count', 0)}"
            ),
            "active": True,
        },
        {
            "id": "expectation",
            "label": "Internal expectation retained",
            "detail": f"presence expectation {rec.presence_expectation_at_loss:.2f}",
            "active": True,
        },
        {
            "id": "disappearance",
            "label": "Entity no longer in world",
            "detail": f"disappeared tick {rec.disappearance_tick}",
            "active": True,
        },
        {
            "id": "prediction_mismatch",
            "label": "Prediction mismatch",
            "detail": (
                f"disruption {rec.prediction_disruption:.2f} "
                f"(peak {rec.peak_prediction_disruption:.2f})"
            ),
            "active": rec.prediction_disruption >= 0.15
            or rec.peak_prediction_disruption >= 0.2,
        },
        {
            "id": "memory",
            "label": "Memory still retrieved / modelled",
            "detail": (
                f"retrievals after {rec.memory_retrievals_after} · "
                f"memory now {rec.memory_strength_current:.2f}"
            ),
            "active": rec.memory_retrievals_after > 0 or rec.memory_strength_current > 0.05,
        },
        {
            "id": "search",
            "label": "Search / pursuit of expected locus",
            "detail": (
                f"search attempts {rec.search_attempts} · "
                f"pressure {rec.search_pressure:.2f}"
            ),
            "active": rec.search_attempts > 0 or rec.search_pressure >= 0.1,
        },
        {
            "id": "social_loss",
            "label": "Elevated social-loss aggregate",
            "detail": (
                f"social_loss {rec.social_loss:.2f} "
                f"(peak {rec.peak_social_loss:.2f})"
            ),
            "active": rec.social_loss >= 0.2 or rec.peak_social_loss >= 0.25,
        },
        {
            "id": "adaptation",
            "label": "Computational adaptation",
            "detail": (
                f"tick {rec.adaptation_tick}"
                if rec.adapted
                else "criterion not yet met"
            ),
            "active": bool(rec.adapted),
        },
    ]
    return {
        "entity_id": rec.entity_id,
        "tick": tick,
        "disappearance_tick": rec.disappearance_tick,
        "steps": steps,
        "disclaimer": DISCLAIMER,
        "summary": (
            f"After {rec.entity_id} disappeared, expectation and observation diverged; "
            f"memory and search pressure are measured consequences — not feelings."
        ),
    }


def detect_patterns(metrics: JSONDict, deltas: JSONDict, chain: JSONDict | None) -> list[JSONDict]:
    """Named computational patterns when evidence thresholds are met."""
    patterns: list[JSONDict] = []
    pe = float(metrics.get("prediction_error") or 0.0)
    sl = float(metrics.get("social_loss") or 0.0)
    sp = float(metrics.get("search_pressure") or 0.0)
    pe_d = (deltas.get("prediction_error") or {}).get("delta", 0.0)
    sl_d = (deltas.get("social_loss") or {}).get("delta", 0.0)
    seek_d = (deltas.get("seek_attempts") or {}).get("delta", 0.0)

    if pe >= 0.35 or pe_d >= 0.12:
        patterns.append(
            {
                "id": "prediction_mismatch",
                "label": "Prediction-mismatch",
                "evidence": f"PE={pe:.2f}, Δ={pe_d:+.2f}",
            }
        )
    if sl >= 0.35 or sl_d >= 0.15:
        patterns.append(
            {
                "id": "social_loss_state",
                "label": "Social-loss state",
                "evidence": f"social_loss={sl:.2f}, Δ={sl_d:+.2f}",
            }
        )
    if sp >= 0.2 or seek_d >= 1:
        patterns.append(
            {
                "id": "residual_search",
                "label": "Residual search",
                "evidence": f"search_pressure={sp:.2f}, seek Δ={seek_d:+.0f}",
            }
        )
    if chain and any(s["id"] == "adaptation" and s["active"] for s in chain.get("steps", [])):
        patterns.append(
            {
                "id": "adaptation",
                "label": "Adaptation toward baseline",
                "evidence": "adaptation criterion met for a historical entity",
            }
        )
    social_d = (deltas.get("social_drive") or {}).get("delta", 0.0)
    if sl >= 0.3 and social_d < -0.05:
        patterns.append(
            {
                "id": "withdrawal_like",
                "label": "Withdrawal-like social-drive drop",
                "evidence": f"social_drive Δ={social_d:+.2f} with elevated social_loss",
            }
        )
    return patterns


def build_life_timeline(sim, spectate_id: str, *, limit: int = 24) -> list[JSONDict]:
    """Rolling life moments from observatory events for the spectated agent."""
    hist = sim.observatory.histories.get(spectate_id)
    if hist is None:
        return []
    interesting = {
        LifeEventType.ENTITY_ENCOUNTERED,
        LifeEventType.SOCIAL_INTERACTION,
        LifeEventType.COMMUNICATION_SENT,
        LifeEventType.COMMUNICATION_RECEIVED,
        LifeEventType.COMMUNICATION_VERIFIED,
        LifeEventType.DISAPPEARANCE_OTHER,
        LifeEventType.ENTITY_ABSENT_EXPECTED,
        LifeEventType.MEMORY_ACTIVATED,
        LifeEventType.PREDICTION_ERROR,
        LifeEventType.AFFECTIVE_CHANGE,
        LifeEventType.INVESTIGATION,
        LifeEventType.COLLAPSE_STAGE,
        LifeEventType.RELATIONSHIP_UPDATED,
    }
    rows: list[JSONDict] = []
    for ev in hist.events:
        if ev.event_type not in interesting:
            continue
        rows.append(
            {
                "tick": ev.tick,
                "kind": ev.event_type.value,
                "summary": ev.summary or ev.event_type.value,
            }
        )
    return rows[-limit:]


def significant_partner_id(agent, *, active_ids: set[str]) -> str | None:
    """Strongest historical bond, else strongest active relationship."""
    best_id = None
    best_score = -1.0
    if hasattr(agent, "loss_tracker"):
        for eid, rec in agent.loss_tracker.records.items():
            if rec.status == EntityStatus.HISTORICAL and rec.significance > best_score:
                best_score = rec.significance
                best_id = eid
    if best_id:
        return best_id
    if hasattr(agent, "relationships"):
        for oid, rel in agent.relationships.relationships.items():
            score = float(rel.strength) + 0.3 * float(rel.trust)
            if score > best_score:
                best_score = score
                best_id = oid
    return best_id


def behaviour_rates(agent, baseline: JSONDict | None) -> JSONDict:
    """Before→after behaviour counters for the spectated agent (computational only)."""
    seeks = int(getattr(agent, "seek_attempts", 0))
    investigate = int(getattr(agent, "investigation_actions", 0))
    communicate = 0
    if hasattr(agent, "social_action_counts"):
        communicate = int(agent.social_action_counts.get("communicate", 0) or 0)
    messages = int(getattr(agent, "messages_sent_count", 0))
    out: JSONDict = {
        "seek_attempts": seeks,
        "investigation_actions": investigate,
        "communicate_actions": communicate,
        "messages_sent": messages,
    }
    if baseline:
        out["delta_seek"] = seeks - int(baseline.get("seek_attempts") or 0)
        out["delta_investigate"] = investigate - int(
            baseline.get("investigation_actions") or 0
        )
        out["delta_messages"] = messages - int(baseline.get("messages_sent") or 0)
    return out


def phase_countdown(
    *,
    phase: str,
    phase_ticks: int,
    ticks_between: int,
    final_ticks: int,
    schedule: list[int],
    schedule_idx: int,
    population: int,
    demo_mode: str | None,
    demo_bond_ticks: int,
    demo_post_ticks: int,
    tick_index: int,
    demo_stranger_ticks: int = 50,
) -> JSONDict:
    """Human-facing countdown for next meaningful phase change."""
    if demo_mode in {"loss", "contrast"}:
        if phase == "bond":
            left = max(0, demo_bond_ticks - phase_ticks)
            return {
                "label": "Bonding",
                "next": "Partner disappearance",
                "ticks_remaining": left,
                "detail": f"{left} ticks until significant partner is removed",
            }
        if phase == "post_loss":
            left = max(0, demo_post_ticks - phase_ticks)
            return {
                "label": "Post-disappearance",
                "next": "Session end",
                "ticks_remaining": left,
                "detail": "Watch prediction mismatch, search, and social-loss",
            }
        if phase == "post_friend":
            left = max(0, demo_post_ticks - phase_ticks)
            return {
                "label": "After friend removal",
                "next": "Stranger removal",
                "ticks_remaining": left,
                "detail": "Computational response to bonded partner absence",
            }
        if phase == "post_stranger":
            left = max(0, demo_stranger_ticks - phase_ticks)
            return {
                "label": "After stranger removal",
                "next": "Contrast summary",
                "ticks_remaining": left,
                "detail": "Compare to friend-removal response (control contrast)",
            }
    if phase == "formation" and schedule:
        left = max(0, ticks_between - phase_ticks)
        return {
            "label": "Formation",
            "next": f"Collapse toward {schedule[0]}",
            "ticks_remaining": left,
            "detail": f"{left} ticks until population reduction begins",
        }
    if phase == "collapse" and schedule and schedule_idx < len(schedule):
        target = schedule[schedule_idx]
        if population > target:
            return {
                "label": "Collapsing",
                "next": f"Population {target}",
                "ticks_remaining": population - target,
                "detail": f"{population - target} removals until stage {target}",
            }
        left = max(0, ticks_between - phase_ticks)
        return {
            "label": f"At {population}",
            "next": "Next collapse stage",
            "ticks_remaining": left,
            "detail": f"Dwell {left} ticks before next stage",
        }
    if phase == "final":
        left = max(0, final_ticks - phase_ticks)
        return {
            "label": "Final survivor",
            "next": "End",
            "ticks_remaining": left,
            "detail": f"{left} ticks of solo observation remaining",
        }
    return {
        "label": phase,
        "next": None,
        "ticks_remaining": None,
        "detail": "Free run — no scheduled disappearances",
    }
