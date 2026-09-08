"""Live spectator — compact snapshots from the authoritative Python simulation."""

from __future__ import annotations

from the_last_ai.observatory.affect import affect_label, compute_affect
from the_last_ai.spectator.experience import (
    DISCLAIMER,
    behaviour_rates,
    build_causal_chain,
    build_life_timeline,
    build_spotlight,
    detect_patterns,
    metric_deltas,
    phase_countdown,
    significant_partner_id,
)
from the_last_ai.types import CellType, JSONDict


def _walls(sim) -> list[list[int]]:
    walls: list[list[int]] = []
    grid = sim.world.grid
    for y in range(grid.height):
        for x in range(grid.width):
            if int(grid.cells[y, x]) == int(CellType.WALL):
                walls.append([x, y])
    return walls


def _resources(sim) -> list[list[int]]:
    resources: list[list[int]] = []
    grid = sim.world.grid
    for y in range(grid.height):
        for x in range(grid.width):
            if int(grid.cells[y, x]) == int(CellType.RESOURCE):
                resources.append([x, y])
    return resources


def build_agent_metrics(agent, *, active_ids: set[str]) -> JSONDict:
    """Live metrics for one agent — derived only from real internals."""
    affect, factors = compute_affect(agent, active_ids=active_ids)
    label, reasons = affect_label(affect)
    mot = getattr(agent, "last_motivation", None)
    mem_count = 0
    if hasattr(agent, "ltm"):
        mem_count = len(agent.ltm.entities)
    rel_count = 0
    if hasattr(agent, "relationships"):
        rel_count = len(agent.relationships.relationships)
    comm_mem = 0
    if hasattr(agent, "communication_memory"):
        comm_mem = len(getattr(agent.communication_memory, "records", []) or [])
    tracked = 0
    if hasattr(agent, "world_model"):
        tracked = len(getattr(agent.world_model.agents, "entities", {}) or {})
    loss_agg = 0.0
    search_pressure = 0.0
    hist_count = 0
    if hasattr(agent, "loss_tracker"):
        loss_agg = float(agent.loss_tracker.aggregate_social_loss)
        hist = [
            r
            for r in agent.loss_tracker.records.values()
            if r.status.value == "HISTORICAL"
        ]
        hist_count = len(hist)
        if hist:
            search_pressure = max(r.search_pressure for r in hist)
    return {
        "agent_id": agent.agent_id,
        "energy": float(agent.state.energy),
        "energy_pct": max(0.0, min(100.0, float(agent.state.energy))),
        "reward_total": float(getattr(agent, "total_reward", 0.0)),
        "reward_per_tick": float(factors.get("reward_per_tick", 0.0)),
        "age": int(agent.state.age),
        "social_drive": float(affect.social_drive),
        "exploration": float(getattr(mot, "exploration_drive", 0.0) or 0.0),
        "uncertainty": float(affect.uncertainty),
        "security": float(affect.security),
        "novelty": float(affect.novelty),
        "prediction_error": float(affect.prediction_error),
        "social_loss": float(affect.social_loss),
        "goal_success": float(affect.goal_success),
        "valence": float(affect.valence),
        "activation": float(affect.activation),
        "affect_label": label,
        "affect_reasons": reasons,
        "memories": mem_count,
        "relationships": rel_count,
        "communication_memories": comm_mem,
        "tracked_entities": tracked,
        "historical_entities": hist_count,
        "seek_attempts": int(getattr(agent, "seek_attempts", 0)),
        "investigation_actions": int(getattr(agent, "investigation_actions", 0)),
        "messages_sent": int(getattr(agent, "messages_sent_count", 0)),
        "aggregate_social_loss": loss_agg,
        "search_pressure": search_pressure,
        "last_action": agent.last_action.value if agent.last_action else None,
        "disclaimer": DISCLAIMER,
    }


def build_world_static(sim) -> JSONDict:
    """One-time world geometry (walls). Resources change over time."""
    return {
        "width": sim.world.grid.width,
        "height": sim.world.grid.height,
        "walls": _walls(sim),
        "bordered": bool(sim.config.bordered),
        "seed": int(sim.config.seed),
        "agent_type": sim.config.agent_type,
    }


def build_live_snapshot(
    sim,
    *,
    spectate_id: str,
    speed: float,
    paused: bool,
    phase: str,
    elapsed_seconds: float,
    finished: bool = False,
    tick_index: int = 0,
    baseline_metrics: JSONDict | None = None,
    schedule: list[int] | None = None,
    schedule_idx: int = 0,
    phase_ticks: int = 0,
    ticks_between: int = 30,
    final_ticks: int = 40,
    demo_mode: str | None = None,
    demo_bond_ticks: int = 40,
    demo_post_ticks: int = 80,
    demo_stranger_ticks: int = 50,
    cinema_dim: bool = True,
    metric_series: list[JSONDict] | None = None,
    contrast: JSONDict | None = None,
) -> JSONDict:
    """Current-frame snapshot for the browser renderer (not full history)."""
    active = list(sim.active_agent_ids())
    active_set = set(active)
    if spectate_id not in active_set and active:
        spectate_id = sorted(active)[0]

    partner = None
    if spectate_id in sim.agents:
        partner = significant_partner_id(sim.agents[spectate_id], active_ids=active_set)

    agents = []
    for aid in active:
        pos = sim.world.agent_positions[aid]
        agents.append(
            {
                "id": aid,
                "x": pos.x,
                "y": pos.y,
                "spectated": aid == spectate_id,
                "significant_partner": aid == partner and partner in active_set,
                "dim": cinema_dim and aid != spectate_id and aid != partner,
            }
        )

    metrics = None
    deltas = {}
    spotlight: list[JSONDict] = []
    chain = None
    patterns: list[JSONDict] = []
    rates: JSONDict = {}
    if spectate_id in sim.agents:
        agent = sim.agents[spectate_id]
        metrics = build_agent_metrics(agent, active_ids=active_set)
        deltas = metric_deltas(metrics, baseline_metrics)
        spotlight = build_spotlight(agent, active_ids=active_set)
        chain = build_causal_chain(agent, tick=sim.world.tick)
        patterns = detect_patterns(metrics, deltas, chain)
        rates = behaviour_rates(agent, baseline_metrics)

    social_links = []
    for outcome in sim.interaction_log[-8:]:
        if outcome.tick != sim.world.tick and outcome.tick != sim.world.tick - 1:
            continue
        if "communicate" in (outcome.resolved_kind or ""):
            social_links.append(
                {
                    "a": outcome.agent_a,
                    "b": outcome.agent_b,
                    "kind": "communicate",
                    "tick": outcome.tick,
                }
            )
        elif outcome.resolved_kind:
            social_links.append(
                {
                    "a": outcome.agent_a,
                    "b": outcome.agent_b,
                    "kind": outcome.resolved_kind,
                    "tick": outcome.tick,
                }
            )

    countdown = phase_countdown(
        phase=phase,
        phase_ticks=phase_ticks,
        ticks_between=ticks_between,
        final_ticks=final_ticks,
        schedule=list(schedule or []),
        schedule_idx=schedule_idx,
        population=len(active),
        demo_mode=demo_mode,
        demo_bond_ticks=demo_bond_ticks,
        demo_post_ticks=demo_post_ticks,
        demo_stranger_ticks=demo_stranger_ticks,
        tick_index=tick_index,
    )

    return {
        "type": "snapshot",
        "tick": int(sim.world.tick),
        "tick_index": int(tick_index),
        "population": len(active),
        "spectating": spectate_id,
        "significant_partner": partner,
        "speed": float(speed),
        "paused": bool(paused),
        "phase": phase,
        "demo_mode": demo_mode,
        "elapsed_seconds": float(elapsed_seconds),
        "finished": bool(finished),
        "resources": _resources(sim),
        "agents": agents,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "deltas": deltas,
        "behaviour_rates": rates,
        "metric_series": list(metric_series or []),
        "contrast": contrast,
        "spotlight": spotlight,
        "causal_chain": chain,
        "patterns": patterns,
        "life_timeline": build_life_timeline(sim, spectate_id),
        "countdown": countdown,
        "social_links": social_links[-12:],
        "cinema_dim": cinema_dim,
        "title": "THE LAST AI",
        "disclaimer": DISCLAIMER,
    }


def classify_message_for_spectator(msg: JSONDict, spectate_id: str) -> str:
    """Return highlight class for a communication event."""
    sender = msg.get("sender") or msg.get("sender_id")
    receiver = msg.get("receiver") or msg.get("receiver_id")
    if sender == spectate_id and receiver == spectate_id:
        return "self"
    if sender == spectate_id:
        return "from_spectated"
    if receiver == spectate_id:
        return "to_spectated"
    return "other"


def format_communication_entry(msg: JSONDict, spectate_id: str) -> JSONDict:
    highlight = classify_message_for_spectator(msg, spectate_id)
    tokens = msg.get("tokens") or []
    absence_driven = bool(msg.get("absence_driven"))
    return {
        "tick": msg.get("tick"),
        "message_id": msg.get("message_id"),
        "event": msg.get("event", "communication"),
        "sender": msg.get("sender") or msg.get("sender_id"),
        "receiver": msg.get("receiver") or msg.get("receiver_id"),
        "tokens": list(tokens),
        "text": " ".join(str(t) for t in tokens),
        "highlight": highlight,
        "claimed_position": msg.get("claimed_position"),
        "sender_reliability": msg.get("sender_reliability"),
        "receiver_response": msg.get("receiver_response"),
        "verified": msg.get("verified"),
        "outcome": msg.get("outcome"),
        "concepts": list(msg.get("concepts") or []),
        "sender_position": msg.get("sender_position"),
        "receiver_position": msg.get("receiver_position"),
        "construction_reason": msg.get("construction_reason"),
        "absence_driven": absence_driven,
        "about_entity_id": msg.get("about_entity_id"),
        "tag": "absence-driven" if absence_driven else None,
    }
