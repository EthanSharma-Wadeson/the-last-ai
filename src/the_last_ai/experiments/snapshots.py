"""Capture a research snapshot of an agent's internal state."""

from __future__ import annotations

from the_last_ai.types import JSONDict


def capture_agent_mind(agent, *, sim=None, label: str = "") -> JSONDict:
    """
    Record memories, relationships, world model, values, prediction errors, behaviour.

    Used at each disappearance event and at the end of The Last AI experiment.
    """
    snapshot: JSONDict = {
        "label": label,
        "agent_id": getattr(agent, "agent_id", None),
        "agent_type": agent.__class__.__name__,
        "energy": agent.state.energy,
        "age": agent.state.age,
        "total_reward": getattr(agent, "total_reward", 0.0),
        "resources_collected": getattr(agent, "resources_collected", 0),
        "last_action": agent.last_action.value if agent.last_action else None,
        "short_term_memory": agent.memory.to_dict() if hasattr(agent, "memory") else None,
    }

    if hasattr(agent, "memory_summary"):
        snapshot["entity_memory"] = agent.memory_summary()
    if hasattr(agent, "ltm"):
        tick = agent.last_observation.tick if agent.last_observation else agent.state.age
        snapshot["entity_representations"] = {
            eid: mem.to_dict() for eid, mem in agent.ltm.entities.items()
        }
        snapshot["entity_strengths"] = agent.ltm.strength_map(tick)

    if hasattr(agent, "relationship_summary"):
        snapshot["relationships"] = agent.relationship_summary()
    if hasattr(agent, "q_values"):
        snapshot["learned_values"] = {
            "q_states": len(agent.q_values),
            "epsilon": getattr(agent, "epsilon", None),
            "update_count": getattr(agent, "update_count", 0),
        }
    if hasattr(agent, "world_model_summary"):
        snapshot["world_model"] = agent.world_model_summary()
        snapshot["prediction_error_ema"] = agent.world_model.mean_error_ema
        snapshot["prediction_total_scored"] = agent.world_model.total_scored
    if hasattr(agent, "visit_counts"):
        snapshot["behaviour"] = {
            "unique_cells_visited": len(agent.visit_counts),
            "total_visits": int(sum(agent.visit_counts.values())),
            "seek_attempts": getattr(agent, "seek_attempts", 0),
            "approach_actions": getattr(agent, "approach_actions", 0),
            "avoid_actions": getattr(agent, "avoid_actions", 0),
            "social_action_counts": dict(getattr(agent, "social_action_counts", {})),
            "investigation_actions": getattr(agent, "investigation_actions", 0),
        }
    if hasattr(agent, "communication_summary"):
        active_ids = set(sim.active_agent_ids()) if sim is not None else set()
        snapshot["communication"] = agent.communication_summary(active_ids=active_ids)
    if sim is not None:
        snapshot["population"] = len(sim.active_agent_ids())
        snapshot["tick"] = sim.world.tick
        snapshot["social_graph"] = sim.social_graph.summary()
        if agent.agent_id in sim.world.agent_positions:
            snapshot["position"] = list(sim.world.agent_positions[agent.agent_id].as_tuple())
        # Recent communication events involving this agent (technical only).
        recent = [
            e
            for e in getattr(sim, "communication_log", [])
            if e.get("sender") == agent.agent_id or e.get("receiver") == agent.agent_id
        ][-20:]
        snapshot["communication_events"] = recent

    if hasattr(agent, "loss_tracker"):
        active = set(sim.active_agent_ids()) if sim is not None else set()
        snapshot["computational_loss"] = agent.loss_tracker.summary(active_ids=active)

    return snapshot


def mind_delta(before: JSONDict, after: JSONDict) -> JSONDict:
    """Compact numeric deltas between two mind snapshots."""

    def _get(path: list, data: JSONDict, default=0.0):
        cur = data
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur if isinstance(cur, (int, float)) else default

    return {
        "entity_memory_count": _get(["entity_memory", "entity_count"], after) - _get(
            ["entity_memory", "entity_count"], before
        ),
        "relationship_count": _get(["relationships", "count"], after) - _get(
            ["relationships", "count"], before
        ),
        "tracked_entities": _get(["world_model", "tracked_entities"], after) - _get(
            ["world_model", "tracked_entities"], before
        ),
        "prediction_error_ema": _get(["prediction_error_ema"], after, 0.5) - _get(
            ["prediction_error_ema"], before, 0.5
        ),
        "mean_relationship_strength": _get(["relationships", "mean_strength"], after) - _get(
            ["relationships", "mean_strength"], before
        ),
        "unique_cells_visited": _get(["behaviour", "unique_cells_visited"], after) - _get(
            ["behaviour", "unique_cells_visited"], before
        ),
        "social_graph_edges": _get(["social_graph", "edge_count"], after) - _get(
            ["social_graph", "edge_count"], before
        ),
    }
