"""Save and load complete simulation snapshots."""

from __future__ import annotations

import json
from pathlib import Path

from the_last_ai import __version__
from the_last_ai.agents.base import Agent, create_agent
from the_last_ai.metrics.recorder import MetricsRecorder
from the_last_ai.types import JSONDict
from the_last_ai.world.world import World


def snapshot_to_dict(
    *,
    world: World,
    agents: dict[str, Agent],
    seed: int,
    metrics: MetricsRecorder | None = None,
    config: JSONDict | None = None,
) -> JSONDict:
    return {
        "software_version": __version__,
        "seed": seed,
        "config": config or {},
        "world": world.to_dict(),
        "agents": {agent_id: agent.to_dict() for agent_id, agent in agents.items()},
        "metrics": metrics.to_dict() if metrics is not None else None,
    }


def save_snapshot(path: str | Path, data: JSONDict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_snapshot(path: str | Path) -> JSONDict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def restore_agents(data: JSONDict) -> dict[str, Agent]:
    agents: dict[str, Agent] = {}
    for agent_id, agent_data in data.get("agents", {}).items():
        agent = create_agent(agent_data["type"], agent_id)
        agent.load_state(agent_data)
        agents[agent_id] = agent
    return agents


def restore_world(data: JSONDict) -> World:
    return World.from_dict(data["world"])


def restore_metrics(data: JSONDict) -> MetricsRecorder | None:
    if data.get("metrics") is None:
        return None
    return MetricsRecorder.from_dict(data["metrics"])
