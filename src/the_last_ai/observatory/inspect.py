"""CLI helpers and inspect-agent for the Life Observatory."""

from __future__ import annotations

import json
from pathlib import Path

from the_last_ai.experiments.snapshots import capture_agent_mind
from the_last_ai.observatory.export import export_agent_life
from the_last_ai.observatory.history import AgentLifeHistory
from the_last_ai.observatory.recorder import ObservatoryRecorder
from the_last_ai.simulation.engine import Simulation


def inspect_agent_from_path(
    path: str | Path,
    agent_id: str,
    *,
    output_dir: str | Path | None = None,
) -> dict:
    """
    Load a saved simulation, Last AI report folder, or life JSON and export biography.

    Accepts:
    - final_sim_seedN.json (simulation snapshot with observatory)
    - agent_XXX_life.json (already exported)
    - directory containing final_sim / observatory
    """
    path = Path(path)
    if path.is_dir():
        # Prefer observatory export, else final sim
        candidates = list(path.glob(f"{agent_id}_life.json")) + list(
            path.glob(f"**/observatory_seed*/{agent_id}_life.json")
        )
        if candidates:
            path = candidates[0]
        else:
            sims = list(path.glob("final_sim_seed*.json"))
            if not sims:
                raise FileNotFoundError(f"No simulation or life JSON under {path}")
            path = sims[0]

    data = json.loads(path.read_text(encoding="utf-8"))
    out = Path(output_dir) if output_dir else path.parent / "inspect"
    out.mkdir(parents=True, exist_ok=True)

    if data.get("schema") == "the_last_ai.agent_life.v1":
        # Re-export presentation from authoritative life JSON
        history = AgentLifeHistory.from_dict(data.get("history") or data)
        paths = export_agent_life(
            history,
            out,
            mind=data.get("mind_snapshot"),
            meta=data.get("meta"),
        )
        return {"paths": paths, "overview": data.get("overview"), "source": str(path)}

    if "observatory" in data and "agents" in data:
        sim = Simulation.load(path)
        if agent_id not in sim.observatory.histories:
            raise KeyError(f"{agent_id} has no observatory history in {path}")
        history = sim.observatory.histories[agent_id]
        mind = None
        if agent_id in sim.agents:
            mind = capture_agent_mind(sim.agents[agent_id], sim=sim, label="inspect")
        paths = export_agent_life(
            history,
            out,
            mind=mind,
            active_ids=set(sim.active_agent_ids()),
            meta={"source": str(path)},
        )
        return {
            "paths": paths,
            "overview": history.overview(
                active_ids=set(sim.active_agent_ids()), final_tick=sim.world.tick
            ),
            "source": str(path),
        }

    if "histories" in data:
        rec = ObservatoryRecorder.from_dict(data)
        if agent_id not in rec.histories:
            raise KeyError(agent_id)
        history = rec.histories[agent_id]
        paths = export_agent_life(history, out, meta={"source": str(path)})
        return {"paths": paths, "overview": history.overview(), "source": str(path)}

    raise ValueError(f"Unrecognised observatory/simulation payload: {path}")
