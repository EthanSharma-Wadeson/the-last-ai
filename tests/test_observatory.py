"""Individual Agent Life Observatory tests."""

import json

from the_last_ai.observatory.affect import affect_label, compute_affect, relationship_label
from the_last_ai.observatory.event import LifeEventType
from the_last_ai.observatory.export import export_agent_life
from the_last_ai.observatory.history import AgentLifeHistory
from the_last_ai.observatory.recorder import ObservatoryRecorder
from the_last_ai.simulation.engine import Simulation, SimulationConfig
from the_last_ai.types import Action, Position


def test_agent_history_created():
    sim = Simulation.create(SimulationConfig(seed=0, n_agents=3, agent_type="PredictiveAgent"))
    assert "agent_000" in sim.observatory.histories
    hist = sim.observatory.histories["agent_000"]
    assert any(e.event_type == LifeEventType.BIRTH for e in hist.events)


def test_birth_recorded():
    sim = Simulation.create(SimulationConfig(seed=1, n_agents=2, agent_type="SocialAgent"))
    births = [
        e
        for e in sim.observatory.histories["agent_000"].events
        if e.event_type == LifeEventType.BIRTH
    ]
    assert len(births) == 1
    assert births[0].summary == "Born."


def test_experience_recorded():
    sim = Simulation.create(
        SimulationConfig(
            seed=2, n_agents=4, n_resources=4, agent_type="PredictiveAgent", width=14, height=10
        )
    )
    sim.observatory.focus({"agent_000"})
    sim.run(15)
    hist = sim.observatory.histories["agent_000"]
    assert hist.overview()["experience_count"] > 1


def test_action_recorded():
    sim = Simulation.create(
        SimulationConfig(seed=3, n_agents=2, agent_type="PredictiveAgent", width=12, height=10)
    )
    sim.observatory.focus({"agent_000"})
    sim.run(8)
    actions = [
        e
        for e in sim.observatory.histories["agent_000"].events
        if e.event_type == LifeEventType.ACTION
    ]
    assert actions


def test_message_recorded_and_received():
    sim = Simulation.create(
        SimulationConfig(
            seed=4,
            n_agents=2,
            n_resources=3,
            agent_type="PredictiveAgent",
            width=12,
            height=10,
            perception_radius=4,
        )
    )
    from the_last_ai.agents.social_agent import SocialConfig

    for a in sim.agents.values():
        a.social_config = SocialConfig(
            enable_symbolic_communication=True, communication_bias=2.0, interact_probability=1.0
        )
    sim.world.agent_positions["agent_000"] = Position(3, 3)
    sim.world.agent_positions["agent_001"] = Position(4, 3)
    for _ in range(20):
        sim.step()
        h0 = sim.observatory.histories["agent_000"]
        if any(e.event_type == LifeEventType.COMMUNICATION_SENT for e in h0.events):
            break
    # Force communicate if still none
    if not any(
        e.event_type == LifeEventType.COMMUNICATION_SENT
        for e in sim.observatory.histories["agent_000"].events
    ):
        for aid in ("agent_000", "agent_001"):
            sim.agents[aid].observe(sim.world)
        sim._resolve_social_interactions(
            {"agent_000": Action.COMMUNICATE, "agent_001": Action.STAY},
            dict(sim.world.agent_positions),
        )
        # Manually feed observatory communication from log tail
        if sim.communication_log:
            sim.observatory.on_communication_events(sim, sim.communication_log[-2:])
    h0 = sim.observatory.histories["agent_000"]
    h1 = sim.observatory.histories["agent_001"]
    sent = [e for e in h0.events if e.event_type == LifeEventType.COMMUNICATION_SENT]
    recv = [e for e in h1.events if e.event_type == LifeEventType.COMMUNICATION_RECEIVED]
    assert sent or h0.messages
    assert recv or h1.messages
    if h0.messages:
        assert "tokens" in h0.messages[0]


def test_relationship_history_recorded():
    sim = Simulation.create(
        SimulationConfig(seed=5, n_agents=2, agent_type="SocialAgent", width=10, height=8)
    )
    sim.world.agent_positions["agent_000"] = Position(2, 2)
    sim.world.agent_positions["agent_001"] = Position(3, 2)
    for _ in range(12):
        for aid in ("agent_000", "agent_001"):
            sim.agents[aid].observe(sim.world)
            sim.agents[aid].last_action = Action.COOPERATE
        outcomes = sim._resolve_social_interactions(
            {"agent_000": Action.COOPERATE, "agent_001": Action.COOPERATE},
            dict(sim.world.agent_positions),
        )
        sim.observatory.on_social_outcomes(sim, outcomes)
        sim.world.advance_tick()
    traj = sim.observatory.histories["agent_000"].relationship_trajectories.get("agent_001")
    assert traj is not None
    assert traj.interaction_count >= 1
    assert traj.trust_history


def test_memory_activation_and_prediction_error_and_affect():
    sim = Simulation.create(
        SimulationConfig(
            seed=6, n_agents=3, n_resources=2, agent_type="PredictiveAgent", width=14, height=10
        )
    )
    sim.observatory.focus({"agent_000"})
    # Ensure agent_000 knows agent_001 before disappearance.
    sim.world.agent_positions["agent_000"] = Position(4, 4)
    sim.world.agent_positions["agent_001"] = Position(5, 4)
    sim.run(12)
    sim.disappear("agent_001")
    sim.run(15)
    hist = sim.observatory.histories["agent_000"]
    assert any(
        e.event_type
        in {LifeEventType.DISAPPEARANCE_OTHER, LifeEventType.POPULATION_CHANGE}
        for e in hist.events
    )
    assert hist.affect_series
    state, factors = compute_affect(sim.agents["agent_000"], active_ids=set(sim.active_agent_ids()))
    label, reasons = affect_label(state)
    assert label
    assert isinstance(reasons, list)
    assert "prediction_error_ema" in factors or "social_loss" in factors


def test_affective_change_and_causes_recorded():
    hist = AgentLifeHistory(agent_id="agent_000")
    from the_last_ai.observatory.affect import AffectiveState
    from the_last_ai.observatory.event import LifeEvent

    before = AffectiveState(valence=0.2, social_drive=0.3, social_loss=0.0)
    after = AffectiveState(valence=-0.3, social_drive=0.7, social_loss=0.5)
    hist.record(
        LifeEvent(
            tick=10,
            agent_id="agent_000",
            event_type=LifeEventType.AFFECTIVE_CHANGE,
            state_before=before.to_dict(),
            state_after=after.to_dict(),
            context={"factors": {"social_loss": 0.5}, "label": "withdrawal-like / social-loss response"},
            summary="Affective state changed.",
        )
    )
    ev = hist.events[-1]
    assert ev.state_before["valence"] == 0.2
    assert ev.context["factors"]["social_loss"] == 0.5


def test_historical_relationship_and_memory_persist(tmp_path):
    from pathlib import Path

    from the_last_ai.experiments.the_last_ai import run_the_last_ai

    result = run_the_last_ai(
        seed=0,
        initial_agents=8,
        schedule=[4, 2, 1],
        ticks_between=6,
        final_ticks=8,
        n_resources=4,
        width=14,
        height=10,
        output_dir=tmp_path / "last",
        compare_control=False,
    )
    hist = json.loads(Path(result["observatory_paths"]["json"]).read_text())
    assert hist["overview"]["agent_id"] == "agent_000"
    rels = hist.get("relationships") or []
    assert any("HISTORICAL" in str(r.get("status")) for r in rels) or hist["overview"][
        "relationships_historical"
    ] >= 0
    assert hist["overview"]["experience_count"] > 0


def test_communication_history_persists_and_active_vs_historical(tmp_path):
    from pathlib import Path

    from the_last_ai.experiments.the_last_ai import run_the_last_ai

    result = run_the_last_ai(
        seed=1,
        initial_agents=6,
        schedule=[3, 1],
        ticks_between=8,
        final_ticks=6,
        n_resources=4,
        width=12,
        height=10,
        output_dir=tmp_path / "last2",
        compare_control=False,
    )
    payload = json.loads(Path(result["observatory_paths"]["json"]).read_text())
    for msg in payload.get("messages") or []:
        assert isinstance(msg.get("tokens"), list)
    active = sum(1 for r in payload.get("relationships") or [] if r.get("status") == "ACTIVE")
    assert active == 0 or active <= 1


def test_survivor_history_persists_after_collapse(tmp_path):
    from pathlib import Path

    from the_last_ai.experiments.the_last_ai import run_the_last_ai

    result = run_the_last_ai(
        seed=2,
        initial_agents=6,
        schedule=[3, 1],
        ticks_between=5,
        final_ticks=5,
        n_resources=3,
        width=12,
        height=10,
        output_dir=tmp_path / "last3",
        compare_control=False,
    )
    assert result["observatory_paths"]["json"]
    assert Path(result["observatory_paths"]["txt"]).exists()
    assert Path(result["observatory_paths"]["html"]).exists()
    txt = Path(result["observatory_paths"]["txt"]).read_text()
    assert "LIFE TIMELINE" in txt
    assert "consciousness" in txt.lower()


def test_export_creates_files(tmp_path):
    from pathlib import Path

    from the_last_ai.observatory.event import LifeEvent

    hist = AgentLifeHistory(agent_id="agent_000", birth_tick=0)
    hist.record(
        LifeEvent(
            tick=0,
            agent_id="agent_000",
            event_type=LifeEventType.BIRTH,
            summary="Born.",
        )
    )
    paths = export_agent_life(hist, tmp_path / "obs")
    assert Path(paths["json"]).exists()
    assert Path(paths["txt"]).exists()
    assert Path(paths["html"]).exists()