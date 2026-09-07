"""Tests for primitive symbolic communication."""

from the_last_ai.agents.predictive_agent import PredictiveAgent
from the_last_ai.agents.social_agent import SocialConfig
from the_last_ai.communication.memory import CommunicationMemory, CommunicationRecord
from the_last_ai.communication.message import Message
from the_last_ai.communication.system import (
    construct_message,
    deliver_message,
    exchange_communication,
    verify_pending_claims,
)
from the_last_ai.communication.vocabulary import DEFAULT_VOCABULARY, Concept, Vocabulary
from the_last_ai.experiments.communication_experiment import run_communication_seed, run_experiment_8
from the_last_ai.experiments.interpret import detect_kind, interpret_json_file
from the_last_ai.simulation.engine import Simulation, SimulationConfig
from the_last_ai.social.types import InteractionKind
from the_last_ai.types import Action, CellType, Position


def test_message_creation():
    msg = Message(
        sender_id="agent_001",
        receiver_id="agent_004",
        tokens=["food", "here"],
        tick=142,
        sender_position=(4, 7),
        claimed_position=(5, 7),
    )
    assert msg.tokens == ["food", "here"]
    assert "food" in msg.render_line()
    data = msg.to_dict()
    assert Message.from_dict(data).tokens == ["food", "here"]


def test_message_delivery():
    sim = Simulation.create(
        SimulationConfig(seed=0, n_agents=2, n_resources=2, agent_type="PredictiveAgent", width=12, height=10)
    )
    a, b = sim.agents["agent_000"], sim.agents["agent_001"]
    assert isinstance(a, PredictiveAgent) and isinstance(b, PredictiveAgent)
    sim.world.agent_positions["agent_000"] = Position(3, 3)
    sim.world.agent_positions["agent_001"] = Position(4, 3)
    a.observe(sim.world)
    msg = construct_message(a, receiver_id="agent_001", observation=a.last_observation)
    assert msg is not None
    response = deliver_message(b, msg, investigate_threshold=0.0)
    assert b.messages_received_count == 1
    assert b.communication_memory.summary()["count"] == 1
    assert response in {"investigate", "acknowledge", "recorded", "approach_sender", "assist_move"}


def test_message_not_delivered_to_wrong_agent():
    msg = Message(
        sender_id="agent_000",
        receiver_id="agent_001",
        tokens=["hi"],
        tick=1,
    )
    sim = Simulation.create(
        SimulationConfig(seed=1, n_agents=3, agent_type="PredictiveAgent", width=12, height=10)
    )
    wrong = sim.agents["agent_002"]
    assert isinstance(wrong, PredictiveAgent)
    deliver_message(wrong, msg)
    # Wrong agent can still store if called directly — exchange path targets listener only.
    events = exchange_communication(
        sim.agents["agent_000"],
        sim.agents["agent_001"],
        intent_a=InteractionKind.COMMUNICATE,
        intent_b=None,
        tick=1,
        positions=dict(sim.world.agent_positions),
    )
    assert all(e.receiver == "agent_001" for e in events)


def test_message_memory_capacity():
    mem = CommunicationMemory(capacity=3)
    for i in range(5):
        mem.add(
            CommunicationRecord(
                message_id=str(i),
                tick=i,
                sender_id="a",
                receiver_id="b",
                tokens=["hi"],
            )
        )
    assert mem.summary()["count"] == 3


def test_communication_changes_behaviour():
    sim = Simulation.create(
        SimulationConfig(
            seed=2,
            n_agents=2,
            n_resources=3,
            agent_type="PredictiveAgent",
            width=14,
            height=10,
            perception_radius=4,
        )
    )
    a, b = sim.agents["agent_000"], sim.agents["agent_001"]
    assert isinstance(a, PredictiveAgent) and isinstance(b, PredictiveAgent)
    a.social_config = SocialConfig(enable_symbolic_communication=True)
    b.social_config = SocialConfig(enable_symbolic_communication=True)
    target = Position(8, 5)
    sim.world.grid.set(target, CellType.RESOURCE)
    msg = Message(
        sender_id="agent_000",
        receiver_id="agent_001",
        tokens=["food", "here"],
        tick=0,
        claimed_position=target.as_tuple(),
        concepts=["resource", "location_here"],
    )
    # High reliability so investigation triggers.
    b.relationships.get_or_create("agent_000").information_reliability = 0.9
    deliver_message(b, msg, investigate_threshold=0.35)
    assert b.pending_investigation is not None
    before = b.investigation_actions
    sim.world.agent_positions["agent_001"] = Position(6, 5)
    b.observe(sim.world)
    action = b.select_action(b.last_observation, sim.rng)
    assert action in {Action.MOVE_EAST, Action.MOVE_WEST, Action.MOVE_NORTH, Action.MOVE_SOUTH, Action.STAY}
    # Force investigate path
    b.investigation_priority = 1.0
    action = b._maybe_investigate_from_communication(b.last_observation, sim.rng)
    assert action is not None
    assert b.investigation_actions >= before


def test_successful_information_updates_reliability():
    sim = Simulation.create(
        SimulationConfig(seed=3, n_agents=2, n_resources=1, agent_type="PredictiveAgent", width=12, height=10)
    )
    b = sim.agents["agent_001"]
    assert isinstance(b, PredictiveAgent)
    target = Position(5, 5)
    sim.world.grid.set(target, CellType.RESOURCE)
    msg = Message(
        sender_id="agent_000",
        receiver_id="agent_001",
        tokens=["food", "here"],
        tick=0,
        claimed_position=target.as_tuple(),
        concepts=["resource", "location_here"],
    )
    b.relationships.get_or_create("agent_000").information_reliability = 0.5
    deliver_message(b, msg, investigate_threshold=0.0)
    sim.world.agent_positions["agent_001"] = target
    events = verify_pending_claims(b, sim.world)
    assert events and events[0].verified is True
    assert b.relationships.get("agent_000").information_reliability > 0.5
    assert b.relationships.get("agent_000").successful_messages >= 1


def test_failed_information_updates_reliability():
    sim = Simulation.create(
        SimulationConfig(seed=4, n_agents=2, n_resources=0, agent_type="PredictiveAgent", width=12, height=10)
    )
    b = sim.agents["agent_001"]
    assert isinstance(b, PredictiveAgent)
    target = Position(5, 5)
    sim.world.grid.set(target, CellType.EMPTY)
    msg = Message(
        sender_id="agent_000",
        receiver_id="agent_001",
        tokens=["food", "here"],
        tick=0,
        claimed_position=target.as_tuple(),
        concepts=["resource", "location_here"],
    )
    b.relationships.get_or_create("agent_000").information_reliability = 0.7
    deliver_message(b, msg, investigate_threshold=0.0)
    sim.world.agent_positions["agent_001"] = target
    events = verify_pending_claims(b, sim.world)
    assert events and events[0].verified is False
    assert b.relationships.get("agent_000").information_reliability < 0.7
    assert b.relationships.get("agent_000").failed_messages >= 1


def test_communication_updates_relationship():
    sim = Simulation.create(
        SimulationConfig(seed=5, n_agents=2, agent_type="PredictiveAgent", width=10, height=8)
    )
    sim.world.agent_positions["agent_000"] = Position(2, 2)
    sim.world.agent_positions["agent_001"] = Position(3, 2)
    for aid in ("agent_000", "agent_001"):
        sim.agents[aid].observe(sim.world)
        sim.agents[aid].last_action = Action.COMMUNICATE
    outcomes = sim._resolve_social_interactions(
        {"agent_000": Action.COMMUNICATE, "agent_001": Action.COMMUNICATE},
        dict(sim.world.agent_positions),
    )
    assert outcomes
    a = sim.agents["agent_000"]
    assert isinstance(a, PredictiveAgent)
    rel = a.relationships.get("agent_001")
    assert rel is not None
    assert rel.communicate_count >= 1


def test_communication_persists_after_agent_disappearance():
    result = run_communication_seed(seed=6, enable_communication=True, collapse=True)
    assert result["tests"]["F_collapse"]["memory_after_collapse"] > 0
    assert result["tests"]["F_collapse"]["active_population"] == 1


def test_historical_messages_are_not_active_relationships():
    result = run_communication_seed(seed=7, enable_communication=True, collapse=True)
    assert result["tests"]["F_collapse"]["active_communication_counterparts"] == 0
    assert result["tests"]["F_collapse"]["historical_records"] >= 0
    mind = result["survivor_mind"]
    comm = mind.get("communication", {})
    for meta in (comm.get("communication_relationships") or {}).values():
        assert meta.get("active_counterpart") is False


def test_technical_json_contains_communication(tmp_path):
    out = run_experiment_8(seeds=[0], output_dir=tmp_path / "e8")
    seed_path = tmp_path / "e8" / "communication_seed0.json"
    data = __import__("json").loads(seed_path.read_text())
    assert "communication_log" in data
    assert data["metrics"]["messages_sent"] >= 0
    assert out["report_path"]


def test_interpreted_json_contains_communication_analysis(tmp_path):
    run_experiment_8(seeds=[0], output_dir=tmp_path / "e8")
    interpreted = __import__("json").loads(
        (tmp_path / "e8" / "communication_seed0_interpreted.json").read_text()
    )
    assert interpreted["experiment"] == "communication"
    assert "observations" in interpreted
    assert any("non_claims" in interpreted for _ in [0])
    text = interpret_json_file(tmp_path / "e8" / "communication_seed0.json")
    assert "Communication" in text


def test_deterministic_seed_reproduces_results():
    a = run_communication_seed(seed=11, enable_communication=True)
    b = run_communication_seed(seed=11, enable_communication=True)
    assert a["metrics"]["messages_sent"] == b["metrics"]["messages_sent"]
    assert a["metrics"]["messages_received"] == b["metrics"]["messages_received"]
    assert a["tests"]["A_basic_communication"]["passed"] == b["tests"]["A_basic_communication"]["passed"]


def test_vocabulary_separates_token_from_meaning():
    vocab = Vocabulary()
    assert vocab.meaning("food") == Concept.RESOURCE
    # Replace token without rewriting pipeline.
    vocab.token_to_concept = {"ka": Concept.RESOURCE, "here": Concept.LOCATION_HERE}
    assert vocab.meaning("ka") == Concept.RESOURCE
    assert "food" not in vocab.token_to_concept


def test_detect_kind_communication():
    assert detect_kind({"tests": {"A_basic_communication": {}}, "seed": 0}) == "communication_seed"
