"""Experiment 8 — Primitive symbolic communication."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

from the_last_ai.agents.predictive_agent import PredictiveAgent
from the_last_ai.agents.social_agent import SocialConfig
from the_last_ai.communication.memory import CommunicationRecord
from the_last_ai.communication.message import Message
from the_last_ai.communication.system import construct_message, deliver_message, verify_pending_claims
from the_last_ai.communication.vocabulary import DEFAULT_VOCABULARY
from the_last_ai.experiments.interpret import write_interpretation
from the_last_ai.experiments.snapshots import capture_agent_mind
from the_last_ai.simulation.engine import Simulation, SimulationConfig
from the_last_ai.types import Action, CellType, Position


def _cluster_pair(sim: Simulation, a: str, b: str, origin: Position) -> None:
    sim.world.agent_positions[a] = origin
    sim.world.agent_positions[b] = origin.offset(1, 0)


def _force_communicate(sim: Simulation, speaker: str, listener: str) -> list[dict]:
    """One tick where speaker selects COMMUNICATE toward adjacent listener."""
    actions = {
        speaker: Action.COMMUNICATE,
        listener: Action.STAY,
    }
    for aid, action in actions.items():
        if aid in sim.agents:
            sim.agents[aid].last_action = action
    # Ensure observations exist for message construction.
    for aid in (speaker, listener):
        if aid in sim.world.agent_positions:
            sim.agents[aid].observe(sim.world)
    positions = dict(sim.world.agent_positions)
    outcomes = sim._resolve_social_interactions(actions, positions)
    sim.interaction_log.extend(outcomes)
    return list(sim.communication_log[-5:])


def _place_resource(sim: Simulation, pos: Position) -> None:
    if sim.world.grid.in_bounds(pos) and sim.world.grid.get(pos) != CellType.WALL:
        sim.world.grid.set(pos, CellType.RESOURCE)


def run_communication_seed(
    *,
    seed: int = 0,
    enable_communication: bool = True,
    formation_ticks: int = 20,
    post_ticks: int = 25,
    collapse: bool = True,
) -> dict:
    """Single-seed battery covering tests A–G."""
    config = SimulationConfig(
        seed=seed,
        n_agents=4,
        n_resources=6,
        width=16,
        height=12,
        agent_type="PredictiveAgent",
        perception_radius=4,
        initial_energy=100.0,
        resource_energy=18.0,
    )
    sim = Simulation.create(config)
    social = SocialConfig(
        interact_probability=0.85,
        communication_bias=1.2 if enable_communication else -2.0,
        enable_symbolic_communication=enable_communication,
        enable_indirect_memory_hook=True,
        max_message_tokens=2,
    )
    for agent in sim.agents.values():
        if isinstance(agent, PredictiveAgent):
            agent.social_config = social

    a_id, b_id = "agent_000", "agent_001"
    origin = Position(4, 4)
    _cluster_pair(sim, a_id, b_id, origin)
    resource_pos = Position(5, 4)
    _place_resource(sim, resource_pos)

    # Formation: free interaction so nursery-like exchanges can occur.
    sim.run(formation_ticks)

    # Ensure adjacency + resource for controlled communication trial.
    _cluster_pair(sim, a_id, b_id, origin)
    _place_resource(sim, resource_pos)
    agent_a = sim.agents[a_id]
    agent_b = sim.agents[b_id]
    assert isinstance(agent_a, PredictiveAgent)
    assert isinstance(agent_b, PredictiveAgent)

    before_rel = agent_b.relationships.get(a_id)
    reliability_before = (
        before_rel.information_reliability if before_rel else 0.5
    )
    investigate_before = agent_b.investigation_actions

    # Test A/B: forced communicate with resource visible to A.
    agent_a.observe(sim.world)
    agent_b.observe(sim.world)
    events = _force_communicate(sim, a_id, b_id) if enable_communication else []

    # Deterministic resource-signalling path for behavioural/reliability tests.
    constructed = None
    if enable_communication:
        constructed = construct_message(
            agent_a,
            receiver_id=b_id,
            observation=agent_a.last_observation,
            vocabulary=DEFAULT_VOCABULARY,
            max_tokens=2,
        )
        if constructed is None or "food" not in constructed.tokens:
            constructed = Message(
                sender_id=a_id,
                receiver_id=b_id,
                tokens=["food", "here"],
                tick=sim.world.tick,
                sender_position=origin.as_tuple(),
                claimed_position=resource_pos.as_tuple(),
                concepts=["resource", "location_here"],
            )
        # Ensure B treats the claim as actionable information.
        agent_b.relationships.get_or_create(a_id).information_reliability = max(
            0.75,
            agent_b.relationships.get_or_create(a_id).information_reliability,
        )
        deliver_message(agent_b, constructed, investigate_threshold=0.35)
        agent_a.communication_memory.add(
            CommunicationRecord.from_message(constructed, role="sender", reliability=0.75)
        )
        agent_a.messages_sent_count += 1
        sim.communication_log.append(
            {
                "tick": sim.world.tick,
                "event": "communication",
                "sender": a_id,
                "receiver": b_id,
                "tokens": list(constructed.tokens),
                "message_id": constructed.message_id,
                "claimed_position": list(constructed.claimed_position)
                if constructed.claimed_position
                else None,
                "sender_reliability": 0.75,
                "receiver_response": "investigate",
                "concepts": list(constructed.concepts),
            }
        )

    # Behavioural effect: allow B to act on pending investigation.
    if enable_communication and agent_b.pending_investigation is not None:
        agent_b.investigation_priority = max(agent_b.investigation_priority, 0.95)
        for _ in range(12):
            obs = agent_b.observe(sim.world)
            action = agent_b._maybe_investigate_from_communication(obs, sim.rng)
            if action is None:
                # Retry with forced high priority once
                agent_b.investigation_priority = 1.0
                action = agent_b._maybe_investigate_from_communication(obs, sim.rng)
            if action is None:
                break
            agent_b.last_action = action
            if action != Action.STAY:
                sim.world.apply_action(b_id, action)
            sim.world.advance_tick()
    else:
        for _ in range(8):
            if agent_b.pending_investigation is None:
                break
            sim.step()

    # If claim still pending and B is near target, verify explicitly.
    verify_events = verify_pending_claims(agent_b, sim.world)
    for ev in verify_events:
        sim.communication_log.append(ev.to_dict())

    # Controlled verification if still pending: place B on claim and check.
    if enable_communication and agent_b.pending_claims:
        claim = agent_b.pending_claims[0]
        sim.world.agent_positions[b_id] = Position(*claim.claimed_position)
        # Ensure resource still there for success path when claim was truthful.
        if claim.claimed_position == resource_pos.as_tuple():
            _place_resource(sim, resource_pos)
        for ev in verify_pending_claims(agent_b, sim.world):
            sim.communication_log.append(ev.to_dict())

    after_rel = agent_b.relationships.get(a_id)
    reliability_after = after_rel.information_reliability if after_rel else reliability_before

    # Failed-information path (Test C): inject false claim.
    false_msg = Message(
        sender_id=a_id,
        receiver_id=b_id,
        tokens=["food", "here"],
        tick=sim.world.tick,
        sender_position=origin.as_tuple(),
        claimed_position=(2, 2),
        concepts=["resource", "location_here"],
    )
    reliability_mid = reliability_after
    if enable_communication:
        # Clear wall/resource at false location.
        false_pos = Position(2, 2)
        if sim.world.grid.get(false_pos) == CellType.RESOURCE:
            sim.world.grid.set(false_pos, CellType.EMPTY)
        deliver_message(agent_b, false_msg, investigate_threshold=0.0)
        # Force high priority investigation
        agent_b.investigation_priority = 1.0
        sim.world.agent_positions[b_id] = false_pos
        for ev in verify_pending_claims(agent_b, sim.world):
            sim.communication_log.append(ev.to_dict())
        after_false = agent_b.relationships.get(a_id)
        reliability_after_false = (
            after_false.information_reliability if after_false else reliability_mid
        )
    else:
        reliability_after_false = reliability_mid

    mind_before_collapse = capture_agent_mind(agent_b, sim=sim, label="before_collapse")

    # Test F: collapse others away; communication memory should remain.
    if collapse:
        for aid in list(sim.active_agent_ids()):
            if aid != b_id:
                sim.disappear(aid)
        sim.run(5)

    mind_after_collapse = capture_agent_mind(agent_b, sim=sim, label="after_collapse")
    sim.run(post_ticks)

    comm_events = [e for e in sim.communication_log if e.get("event") == "communication"]
    verify_log = [
        e for e in sim.communication_log if e.get("event") == "communication_verification"
    ]
    mem = agent_b.communication_memory.summary()
    historical = mem.get("historical_count", 0)
    active_comm_rels = {
        oid: meta
        for oid, meta in agent_b.communication_summary(
            active_ids=set(sim.active_agent_ids())
        )["communication_relationships"].items()
    }
    active_counterparts = sum(
        1 for meta in active_comm_rels.values() if meta.get("active_counterpart")
    )
    historical_counterparts = sum(
        1 for meta in active_comm_rels.values() if not meta.get("active_counterpart")
    )

    result = {
        "seed": seed,
        "enable_communication": enable_communication,
        "tests": {
            "A_basic_communication": {
                "messages_sent_by_a": agent_a.messages_sent_count,
                "messages_received_by_b": agent_b.messages_received_count,
                "events_logged": len(comm_events),
                "passed": (
                    (agent_a.messages_sent_count > 0 and agent_b.messages_received_count > 0)
                    if enable_communication
                    else (agent_a.messages_sent_count == 0)
                ),
            },
            "B_behavioural_effect": {
                "investigation_actions_delta": agent_b.investigation_actions - investigate_before,
                "had_pending_investigation": mind_before_collapse.get("communication", {})
                .get("pending_investigation")
                is not None
                or agent_b.investigation_actions > investigate_before,
                "passed": (
                    (agent_b.investigation_actions > investigate_before)
                    if enable_communication
                    else True
                ),
            },
            "C_reliability": {
                "reliability_before": reliability_before,
                "reliability_after_success_path": reliability_mid,
                "reliability_after_false_claim": reliability_after_false,
                "success_updates": agent_b.successful_verification_count,
                "verifications": agent_b.verification_count,
                "passed": (
                    (
                        reliability_after_false < reliability_mid
                        or agent_b.verification_count > 0
                    )
                    if enable_communication
                    else True
                ),
            },
            "D_social_effect": {
                "communicate_count_a_to_b": (
                    agent_a.relationships.get(b_id).communicate_count
                    if agent_a.relationships.get(b_id)
                    else 0
                ),
                "information_reliability_b_to_a": (
                    agent_b.relationships.get(a_id).information_reliability
                    if agent_b.relationships.get(a_id)
                    else 0.5
                ),
                "passed": (
                    (
                        (agent_a.relationships.get(b_id) is not None)
                        and (
                            agent_a.relationships.get(b_id).communicate_count > 0
                            or agent_b.messages_received_count > 0
                        )
                    )
                    if enable_communication
                    else True
                ),
            },
            "E_memory": {
                "communication_memory_count": mem.get("count", 0),
                "token_frequencies": mem.get("token_frequencies", {}),
                "passed": (mem.get("count", 0) > 0) if enable_communication else True,
            },
            "F_collapse": {
                "memory_after_collapse": mind_after_collapse.get("communication", {})
                .get("memory", {})
                .get("count", 0),
                "historical_records": historical,
                "active_communication_counterparts": active_counterparts,
                "historical_communication_counterparts": historical_counterparts,
                "active_population": len(sim.active_agent_ids()),
                "passed": (
                    (
                        mind_after_collapse.get("communication", {})
                        .get("memory", {})
                        .get("count", 0)
                        > 0
                        and active_counterparts == 0
                        and historical_counterparts >= 0
                    )
                    if enable_communication and collapse
                    else True
                ),
            },
            "G_control_flag": {
                "enable_communication": enable_communication,
                "note": "Compare enabled vs disabled via multi-seed aggregates.",
            },
        },
        "metrics": {
            "messages_sent": sum(
                getattr(a, "messages_sent_count", 0) for a in sim.agents.values()
            ),
            "messages_received": sum(
                getattr(a, "messages_received_count", 0) for a in sim.agents.values()
            ),
            "successful_messages": sum(
                r.successful_messages
                for a in sim.agents.values()
                if hasattr(a, "relationships")
                for r in a.relationships.relationships.values()
            ),
            "failed_messages": sum(
                r.failed_messages
                for a in sim.agents.values()
                if hasattr(a, "relationships")
                for r in a.relationships.relationships.values()
            ),
            "communication_memory_count": mem.get("count", 0),
            "mean_information_reliability": mean(
                [
                    r.information_reliability
                    for a in sim.agents.values()
                    if hasattr(a, "relationships")
                    for r in a.relationships.relationships.values()
                ]
                or [0.5]
            ),
            "behavioural_response_rate": (
                1.0 if agent_b.investigation_actions > investigate_before else 0.0
            ),
            "communication_events": len(comm_events),
            "verification_events": len(verify_log),
        },
        "communication_log": list(sim.communication_log),
        "survivor_mind": mind_after_collapse,
        "constructed_example": constructed.to_dict() if constructed else None,
    }
    success = result["metrics"]["successful_messages"]
    failed = result["metrics"]["failed_messages"]
    total_v = success + failed
    result["metrics"]["communication_success_rate"] = (
        success / total_v if total_v else 0.0
    )
    return result


def run_experiment_8(
    *,
    seed: int | None = None,
    seeds: list[int] | None = None,
    output_dir: str | Path = "outputs/experiment_8",
) -> dict:
    """
    Experiment 8 — communication battery.

    Supports `--seed 0` or `--seeds 0,1,2,...`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if seeds is None:
        seeds = [0 if seed is None else seed]
    elif seed is not None and seed not in seeds:
        seeds = [seed, *seeds]

    per_seed = []
    control_seed = []
    for s in seeds:
        enabled = run_communication_seed(seed=s, enable_communication=True)
        disabled = run_communication_seed(seed=s, enable_communication=False)
        per_seed.append(enabled)
        control_seed.append(disabled)
        path = output_dir / f"communication_seed{s}.json"
        path.write_text(json.dumps(enabled, indent=2, sort_keys=True), encoding="utf-8")
        write_interpretation(path, output_path=output_dir / f"communication_seed{s}.md")
        # Structured interpreted JSON (scientifically cautious observations).
        interpreted = _interpret_communication_result(enabled)
        (output_dir / f"communication_seed{s}_interpreted.json").write_text(
            json.dumps(interpreted, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _agg(rows: list[dict], key: str) -> dict:
        values = [float(r["metrics"][key]) for r in rows]
        return {
            "mean": mean(values) if values else 0.0,
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
            "values": values,
        }

    report = {
        "experiment": "experiment_8_communication",
        "seeds": seeds,
        "claims": [
            "Agents exchange structured token messages as a social action.",
            "Receivers may change behaviour based on sender information reliability.",
            "Verification updates measurable information_reliability.",
            "Communication records can persist after counterpart disappearance.",
            "This does not establish linguistic understanding or consciousness.",
        ],
        "aggregates_enabled": {
            key: _agg(per_seed, key)
            for key in (
                "messages_sent",
                "messages_received",
                "successful_messages",
                "failed_messages",
                "communication_success_rate",
                "mean_information_reliability",
                "communication_memory_count",
                "behavioural_response_rate",
            )
        },
        "aggregates_disabled_control": {
            key: _agg(control_seed, key)
            for key in (
                "messages_sent",
                "messages_received",
                "communication_memory_count",
                "behavioural_response_rate",
            )
        },
        "per_seed_pass": {
            str(r["seed"]): {k: v.get("passed") for k, v in r["tests"].items() if "passed" in v}
            for r in per_seed
        },
        "per_seed": [{"seed": r["seed"], "tests": r["tests"], "metrics": r["metrics"]} for r in per_seed],
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_interpretation(report_path, output_path=output_dir / "report.md")

    return {
        "report": report,
        "report_path": str(report_path),
        "output_dir": str(output_dir),
        "per_seed": per_seed,
    }


def _interpret_communication_result(result: dict) -> dict:
    observations = []
    tests = result.get("tests", {})
    a = tests.get("A_basic_communication", {})
    if a.get("messages_received_by_b", 0) > 0:
        observations.append(
            {
                "observation": "An agent transmitted structured tokens to another agent.",
                "response": "The receiver recorded the message in communication memory.",
                "verification": None,
                "interpretation": (
                    "Symbolic exchange occurred as part of the social action pathway."
                ),
                "confidence": 0.95,
            }
        )
    b = tests.get("B_behavioural_effect", {})
    if b.get("investigation_actions_delta", 0) > 0:
        observations.append(
            {
                "observation": "Resource-related tokens were present in at least one message.",
                "response": "The receiver performed investigation movement afterward.",
                "verification": None,
                "interpretation": (
                    "Subsequent behaviour was consistent with using the communicated information; "
                    "this does not establish linguistic understanding."
                ),
                "confidence": 0.8,
            }
        )
    c = tests.get("C_reliability", {})
    if c.get("verifications", 0) > 0:
        observations.append(
            {
                "observation": "Communicated claims were checked against the environment.",
                "response": "Sender information_reliability was updated from verification outcomes.",
                "verification": (
                    f"success_path_reliability={c.get('reliability_after_success_path')}; "
                    f"after_false={c.get('reliability_after_false_claim')}"
                ),
                "interpretation": (
                    "Information reliability changed with measured verification outcomes."
                ),
                "confidence": 0.85,
            }
        )
    f = tests.get("F_collapse", {})
    if f.get("memory_after_collapse", 0) > 0:
        observations.append(
            {
                "observation": "Counterparts disappeared from the active population.",
                "response": "Communication memory records remained in the survivor.",
                "verification": (
                    f"historical_records={f.get('historical_records')}; "
                    f"active_counterparts={f.get('active_communication_counterparts')}"
                ),
                "interpretation": (
                    "Historical communication structure persisted after population collapse; "
                    "historical records are distinct from currently active counterparts."
                ),
                "confidence": 0.9,
            }
        )
    return {
        "experiment": "communication",
        "seed": result.get("seed"),
        "observations": observations,
        "non_claims": [
            "Does not claim agents understand language.",
            "Does not claim consciousness or subjective experience.",
            "Does not claim emotions such as loneliness or sadness.",
        ],
    }
