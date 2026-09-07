"""Tests for computational loss subsystem and Experiment 9."""

from __future__ import annotations

import json

from the_last_ai.experiments.computational_loss import (
    interpret_computational_loss,
    run_computational_loss_seed,
    run_experiment_9,
    run_loss_condition,
)
from the_last_ai.loss.model import (
    EntityStatus,
    aggregate_social_loss,
    compute_entity_significance,
    compute_entity_social_loss,
)
from the_last_ai.loss.tracker import LossTracker
from the_last_ai.observatory.export import build_export_payload, render_biography_html
from the_last_ai.simulation.engine import Simulation, SimulationConfig
from the_last_ai.types import Position


def test_equations_monotonic_in_significance():
    weak = compute_entity_social_loss(
        significance=0.2,
        presence_expectation=0.8,
        information_reliability=0.7,
        ticks_since_disappearance=0,
        prediction_disruption=0.0,
        search_pressure=0.0,
        memory_retrieved_this_tick=False,
    )
    strong = compute_entity_social_loss(
        significance=0.9,
        presence_expectation=0.8,
        information_reliability=0.7,
        ticks_since_disappearance=0,
        prediction_disruption=0.0,
        search_pressure=0.0,
        memory_retrieved_this_tick=False,
    )
    assert strong > weak


def test_aggregate_saturates_less_than_sum():
    vals = [0.5, 0.5, 0.5]
    assert aggregate_social_loss(vals) < sum(vals)
    assert 0.0 <= aggregate_social_loss(vals) <= 1.0


def test_disappearance_recorded_and_historical():
    sim = Simulation.create(
        SimulationConfig(
            seed=11,
            n_agents=3,
            n_resources=2,
            agent_type="PredictiveAgent",
            width=12,
            height=10,
            perception_radius=4,
        )
    )
    sim.world.agent_positions["agent_000"] = Position(3, 3)
    sim.world.agent_positions["agent_001"] = Position(4, 3)
    sim.run(30)
    observer = sim.agents["agent_000"]
    assert hasattr(observer, "loss_tracker")
    sim.disappear("agent_001", reversible=False)
    rec = observer.loss_tracker.get("agent_001")
    assert rec is not None
    assert rec.status == EntityStatus.HISTORICAL
    assert rec.disappearance_tick is not None
    assert "agent_001" in observer.ltm.entities  # memory not wiped
    assert observer.relationships.get("agent_001") is not None


def test_prediction_and_loss_update_after_absence():
    result = run_loss_condition(
        seed=7,
        label="strong",
        formation_ticks=40,
        post_ticks=40,
        formation_bias=1.4,
        disappear=True,
    )
    result.pop("_sim", None)
    result.pop("_history", None)
    assert result["disappearance_tick"] is not None
    assert result["relationship_before"]["interaction_count"] >= 1
    assert result["peaks"]["peak_social_loss"] >= 0.0
    rec = result["loss_record"]
    assert rec.get("status") == "HISTORICAL"
    assert len(rec.get("loss_trajectory") or []) >= 1
    # Memory should persist (may decay but representation remains)
    assert result["relationship_after"]["memory_strength"] >= 0.0


def test_weak_vs_strong_can_differ():
    weak = run_loss_condition(
        seed=21,
        label="minimal",
        formation_ticks=2,
        post_ticks=50,
        formation_bias=0.1,
        interact_probability=0.4,
        disappear=True,
    )
    strong = run_loss_condition(
        seed=21,
        label="strong",
        formation_ticks=55,
        post_ticks=50,
        formation_bias=1.5,
        disappear=True,
    )
    for r in (weak, strong):
        r.pop("_sim", None)
        r.pop("_history", None)
    # Relationship strength should differ; social_loss response is allowed to be null
    assert (
        strong["relationship_before"]["relationship_strength"]
        >= weak["relationship_before"]["relationship_strength"]
    )
    # Both produce valid records
    assert weak["loss_record"].get("disappearance_tick") is not None
    assert strong["loss_record"].get("disappearance_tick") is not None


def test_control_no_disappear_has_no_partner_loss_record():
    ctrl = run_loss_condition(
        seed=33,
        label="control_no_disappear",
        formation_ticks=30,
        post_ticks=30,
        disappear=False,
    )
    ctrl.pop("_sim", None)
    ctrl.pop("_history", None)
    assert ctrl["disappearance_tick"] is None
    assert not ctrl["loss_record"] or ctrl["loss_record"].get("disappearance_tick") is None


def test_multiple_disappearances_accumulate_records():
    from the_last_ai.experiments.computational_loss import _run_multiple_disappearances

    multi = _run_multiple_disappearances(seed=44)
    assert multi["survivor_id"] == "agent_000"
    assert len(multi["sequential_peaks"]) >= 1
    counts = multi["final_loss_summary"]["counts"]
    assert counts["historical"] + counts["forgotten"] >= 1


def test_interpreted_derived_from_technical():
    tech = {
        "experiment": "computational_loss",
        "comparison": {
            "peak_social_loss": {"strong": 0.4, "minimal": 0.1},
            "strong_minus_minimal_social_loss": 0.3,
            "strong_minus_control_social_loss": 0.35,
        },
        "non_claims": ["no feelings"],
    }
    interpreted = interpret_computational_loss(tech)
    assert interpreted["from_technical"] is True
    assert interpreted["observations"]
    assert "subjective" in interpreted["scientific_caveat"].lower() or "≠" in interpreted[
        "scientific_caveat"
    ]


def test_observatory_loss_tab_payload():
    result = run_loss_condition(
        seed=5,
        label="strong",
        formation_ticks=35,
        post_ticks=25,
        disappear=True,
    )
    sim = result.pop("_sim")
    hist = result.pop("_history")
    agent = sim.agents[result["agent_id"]]
    from the_last_ai.experiments.snapshots import capture_agent_mind

    mind = capture_agent_mind(agent, sim=sim, label="test")
    assert "computational_loss" in mind
    payload = build_export_payload(
        hist, mind=mind, active_ids=set(sim.active_agent_ids())
    )
    assert "loss" in payload
    assert payload["loss"]["disclaimer"]
    html = render_biography_html(payload)
    assert "LOSS & DISAPPEARANCE" in html
    assert "In plain language" in html


def test_deterministic_seed():
    a = run_loss_condition(
        seed=99, label="moderate", formation_ticks=20, post_ticks=20, disappear=True
    )
    b = run_loss_condition(
        seed=99, label="moderate", formation_ticks=20, post_ticks=20, disappear=True
    )
    for r in (a, b):
        r.pop("_sim", None)
        r.pop("_history", None)
    assert a["peaks"]["peak_social_loss"] == b["peaks"]["peak_social_loss"]
    assert a["relationship_before"] == b["relationship_before"]


def test_experiment_9_writes_files(tmp_path):
    out = run_experiment_9(seeds=[0], output_dir=tmp_path)
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "technical_seed0.json").exists()
    assert (tmp_path / "interpreted_seed0.json").exists()
    tech = json.loads((tmp_path / "technical_seed0.json").read_text())
    assert tech["experiment"] == "computational_loss"
    assert len(tech["conditions"]) == 5
    assert out["report"]["aggregates"]["n_seeds"] == 1


def test_significance_uses_relationship_inputs():
    low = compute_entity_significance(
        strength=0.1,
        trust=0.4,
        interaction_count=1,
        memory_strength=0.1,
        familiarity=0.1,
        communicate_count=0,
        predicted_utility=0.0,
    )
    high = compute_entity_significance(
        strength=0.9,
        trust=0.85,
        interaction_count=50,
        memory_strength=0.8,
        familiarity=0.9,
        communicate_count=20,
        predicted_utility=0.7,
    )
    assert high > low


def test_loss_tracker_roundtrip():
    t = LossTracker()
    t.aggregate_social_loss = 0.42
    data = t.to_dict()
    t2 = LossTracker.from_dict(data)
    assert t2.aggregate_social_loss == 0.42
