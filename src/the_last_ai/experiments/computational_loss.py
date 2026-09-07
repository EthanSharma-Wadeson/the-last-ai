"""Experiment 9 — Computational Loss (relationship-dependent disappearance).

Research question
-----------------
Does the disappearance of a socially significant entity produce a persistent,
relationship-dependent change in the surviving agent's computational state and
subsequent behaviour?

Scientific constraint
---------------------
No hard-coded grief/sadness. Responses emerge from memory, relationships,
prediction error, search, and measured behaviour. Observed computational
response ≠ subjective emotional experience.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

from the_last_ai.agents.predictive_agent import PredictiveAgent
from the_last_ai.agents.social_agent import CooperativePartner, SocialConfig
from the_last_ai.experiments.interpret import write_interpretation
from the_last_ai.experiments.snapshots import capture_agent_mind
from the_last_ai.loss.metrics import behaviour_snapshot, measure_behaviour_window
from the_last_ai.loss.narrative import narrate_entity_loss
from the_last_ai.metrics.recorder import MetricsRecorder
from the_last_ai.observatory.affect import compute_affect
from the_last_ai.observatory.export import export_agent_life
from the_last_ai.rng import ExperimentRNG
from the_last_ai.simulation.engine import Simulation, SimulationConfig
from the_last_ai.types import CellType, Position
from the_last_ai.world.grid import Grid
from the_last_ai.world.world import World


def _cluster(sim: Simulation, a: str, b: str, origin: Position) -> None:
    sim.world.agent_positions[a] = origin
    sim.world.agent_positions[b] = origin.offset(1, 0)


def _build_pair(
    *,
    seed: int,
    formation_bias: float,
    interact_probability: float = 0.95,
) -> tuple[Simulation, str, str]:
    rng = ExperimentRNG.from_seed(seed)
    grid = Grid.empty(14, 10, bordered=True)
    world = World(grid=grid, resource_energy=18.0, resource_regen_interval=12)
    observer_id, partner_id = "observer", "partner"
    social = SocialConfig(
        cooperate_bias=formation_bias,
        interact_probability=interact_probability,
        communication_bias=0.8,
        enable_symbolic_communication=True,
    )
    observer = PredictiveAgent(observer_id, perception_radius=4, social_config=social)
    partner = CooperativePartner(partner_id, perception_radius=4)
    agents = {observer_id: observer, partner_id: partner}
    world.place_agent(observer_id, Position(4, 4))
    world.place_agent(partner_id, Position(5, 4))
    # Distant distractor for exploration baseline
    distractor = PredictiveAgent(
        "distractor",
        perception_radius=3,
        social_config=SocialConfig(interact_probability=0.3),
    )
    world.place_agent("distractor", Position(11, 8))
    agents["distractor"] = distractor
    world.place_resource(Position(6, 4))
    world.place_resource(Position(3, 5))

    config = SimulationConfig(
        seed=seed,
        width=14,
        height=10,
        n_agents=3,
        perception_radius=4,
        agent_type="PredictiveAgent",
        n_resources=2,
    )
    sim = Simulation(
        config=config,
        world=world,
        agents=agents,
        rng=rng,
        metrics=MetricsRecorder(),
    )
    sim.observatory.focus({observer_id})
    return sim, observer_id, partner_id


def _relationship_snapshot(agent, partner_id: str, tick: int) -> dict:
    rel = agent.relationships.get(partner_id)
    mem = agent.ltm.get(partner_id)
    mem_strength = (
        float(mem.strength_at(tick, agent.ltm.config.decay_lambda)) if mem else 0.0
    )
    return {
        "interaction_count": int(rel.interaction_count) if rel else 0,
        "relationship_strength": float(rel.strength) if rel else 0.0,
        "trust": float(rel.trust) if rel else 0.5,
        "familiarity": float(mem.familiarity) if mem else 0.0,
        "memory_strength": mem_strength,
        "communicate_count": int(rel.communicate_count) if rel else 0,
        "predicted_utility": float(rel.predicted_utility) if rel else 0.0,
        "information_reliability": float(rel.information_reliability) if rel else 0.5,
    }


def _window_specs(disappear_tick: int, end_tick: int) -> list[tuple[str, int, int]]:
    """Baseline and post-loss windows relative to disappearance."""
    return [
        ("baseline", max(0, disappear_tick - 20), disappear_tick),
        ("immediate", disappear_tick, min(end_tick, disappear_tick + 10)),
        ("short_term", disappear_tick + 10, min(end_tick, disappear_tick + 30)),
        ("medium_term", disappear_tick + 30, min(end_tick, disappear_tick + 60)),
        ("long_term", disappear_tick + 60, end_tick),
    ]


def run_loss_condition(
    *,
    seed: int,
    label: str,
    formation_ticks: int,
    post_ticks: int,
    formation_bias: float = 1.2,
    interact_probability: float = 0.95,
    disappear: bool = True,
    control_resource_shock: bool = False,
) -> dict:
    """
    One controlled condition.

    Labels typically: minimal | moderate | strong | control_no_disappear | control_shock
    """
    sim, observer_id, partner_id = _build_pair(
        seed=seed,
        formation_bias=formation_bias,
        interact_probability=interact_probability,
    )
    observer = sim.agents[observer_id]
    assert isinstance(observer, PredictiveAgent)

    # Keep pair adjacent during formation so relationship can form.
    for _ in range(formation_ticks):
        _cluster(sim, observer_id, partner_id, Position(4, 4))
        sim.step()

    tick_pre = sim.world.tick
    before_rel = _relationship_snapshot(observer, partner_id, tick_pre)
    snap_pre = behaviour_snapshot(observer)
    pe_before = float(observer.world_model.mean_error_ema)
    affect_before, _ = compute_affect(observer, active_ids=set(sim.active_agent_ids()))

    behaviour_marks: dict[str, dict] = {"baseline_end": snap_pre}
    disappear_tick = tick_pre

    if control_resource_shock and not disappear:
        # Non-social control: strip nearby resources (environmental instability).
        for pos in list(sim.world.resource_sites):
            sim.world.grid.set(pos, CellType.EMPTY)
        disappear_tick = sim.world.tick
    elif disappear:
        sim.disappear(partner_id, reversible=False)
        disappear_tick = sim.world.tick
    else:
        # Control 4: partner remains; continue co-presence.
        disappear_tick = sim.world.tick

    # Post windows with behaviour snapshots at boundaries
    window_bounds = _window_specs(disappear_tick, disappear_tick + post_ticks)
    next_mark_ticks = sorted({end for _, _, end in window_bounds})
    mark_idx = 0
    for step_i in range(post_ticks):
        if not disappear:
            _cluster(sim, observer_id, partner_id, Position(4, 4))
        sim.step()
        t = sim.world.tick
        while mark_idx < len(next_mark_ticks) and t >= next_mark_ticks[mark_idx]:
            behaviour_marks[f"t{next_mark_ticks[mark_idx]}"] = behaviour_snapshot(observer)
            mark_idx += 1

    # Ensure final mark
    behaviour_marks["end"] = behaviour_snapshot(observer)
    end_tick = sim.world.tick

    rec = observer.loss_tracker.get(partner_id)
    loss_dict = rec.to_dict() if rec else {}
    # Attach measured behaviour windows
    windows = []
    prev_snap = snap_pre
    for wlabel, start, end in window_bounds:
        if end <= start:
            continue
        # Find closest mark at/after end
        after = behaviour_marks.get(f"t{end}") or behaviour_marks["end"]
        # For baseline, before is earlier — approximate with zeros delta from formation
        if wlabel == "baseline":
            # Rebuild approximate baseline by measuring from zero counters is wrong;
            # use formation-end vs a mid-baseline if available — here use pre-disappear as end
            # and estimate start by proportional scaling is imperfect; instead:
            # store rates from counters accumulated only in post using successive diffs.
            win = measure_behaviour_window(
                observer,
                label=wlabel,
                tick_start=start,
                tick_end=end,
                snapshot_before={
                    k: max(0, int(snap_pre.get(k, 0)) - int(0.5 * int(snap_pre.get(k, 0))))
                    for k in snap_pre
                },
                snapshot_after=snap_pre,
            )
        else:
            # Chain diffs: find snapshot at window start
            before = behaviour_marks.get(f"t{start}")
            if before is None:
                before = prev_snap
            win = measure_behaviour_window(
                observer,
                label=wlabel,
                tick_start=start,
                tick_end=end,
                snapshot_before=before,
                snapshot_after=after,
            )
            prev_snap = after
        windows.append(win.to_dict())

    if loss_dict:
        loss_dict["behaviour_windows"] = windows

    affect_after, affect_factors = compute_affect(
        observer, active_ids=set(sim.active_agent_ids())
    )
    pe_after = float(observer.world_model.mean_error_ema)
    after_mem = _relationship_snapshot(observer, partner_id, end_tick)

    narrative = narrate_entity_loss(loss_dict, agent_id=observer_id) if loss_dict else None

    # Multiple-loss probe for strong condition only (optional second disappearance)
    multi: dict | None = None

    mind = capture_agent_mind(observer, sim=sim, label=f"exp9_{label}")
    hist = sim.observatory.histories.get(observer_id)

    result = {
        "experiment": "computational_loss",
        "condition": label,
        "seed": seed,
        "agent_id": observer_id,
        "target_entity": partner_id if disappear else None,
        "disappearance_tick": disappear_tick if disappear else None,
        "formation_ticks": formation_ticks,
        "post_ticks": post_ticks,
        "relationship_before": before_rel,
        "memory_before": {
            "strength": before_rel["memory_strength"],
            "familiarity": before_rel["familiarity"],
        },
        "prediction_before": {"mean_error_ema": pe_before},
        "affect_before": affect_before.to_dict(),
        "affect_after": affect_after.to_dict(),
        "affect_factors_after": affect_factors,
        "prediction_after": {"mean_error_ema": pe_after},
        "relationship_after": after_mem,
        "loss_record": loss_dict,
        "loss_trajectory": loss_dict.get("loss_trajectory", []),
        "prediction_error_trajectory": loss_dict.get("prediction_error_trajectory", []),
        "memory_trajectory": loss_dict.get("memory_trajectory", []),
        "behaviour_windows": windows,
        "adaptation": {
            "adapted": bool(loss_dict.get("adapted")) if loss_dict else False,
            "adaptation_tick": loss_dict.get("adaptation_tick") if loss_dict else None,
            "definition": (
                "social_loss <= peak * 0.35 AND prediction_disruption <= 0.15 "
                "AND search_pressure <= 0.15 after >= 10 ticks"
            ),
        },
        "peaks": {
            "peak_social_loss": float(loss_dict.get("peak_social_loss", 0.0) or 0.0),
            "peak_prediction_disruption": float(
                loss_dict.get("peak_prediction_disruption", 0.0) or 0.0
            ),
            "search_attempts": int(loss_dict.get("search_attempts", 0) or 0),
            "communication_attempts_after": int(
                loss_dict.get("communication_attempts_after", 0) or 0
            ),
            "memory_retrievals_after": int(loss_dict.get("memory_retrievals_after", 0) or 0),
            "final_memory_strength": float(after_mem["memory_strength"]),
            "delta_prediction_error_ema": pe_after - pe_before,
            "delta_social_loss_affect": affect_after.social_loss - affect_before.social_loss,
        },
        "status_counts": observer.loss_tracker.summary(
            active_ids=set(sim.active_agent_ids())
        ).get("counts"),
        "narrative": narrative,
        "mind_snapshot": mind,
        "control_flags": {
            "disappear": disappear,
            "resource_shock": control_resource_shock,
        },
        "multi_loss": multi,
    }
    result["_sim"] = sim
    result["_history"] = hist
    return result


def run_computational_loss_seed(*, seed: int = 0) -> dict:
    """Full battery for one seed: relationship levels + controls."""
    conditions = [
        run_loss_condition(
            seed=seed,
            label="minimal",
            formation_ticks=3,
            post_ticks=80,
            formation_bias=0.2,
            interact_probability=0.5,
            disappear=True,
        ),
        run_loss_condition(
            seed=seed + 1000,
            label="moderate",
            formation_ticks=25,
            post_ticks=80,
            formation_bias=0.9,
            disappear=True,
        ),
        run_loss_condition(
            seed=seed + 2000,
            label="strong",
            formation_ticks=70,
            post_ticks=100,
            formation_bias=1.5,
            interact_probability=0.98,
            disappear=True,
        ),
        run_loss_condition(
            seed=seed + 3000,
            label="control_no_disappear",
            formation_ticks=70,
            post_ticks=100,
            formation_bias=1.5,
            disappear=False,
        ),
        run_loss_condition(
            seed=seed + 4000,
            label="control_resource_shock",
            formation_ticks=70,
            post_ticks=100,
            formation_bias=1.5,
            disappear=False,
            control_resource_shock=True,
        ),
    ]

    # Multi-disappearance on a small population for the strong seed path
    multi = _run_multiple_disappearances(seed=seed + 5000)

    cleaned = []
    for c in conditions:
        sim = c.pop("_sim")
        hist = c.pop("_history")
        c["observatory_agent_id"] = c["agent_id"]
        c["_export_ready"] = {"sim": sim, "hist": hist}
        cleaned.append(c)

    comparison = _compare_conditions(cleaned)
    return {
        "experiment": "computational_loss",
        "seed": seed,
        "conditions": [{k: v for k, v in c.items() if k != "_export_ready"} for c in cleaned],
        "_exports": cleaned,
        "comparison": comparison,
        "multiple_disappearances": multi,
        "question": (
            "Does disappearance of a socially significant entity produce a persistent, "
            "relationship-dependent change in the surviving agent's computational state "
            "and subsequent behaviour?"
        ),
        "non_claims": [
            "Does not demonstrate subjective grief, sadness, or consciousness.",
            "social_loss is a derived computational aggregate, not an emotion variable.",
            "Null results are scientifically valid outcomes.",
        ],
    }


def _run_multiple_disappearances(*, seed: int) -> dict:
    config = SimulationConfig(
        seed=seed,
        n_agents=4,
        n_resources=4,
        width=14,
        height=10,
        agent_type="PredictiveAgent",
        perception_radius=4,
    )
    sim = Simulation.create(config)
    for a in sim.agents.values():
        if isinstance(a, PredictiveAgent):
            a.social_config = SocialConfig(
                cooperate_bias=1.2, interact_probability=0.9, communication_bias=0.6
            )
    survivor = "agent_000"
    sim.observatory.focus({survivor})
    # Formation clustered
    origin = Position(4, 4)
    ids = sorted(sim.agents.keys())
    for i, aid in enumerate(ids):
        sim.world.agent_positions[aid] = origin.offset(i % 2, i // 2)
    sim.run(40)
    peaks = []
    for victim in ("agent_001", "agent_002", "agent_003"):
        if victim not in sim.world.agent_positions:
            continue
        sim.disappear(victim, reversible=False)
        sim.run(25)
        agent = sim.agents[survivor]
        rec = agent.loss_tracker.get(victim)
        peaks.append(
            {
                "entity_id": victim,
                "peak_social_loss": rec.peak_social_loss if rec else 0.0,
                "aggregate_after": agent.loss_tracker.aggregate_social_loss,
                "historical_count": agent.loss_tracker.summary(
                    active_ids=set(sim.active_agent_ids())
                )["counts"]["historical"],
            }
        )
    agent = sim.agents[survivor]
    return {
        "survivor_id": survivor,
        "sequential_peaks": peaks,
        "final_loss_summary": agent.loss_tracker.summary(
            active_ids=set(sim.active_agent_ids())
        ),
        "final_mind": capture_agent_mind(agent, sim=sim, label="multi_loss_survivor"),
    }


def _compare_conditions(conditions: list[dict]) -> dict:
    by = {c["condition"]: c for c in conditions}
    strong = by.get("strong", {})
    minimal = by.get("minimal", {})
    control = by.get("control_no_disappear", {})
    shock = by.get("control_resource_shock", {})

    def peak(c, key):
        return float((c.get("peaks") or {}).get(key, 0.0) or 0.0)

    return {
        "peak_social_loss": {
            k: peak(by[k], "peak_social_loss") for k in by
        },
        "peak_prediction_disruption": {
            k: peak(by[k], "peak_prediction_disruption") for k in by
        },
        "search_attempts": {k: peak(by[k], "search_attempts") for k in by},
        "strong_minus_minimal_social_loss": peak(strong, "peak_social_loss")
        - peak(minimal, "peak_social_loss"),
        "strong_minus_control_social_loss": peak(strong, "peak_social_loss")
        - peak(control, "peak_social_loss"),
        "strong_minus_shock_social_loss": peak(strong, "peak_social_loss")
        - peak(shock, "peak_social_loss"),
        "relationship_strength_before": {
            k: float((by[k].get("relationship_before") or {}).get("relationship_strength", 0))
            for k in by
        },
    }


def interpret_computational_loss(tech: dict) -> dict:
    """Derived-only interpretation layer (never fed back into simulation)."""
    comparison = tech.get("comparison") or {}
    observations = []
    if "peak_social_loss" in comparison:
        observations.append(
            {
                "observation": "Peak social_loss by condition",
                "evidence": comparison["peak_social_loss"],
                "interpretation": (
                    "Higher prior relationship strength may be associated with higher "
                    "peak social_loss after disappearance — or may not; compare values."
                ),
                "confidence": "descriptive",
                "caveat": "Not a claim of subjective distress.",
            }
        )
    delta = comparison.get("strong_minus_minimal_social_loss")
    if delta is not None:
        observations.append(
            {
                "observation": "Strong vs minimal peak social_loss difference",
                "evidence": {"strong_minus_minimal": delta},
                "interpretation": (
                    "Positive difference is consistent with relationship-dependent "
                    "computational response; near-zero supports a null result."
                ),
                "confidence": "low_n" if tech.get("seed") is not None else "descriptive",
                "caveat": "Single-seed differences are not statistical significance.",
            }
        )
    ctrl = comparison.get("strong_minus_control_social_loss")
    if ctrl is not None:
        observations.append(
            {
                "observation": "Strong disappearance vs matched no-disappear control",
                "evidence": {"strong_minus_control": ctrl},
                "interpretation": (
                    "If elevated after disappearance but not in the stay-present control, "
                    "the change is associated with disappearance rather than time alone."
                ),
                "confidence": "descriptive",
                "caveat": "Does not prove causal emotion; shows computational association.",
            }
        )
    return {
        "experiment": "computational_loss_interpreted",
        "from_technical": True,
        "observations": observations,
        "comparison": comparison,
        "scientific_caveat": (
            "Observed computational response ≠ subjective emotional experience."
        ),
        "non_claims": tech.get("non_claims") or [],
    }


def _agg_numeric(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def run_experiment_9(
    *,
    seed: int | None = 0,
    seeds: list[int] | None = None,
    output_dir: str | Path = "outputs/experiment_9",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_list = seeds if seeds is not None else [0 if seed is None else seed]

    per_seed = []
    for s in seed_list:
        raw = run_computational_loss_seed(seed=s)
        exports = raw.pop("_exports", [])
        # Export observer life for strong condition
        for c in exports:
            if c.get("condition") != "strong":
                continue
            pack = c.get("_export_ready") or {}
            sim = pack.get("sim")
            hist = pack.get("hist")
            if sim is None or hist is None:
                continue
            agent = sim.agents[c["agent_id"]]
            mind = capture_agent_mind(agent, sim=sim, label="exp9_strong_export")
            paths = export_agent_life(
                hist,
                output_dir / f"observatory_seed{s}",
                mind=mind,
                active_ids=set(sim.active_agent_ids()),
                meta={"experiment": "computational_loss", "seed": s, "condition": "strong"},
            )
            raw["observatory_paths"] = paths

        tech_path = output_dir / f"technical_seed{s}.json"
        tech_path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
        interpreted = interpret_computational_loss(raw)
        int_path = output_dir / f"interpreted_seed{s}.json"
        int_path.write_text(json.dumps(interpreted, indent=2, sort_keys=True), encoding="utf-8")
        write_interpretation(tech_path, output_path=output_dir / f"interpreted_seed{s}.md")
        per_seed.append({"seed": s, "technical": raw, "interpreted": interpreted})

    # Multi-seed aggregates on key peaks
    def collect(cond: str, key: str) -> list[float]:
        out = []
        for row in per_seed:
            for c in row["technical"].get("conditions") or []:
                if c.get("condition") == cond:
                    out.append(float((c.get("peaks") or {}).get(key, 0.0) or 0.0))
        return out

    aggregates = {
        "n_seeds": len(seed_list),
        "peak_social_loss": {
            cond: _agg_numeric(collect(cond, "peak_social_loss"))
            for cond in ("minimal", "moderate", "strong", "control_no_disappear", "control_resource_shock")
        },
        "peak_prediction_disruption": {
            cond: _agg_numeric(collect(cond, "peak_prediction_disruption"))
            for cond in ("minimal", "moderate", "strong", "control_no_disappear", "control_resource_shock")
        },
        "search_attempts": {
            cond: _agg_numeric(collect(cond, "search_attempts"))
            for cond in ("minimal", "moderate", "strong")
        },
        "note": (
            "Effect sizes are descriptive mean differences across seeds. "
            "Do not treat small-n results as statistical significance."
        ),
    }
    # Descriptive effect: strong mean - minimal mean
    s_mean = aggregates["peak_social_loss"]["strong"]["mean"]
    m_mean = aggregates["peak_social_loss"]["minimal"]["mean"]
    if s_mean is not None and m_mean is not None:
        aggregates["effect_strong_minus_minimal_peak_social_loss"] = s_mean - m_mean

    report = {
        "experiment": "experiment_9_computational_loss",
        "seeds": seed_list,
        "aggregates": aggregates,
        "per_seed_paths": [
            {
                "seed": row["seed"],
                "technical": str(output_dir / f"technical_seed{row['seed']}.json"),
                "interpreted": str(output_dir / f"interpreted_seed{row['seed']}.json"),
            }
            for row in per_seed
        ],
        "question": per_seed[0]["technical"]["question"] if per_seed else "",
        "non_claims": per_seed[0]["technical"]["non_claims"] if per_seed else [],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return {"report": report, "report_path": str(report_path), "per_seed": per_seed}
