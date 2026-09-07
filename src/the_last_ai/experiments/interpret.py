"""Turn technical experiment / mind JSON into a human-readable narrative."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from the_last_ai.types import JSONDict


def _f(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.0f}%"


def _event_title(name: str) -> str:
    if name == "start":
        return "Beginning — before anyone has been known"
    if name == "after_initial_interaction":
        return "After living among the population"
    if name == "final_survivor":
        return "Alone — the final survivor"
    if name.startswith("collapse_to_"):
        n = name.removeprefix("collapse_to_")
        return f"Disappearance event — population falls to {n}"
    if name == "inexperienced_control":
        return "Inexperienced control (same architecture, no social history)"
    return name.replace("_", " ")


def _surprise_phrase(error: float | None, *, tick: int | None = None) -> str:
    if error is None:
        return "prediction surprise is unknown"
    # Fresh agents start near 0.5 EMA before any scored predictions settle.
    if tick == 0 or (error >= 0.45 and tick is not None and tick < 5):
        return "prediction model not yet calibrated"
    if error < 0.05:
        return "the world feels mostly predictable"
    if error < 0.15:
        return "mild ongoing surprise"
    if error < 0.30:
        return "noticeable prediction failure — the model is being contradicted"
    return "severe surprise — expectations and observations clash hard"


def _strength_phrase(strength: float) -> str:
    if strength >= 0.85:
        return "very strong"
    if strength >= 0.6:
        return "strong"
    if strength >= 0.35:
        return "moderate"
    if strength >= 0.15:
        return "fading"
    return "nearly forgotten"


def _trust_phrase(trust: float) -> str:
    if trust >= 0.7:
        return "high trust"
    if trust >= 0.55:
        return "cautious trust"
    if trust >= 0.4:
        return "neutral / uncertain trust"
    return "low trust"


def _resolve_path(raw: str | Path, *, base_dir: Path | None = None) -> Path | None:
    """Resolve experiment-relative paths written from different working directories."""
    path = Path(raw)
    candidates = [path]
    if base_dir is not None:
        candidates.extend(
            [
                base_dir / path,
                base_dir / path.name,
                base_dir / "events" / path.name,
            ]
        )
        # report lives next to events_seedN/
        if path.name:
            for child in base_dir.glob("events_seed*/" + path.name):
                candidates.append(child)
    candidates.append(Path.cwd() / path)
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            key = candidate.resolve()
        except OSError:
            key = candidate
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def detect_kind(data: JSONDict) -> str:
    if data.get("experiment") in {
        "computational_loss",
        "experiment_9_computational_loss",
        "computational_loss_interpreted",
    } or (
        "conditions" in data
        and "comparison" in data
        and data.get("question", "").startswith("Does disappearance")
    ):
        return "computational_loss"
    if data.get("experiment") == "experiment_8_communication" or (
        "aggregates_enabled" in data and "per_seed_pass" in data
    ):
        return "communication_report"
    if "tests" in data and "A_basic_communication" in (data.get("tests") or {}):
        return "communication_seed"
    if "during_absence" in data and ("at_removal" in data or "after_restoration" in data):
        return "restoration"
    if "aggregates" in data and ("by_architecture" in data or "seeds" in data):
        return "architecture_compare"
    if "events" in data and ("question" in data or "schedule" in data or "survivor_id" in data):
        return "last_ai_report"
    if "agent_id" in data and (
        "entity_memory" in data
        or "entity_representations" in data
        or "world_model" in data
        or "communication" in data
    ):
        return "mind"
    return "generic"


def interpret_mind(mind: JSONDict, *, heading: str | None = None) -> str:
    """Narrate one agent-mind snapshot."""
    lines: list[str] = []
    label = heading or _event_title(str(mind.get("label") or "mind snapshot"))
    aid = mind.get("agent_id", "?")
    atype = mind.get("agent_type", "agent")
    tick = mind.get("tick")
    pop = mind.get("population")
    pos = mind.get("position")

    lines.append(f"# {label}")
    lines.append("")
    intro = f"**{aid}** ({atype})"
    bits = []
    if tick is not None:
        bits.append(f"tick {tick}")
    if pop is not None:
        bits.append(f"population {pop}")
    if pos is not None:
        bits.append(f"at {pos}")
    if bits:
        intro += " — " + ", ".join(bits)
    lines.append(intro + ".")
    lines.append("")

    energy = mind.get("energy")
    age = mind.get("age")
    last = mind.get("last_action")
    lines.append("## How it feels right now")
    feel = []
    if energy is not None:
        feel.append(f"energy {_f(energy, 1)}")
    if age is not None:
        feel.append(f"age {age}")
    if last:
        feel.append(f"last action `{last}`")
    pe = mind.get("prediction_error_ema")
    if pe is not None:
        feel.append(
            f"prediction-error EMA {_f(pe, 3)} "
            f"({_surprise_phrase(float(pe), tick=tick if isinstance(tick, int) else None)})"
        )
    lines.append("- " + "; ".join(feel) if feel else "- No status fields present.")
    lines.append("")

    # Memories of others
    reps = mind.get("entity_representations") or {}
    strengths = mind.get("entity_strengths") or {}
    em = mind.get("entity_memory") or {}
    lines.append("## Who it still remembers")
    if not reps and not em.get("entity_count"):
        lines.append("Nobody. Long-term entity memory is empty.")
        lines.append("")
    else:
        count = em.get("entity_count", len(reps))
        lines.append(
            f"It carries internal representations of **{count}** other agents "
            "(a local neighbourhood, not the whole vanished population)."
        )
        lines.append("")
        if not reps and em.get("familiarities"):
            # Summary-only fallback when full reps are absent.
            for eid, fam in sorted(
                (em.get("familiarities") or {}).items(),
                key=lambda kv: float((em.get("strengths") or {}).get(kv[0], 0.0)),
                reverse=True,
            ):
                strength = float((em.get("strengths") or {}).get(eid, 0.0))
                lines.append(
                    f"- **{eid}** — {_strength_phrase(strength)} memory "
                    f"(strength {_f(strength)}, familiarity {_pct(fam)})."
                )
        else:
            ordered = sorted(
                reps.items(),
                key=lambda kv: float(strengths.get(kv[0], kv[1].get("strength", 0.0))),
                reverse=True,
            )
            for eid, rep in ordered:
                strength = float(strengths.get(eid, rep.get("strength", 0.0)))
                fam = float(rep.get("familiarity", 0.0))
                last_seen = rep.get("last_seen")
                expected = rep.get("expected_location")
                interactions = rep.get("interaction_count", 0)
                unc = float(rep.get("uncertainty", 1.0))
                lines.append(
                    f"- **{eid}** — {_strength_phrase(strength)} memory "
                    f"(strength {_f(strength)}, familiarity {_pct(fam)}, "
                    f"uncertainty {_pct(unc)}). "
                    f"Last seen at tick {last_seen}; expects them near {expected}; "
                    f"{interactions} remembered interactions."
                )
        lines.append("")

    # Relationships
    rel_block = mind.get("relationships") or {}
    rels = rel_block.get("relationships") or {}
    lines.append("## Bonds and trust")
    if not rels:
        lines.append("No direct relationships are stored.")
        lines.append("")
    else:
        lines.append(
            f"**{len(rels)}** direct bond(s); mean strength {_f(rel_block.get('mean_strength'))}, "
            f"mean trust {_f(rel_block.get('mean_trust'))}."
        )
        last_partner = rel_block.get("last_interaction_partner")
        if last_partner:
            lines.append(f"Most recent partner: **{last_partner}**.")
        lines.append("")
        for eid, rel in sorted(
            rels.items(), key=lambda kv: float(kv[1].get("strength", 0.0)), reverse=True
        ):
            lines.append(
                f"- **{eid}** — {_trust_phrase(float(rel.get('trust', 0.5)))} "
                f"(trust {_f(rel.get('trust'))}, strength {_f(rel.get('strength'))}). "
                f"{rel.get('interaction_count', 0)} interactions "
                f"({rel.get('positive_interactions', 0)} positive / "
                f"{rel.get('negative_interactions', 0)} negative); "
                f"cooperate {rel.get('cooperate_count', 0)}, "
                f"compete {rel.get('compete_count', 0)}, "
                f"share {rel.get('share_count', 0)}, "
                f"communicate {rel.get('communicate_count', 0)}."
            )
        lines.append("")

    # World model
    wm = mind.get("world_model") or {}
    lines.append("## What it still predicts about the world")
    if not wm:
        lines.append("No world-model summary in this snapshot.")
        lines.append("")
    else:
        tracked = wm.get("tracked_entities", 0)
        lines.append(
            f"Tracks **{tracked}** entities in its dynamics model "
            f"(mean error EMA {_f(wm.get('mean_error_ema'), 3)}; "
            f"{wm.get('total_scored', 0)} predictions scored)."
        )
        recent = wm.get("recent_error_by_kind") or {}
        if recent:
            parts = [f"{k.replace('_', ' ')} {_f(v, 3)}" for k, v in sorted(recent.items())]
            lines.append("Recent surprise by kind: " + "; ".join(parts) + ".")
        models = wm.get("entity_models") or {}
        if models:
            lines.append("")
            lines.append("Internal forecasts for remembered others:")
            for eid, model in sorted(
                models.items(),
                key=lambda kv: float(kv[1].get("confidence", 0.0)),
                reverse=True,
            ):
                lines.append(
                    f"- **{eid}** — expects presence near {model.get('last_position')}, "
                    f"likely move `{model.get('likely_direction')}`, "
                    f"confidence {_pct(model.get('confidence'))} "
                    f"({model.get('observations', 0)} observations)."
                )
        lines.append("")

    # Behaviour
    beh = mind.get("behaviour") or {}
    lines.append("## How it has been acting")
    if not beh:
        lines.append("No behaviour summary.")
        lines.append("")
    else:
        social = beh.get("social_action_counts") or {}
        social_bits = ", ".join(f"{k} {v}" for k, v in sorted(social.items()) if v)
        lines.append(
            f"- Explored {beh.get('unique_cells_visited', 0)} unique cells "
            f"({beh.get('total_visits', 0)} visits)."
        )
        lines.append(
            f"- Sought remembered agents {beh.get('seek_attempts', 0)} times; "
            f"approach {beh.get('approach_actions', 0)}, avoid {beh.get('avoid_actions', 0)}."
        )
        if social_bits:
            lines.append(f"- Social repertoire: {social_bits}.")
        lines.append("")

    # Values
    lv = mind.get("learned_values") or {}
    if lv:
        lines.append("## Learned values")
        lines.append(
            f"Tabular policy covers {lv.get('q_states', 0)} states "
            f"({lv.get('update_count', 0)} updates); "
            f"exploration ε = {_f(lv.get('epsilon'), 3)}."
        )
        lines.append("")

    # Social graph caveat
    graph = mind.get("social_graph") or {}
    if graph:
        edges = graph.get("edge_count", 0)
        own = [
            e
            for e in (graph.get("edges") or {})
            if str(e).startswith(f"{aid}->") or str(e).endswith(f"->{aid}")
        ]
        lines.append("## Social graph residue")
        lines.append(
            f"The recorded world social graph still lists **{edges}** edges "
            f"(historical interactions among many agents). "
            f"**{len(own)}** of those touch {aid} directly. "
            "Graph size is not the same as how many people this mind personally knows."
        )
        lines.append("")

    # Communication
    comm = mind.get("communication") or {}
    if comm:
        lines.append("## Communication residue")
        lines.append(
            f"Symbolic communication enabled: **{comm.get('enabled')}**. "
            f"Sent {comm.get('messages_sent', 0)}, received {comm.get('messages_received', 0)}; "
            f"verifications {comm.get('verification_count', 0)} "
            f"(successful {comm.get('successful_verification_count', 0)}); "
            f"investigation actions {comm.get('investigation_actions', 0)}."
        )
        mem = comm.get("memory") or {}
        lines.append(
            f"Communication memory holds **{mem.get('count', 0)}** records "
            f"({mem.get('historical_count', 0)} historical / "
            f"{mem.get('active_count', 0)} with still-present counterparts). "
            "Historical records are retained after disappearance; they are not live agents."
        )
        freqs = mem.get("token_frequencies") or {}
        if freqs:
            top = sorted(freqs.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
            lines.append(
                "Frequent tokens: "
                + ", ".join(f"`{t}`×{c}" for t, c in top)
                + "."
            )
        rels = comm.get("communication_relationships") or {}
        if rels:
            lines.append("")
            lines.append("Information-reliability links:")
            for oid, meta in sorted(rels.items()):
                status = "active" if meta.get("active_counterpart") else "historical"
                lines.append(
                    f"- **{oid}** ({status}) — reliability "
                    f"{_f(meta.get('information_reliability'))}; "
                    f"success {meta.get('successful_messages', 0)} / "
                    f"fail {meta.get('failed_messages', 0)}; "
                    f"communicate_count {meta.get('communicate_count', 0)}."
                )
        lines.append("")
        lines.append(
            "_This describes symbol usage and information exchange, "
            "not linguistic understanding or subjective experience._"
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _delta_sentence(delta: JSONDict) -> str:
    parts = []
    mem = delta.get("entity_memory_count", 0)
    rel = delta.get("relationship_count", 0)
    track = delta.get("tracked_entities", 0)
    pe = delta.get("prediction_error_ema", 0.0)
    edges = delta.get("social_graph_edges", 0)
    if mem:
        parts.append(f"entity memories {mem:+d}")
    if rel:
        parts.append(f"relationships {rel:+d}")
    if track:
        parts.append(f"tracked entities {track:+d}")
    if edges:
        parts.append(f"graph edges {edges:+d}")
    if abs(float(pe or 0.0)) >= 0.02:
        direction = "rose" if float(pe) > 0 else "fell"
        parts.append(f"prediction surprise {direction} by {_f(abs(float(pe)), 3)}")
    if not parts:
        return "Internal structure barely moved — same memories and bonds, different world outside."
    return "Changed: " + "; ".join(parts) + "."


def interpret_last_ai_report(
    report: JSONDict,
    *,
    load_event_minds: bool = True,
    base_dir: Path | None = None,
) -> str:
    """Narrate the centrepiece collapse report (+ optional deep dive on final mind)."""
    lines: list[str] = []
    lines.append("# The Last AI — human reading")
    lines.append("")
    q = report.get("question")
    if q:
        lines.append(f"> {q}")
        lines.append("")
    lines.append(
        f"Seed **{report.get('seed')}**. Started with **{report.get('initial_agents')}** agents; "
        f"collapse schedule `{report.get('schedule')}`. "
        f"Designated survivor: **{report.get('survivor_id')}** "
        f"(kept on purpose — not natural selection)."
    )
    lines.append("")

    events = report.get("events") or []
    deltas = {(d.get("to"), d.get("from")): d.get("delta", {}) for d in report.get("trajectory_deltas") or []}
    # also index by `to` alone
    delta_to = {d.get("to"): d.get("delta", {}) for d in report.get("trajectory_deltas") or []}

    lines.append("## Timeline of what remained inside the survivor")
    lines.append("")
    for ev in events:
        name = ev.get("event", "?")
        tick = ev.get("tick")
        pop = ev.get("population")
        summary = ev.get("summary") or {}
        lines.append(f"### {_event_title(str(name))}")
        lines.append(f"Tick {tick}, population {pop}.")
        lines.append(
            f"- Remembers {summary.get('entity_memory_count', 0)} entities; "
            f"{summary.get('relationships', 0)} direct relationships; "
            f"tracks {summary.get('tracked_entities', 0)} in the world model."
        )
        pe = summary.get("prediction_error_ema")
        if pe is not None:
            lines.append(
                f"- Prediction-error EMA {_f(pe, 3)} — "
                f"{_surprise_phrase(float(pe), tick=tick if isinstance(tick, int) else None)}."
            )
        d = delta_to.get(name)
        if d:
            lines.append(f"- {_delta_sentence(d)}")
        lines.append("")

    cc = report.get("control_comparison")
    if cc:
        lines.append("## Survivor vs inexperienced control")
        lines.append("")
        s = cc.get("survivor") or {}
        c = cc.get("control") or {}
        d = cc.get("delta_survivor_minus_control") or {}
        lines.append(
            "Same architecture, different history. What the collapse left behind that a blank agent lacks:"
        )
        lines.append("")
        lines.append(
            f"| | Survivor | Control | Extra in survivor |\n"
            f"|---|---:|---:|---:|\n"
            f"| Entity memories | {s.get('entity_memory_count', 0)} | {c.get('entity_memory_count', 0)} | "
            f"{d.get('entity_memory_count', 0)} |\n"
            f"| Relationships | {s.get('relationships', 0)} | {c.get('relationships', 0)} | "
            f"{d.get('relationships', 0)} |\n"
            f"| Tracked entities | {s.get('tracked_entities', 0)} | {c.get('tracked_entities', 0)} | "
            f"{d.get('tracked_entities', 0)} |"
        )
        lines.append("")
        if (d.get("entity_memory_count") or 0) > 0 or (d.get("relationships") or 0) > 0:
            lines.append(
                "**Verdict:** the vanished population left measurable structure in the survivor's mind."
            )
        else:
            lines.append(
                "**Verdict:** little structured residue relative to the control — "
                "for this seed, collapse did not encode much."
            )
        lines.append("")

    if load_event_minds:
        final = next((e for e in reversed(events) if e.get("event") == "final_survivor"), None)
        if final and final.get("mind_path"):
            resolved = _resolve_path(final["mind_path"], base_dir=base_dir)
            if resolved is not None:
                mind = json.loads(resolved.read_text(encoding="utf-8"))
                lines.append("---")
                lines.append("")
                lines.append(interpret_mind(mind))
            else:
                lines.append(
                    f"_Final mind file not found at `{final['mind_path']}` — "
                    "run from the repo root or pass an absolute path._"
                )
                lines.append("")

    if report.get("note"):
        lines.append("## Research note")
        lines.append(str(report["note"]))
        lines.append("")

    # silence unused
    _ = deltas
    return "\n".join(lines).rstrip() + "\n"


def interpret_restoration(data: JSONDict) -> str:
    lines: list[str] = []
    lines.append("# Experiment 6 — Restoration (human reading)")
    lines.append("")
    lines.append(
        f"Seed **{data.get('seed')}**. A formed a bond with B "
        f"({data.get('formation_ticks')} ticks), B disappeared "
        f"({data.get('absence_ticks')} ticks), then B returned "
        f"({data.get('restoration_ticks')} ticks)."
    )
    lines.append("")

    rem = data.get("at_removal") or {}
    lines.append("## At disappearance")
    if rem.get("world_model_has_B"):
        lines.append("A's world model still tracked B.")
    pred = rem.get("presence_prediction") or {}
    outcomes = pred.get("predicted_outcomes") or []
    if outcomes:
        o = outcomes[0]
        lines.append(
            f"When asked about absent B, A still predicted presence near "
            f"{o.get('last_position')} with confidence {_pct(o.get('confidence'))}."
        )
    rel = rem.get("relationship") or {}
    if rel:
        lines.append(
            f"Bond with B: strength {_f(rel.get('strength'))}, trust {_f(rel.get('trust'))}, "
            f"{rel.get('interaction_count', 0)} prior interactions."
        )
    lines.append("")

    absn = data.get("during_absence") or {}
    lines.append("## During absence")
    rate = absn.get("residual_expectation_rate")
    lines.append(
        f"Residual expectation that B still exists: **{_pct(rate)}**. "
        f"Memory persisted: **{absn.get('memory_persisted')}**; "
        f"relationship persisted: **{absn.get('relationship_persisted')}**. "
        f"Prediction-error EMA {_f(absn.get('prediction_error_ema'), 3)} "
        f"({_surprise_phrase(float(absn['prediction_error_ema'])) if absn.get('prediction_error_ema') is not None else 'n/a'})."
    )
    if rate is not None and float(rate) >= 0.5:
        lines.append(
            "Interpretation: B's representation outlived B's body — "
            "the internal model kept predicting a ghost."
        )
    lines.append("")

    after = data.get("after_restoration") or {}
    lines.append("## After B returns")
    adapt = after.get("adaptation_ticks_to_90pct_trust")
    recog = after.get("recognition_tick")
    lines.append(
        f"Recognition tick: **{recog}**. "
        f"Ticks to re-adapt trust (≥90% of restored level): **{adapt}**."
    )
    if adapt is not None and int(adapt) <= 2:
        lines.append("A snapped back quickly — the old model of B was still usable.")
    elif adapt is not None:
        lines.append("Re-integration took longer — the absence period had shifted expectations.")
    lines.append("")
    note = data.get("interpretation_note")
    if note:
        lines.append(f"_{note}_")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def interpret_communication_seed(data: JSONDict) -> str:
    lines = [
        "# Experiment 8 — Communication (human reading)",
        "",
        f"Seed **{data.get('seed')}**. Symbolic communication "
        f"**{'enabled' if data.get('enable_communication') else 'disabled'}**.",
        "",
        "## Test battery",
        "",
    ]
    for name, block in (data.get("tests") or {}).items():
        if not isinstance(block, dict):
            continue
        passed = block.get("passed")
        mark = "PASS" if passed else ("n/a" if passed is None else "FAIL")
        detail = {k: v for k, v in block.items() if k != "passed"}
        lines.append(f"- **{name}**: {mark} — `{detail}`")
    metrics = data.get("metrics") or {}
    lines.extend(
        [
            "",
            "## Metrics",
            f"- messages sent/received: {metrics.get('messages_sent')} / {metrics.get('messages_received')}",
            f"- success rate: {_f(metrics.get('communication_success_rate'), 3)}",
            f"- mean information reliability: {_f(metrics.get('mean_information_reliability'), 3)}",
            f"- communication memory count: {metrics.get('communication_memory_count')}",
            f"- behavioural response rate: {_f(metrics.get('behavioural_response_rate'), 3)}",
            "",
            "## Scientific caution",
            "These results describe communication behaviour, symbol usage, and information "
            "reliability — not language understanding, consciousness, or emotion.",
            "",
        ]
    )
    return "\n".join(lines)


def interpret_communication_report(data: JSONDict) -> str:
    lines = [
        "# Experiment 8 — Multi-seed communication report",
        "",
        f"Seeds: `{data.get('seeds')}`.",
        "",
        "## Aggregates (communication enabled)",
        "",
    ]
    for key, block in sorted((data.get("aggregates_enabled") or {}).items()):
        if isinstance(block, dict):
            lines.append(
                f"- {key.replace('_', ' ')}: mean {_f(block.get('mean'), 3)} "
                f"(std {_f(block.get('std'), 3)}, n={block.get('n')})"
            )
    lines.extend(["", "## Control (communication disabled)", ""])
    for key, block in sorted((data.get("aggregates_disabled_control") or {}).items()):
        if isinstance(block, dict):
            lines.append(
                f"- {key.replace('_', ' ')}: mean {_f(block.get('mean'), 3)} "
                f"(std {_f(block.get('std'), 3)}, n={block.get('n')})"
            )
    lines.extend(
        [
            "",
            "## Claims the system does make",
            *[f"- {c}" for c in data.get("claims") or []],
            "",
        ]
    )
    return "\n".join(lines)


def interpret_architecture_compare(data: JSONDict) -> str:
    lines: list[str] = []
    lines.append("# Architecture comparison — human reading")
    lines.append("")
    aggs = data.get("aggregates") or data.get("by_architecture") or {}
    if not aggs and "results" in data:
        lines.append("Raw multi-seed results present; aggregates missing.")
        lines.append("")
        return "\n".join(lines)

    # aggregates may be {arch: {metric: {mean, std, ...}}}
    lines.append("What each architecture tends to retain after collapse (means across seeds):")
    lines.append("")
    for arch, stats in sorted(aggs.items()):
        if not isinstance(stats, dict):
            continue
        lines.append(f"### {arch}")
        for key in (
            "entity_memory_count",
            "relationships",
            "tracked_entities",
            "graph_edges",
            "prediction_error_ema",
        ):
            block = stats.get(key)
            if isinstance(block, dict) and "mean" in block:
                lines.append(
                    f"- {key.replace('_', ' ')}: mean {_f(block['mean'], 3)} "
                    f"(std {_f(block.get('std'), 3)}, n={block.get('n', '?')})"
                )
            elif isinstance(block, (int, float)):
                lines.append(f"- {key.replace('_', ' ')}: {_f(block, 3)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def interpret_generic(data: JSONDict) -> str:
    keys = ", ".join(sorted(data.keys())[:24])
    return (
        "# JSON interpretation\n\n"
        "Unrecognised experiment schema. Top-level keys: "
        f"{keys}.\n\n"
        "Try a mind snapshot (`agent_id` + `entity_memory`) or a Last AI `report_seed*.json`.\n"
    )


def interpret_computational_loss_report(data: JSONDict) -> str:
    """Human narrative for Experiment 9 technical or report JSON."""
    lines = [
        "# Computational Loss Experiment",
        "",
        str(
            data.get("question")
            or "Does disappearance produce relationship-dependent computational change?"
        ),
        "",
        "## Scientific caution",
        "",
        "Observed computational response ≠ subjective emotional experience. "
        "Labels like social_loss are derived state variables, not feelings.",
        "",
    ]
    if "aggregates" in data:
        agg = data["aggregates"]
        lines.append("## Multi-seed aggregates (peak social_loss)")
        lines.append("")
        for cond, stats in (agg.get("peak_social_loss") or {}).items():
            lines.append(
                f"- **{cond}**: mean {_f(stats.get('mean'), 3)} "
                f"(std {_f(stats.get('std'), 3)}, n={stats.get('n')})"
            )
        if "effect_strong_minus_minimal_peak_social_loss" in agg:
            lines.append("")
            lines.append(
                f"Descriptive effect (strong − minimal): "
                f"{_f(agg['effect_strong_minus_minimal_peak_social_loss'], 3)}"
            )
            lines.append(
                "_Not_ a claim of statistical significance unless n is large and pre-registered._"
            )
        lines.append("")
        return "\n".join(lines)

    comparison = data.get("comparison") or {}
    lines.append("## Condition comparison (this seed)")
    lines.append("")
    for key in ("peak_social_loss", "peak_prediction_disruption", "search_attempts"):
        if key in comparison:
            lines.append(f"**{key.replace('_', ' ')}**")
            for cond, val in (comparison[key] or {}).items():
                lines.append(f"- {cond}: {_f(val, 3)}")
            lines.append("")
    for c in data.get("conditions") or []:
        label = c.get("condition")
        peaks = c.get("peaks") or {}
        before = c.get("relationship_before") or {}
        lines.append(f"### Condition: {label}")
        lines.append(
            f"Before disappearance — interactions {before.get('interaction_count', 0)}, "
            f"strength {_f(before.get('relationship_strength'))}, "
            f"trust {_f(before.get('trust'))}, "
            f"memory {_f(before.get('memory_strength'))}."
        )
        if c.get("disappearance_tick") is not None:
            lines.append(f"Disappeared at tick {c.get('disappearance_tick')}.")
        lines.append(
            f"Peaks — social_loss {_f(peaks.get('peak_social_loss'))}, "
            f"prediction disruption {_f(peaks.get('peak_prediction_disruption'))}, "
            f"search {peaks.get('search_attempts', 0)}, "
            f"memory retrievals {peaks.get('memory_retrievals_after', 0)}."
        )
        narr = c.get("narrative") or {}
        if narr.get("level1_human_summary"):
            lines.append("")
            lines.append(narr["level1_human_summary"])
        lines.append("")
    multi = data.get("multiple_disappearances") or {}
    if multi.get("sequential_peaks"):
        lines.append("## Multiple disappearances")
        lines.append("")
        for row in multi["sequential_peaks"]:
            lines.append(
                f"- {row.get('entity_id')}: peak social_loss {_f(row.get('peak_social_loss'))}, "
                f"aggregate after {_f(row.get('aggregate_after'))}"
            )
        lines.append("")
    lines.append("## Non-claims")
    for c in data.get("non_claims") or []:
        lines.append(f"- {c}")
    lines.append("")
    return "\n".join(lines)


def interpret_data(
    data: JSONDict,
    *,
    load_event_minds: bool = True,
    base_dir: Path | None = None,
) -> str:
    kind = detect_kind(data)
    if kind == "computational_loss":
        return interpret_computational_loss_report(data)
    if kind == "mind":
        return interpret_mind(data)
    if kind == "last_ai_report":
        return interpret_last_ai_report(
            data, load_event_minds=load_event_minds, base_dir=base_dir
        )
    if kind == "restoration":
        return interpret_restoration(data)
    if kind == "architecture_compare":
        return interpret_architecture_compare(data)
    if kind == "communication_seed":
        return interpret_communication_seed(data)
    if kind == "communication_report":
        return interpret_communication_report(data)
    return interpret_generic(data)


def interpret_json_file(
    path: str | Path,
    *,
    load_event_minds: bool = True,
) -> str:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    # If this is a wrapper like {"result": {...}} from some runners
    if (
        isinstance(data, dict)
        and "result" in data
        and detect_kind(data) == "generic"
        and isinstance(data["result"], dict)
    ):
        inner_kind = detect_kind(data["result"])
        if inner_kind != "generic":
            data = data["result"]
    return interpret_data(
        data,
        load_event_minds=load_event_minds,
        base_dir=path.parent,
    )


def write_interpretation(
    path: str | Path,
    *,
    output_path: str | Path | None = None,
    load_event_minds: bool = True,
) -> Path:
    path = Path(path)
    text = interpret_json_file(path, load_event_minds=load_event_minds)
    out = Path(output_path) if output_path else path.with_suffix(".md")
    out.write_text(text, encoding="utf-8")
    return out
