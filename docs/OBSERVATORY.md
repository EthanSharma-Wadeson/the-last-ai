# Individual Agent Life Observatory

## Purpose

Make one agent's **computational life history** inspectable from birth to final state.

This is an instrumentation and presentation layer. It does **not** redesign cognition.

It does **not** claim consciousness, subjective emotion, or literal friendship.

## What is recorded

Structured `LifeEvent` objects with tick, type, state_before/after, and context:

- birth, population changes, disappearances
- encounters, resources, actions, decisions
- communication (exact tokens preserved)
- relationships + trust/reliability trajectories
- memory activation / expected-entity absence
- prediction-error spikes
- affective dimension changes **with recorded factors**

## Affective dimensions (derived, not scripted)

```text
valence, activation, social_drive, uncertainty,
security, novelty, prediction_error, social_loss, goal_success
```

Labels such as `withdrawal-like` / `exploratory` are descriptive translations of those numbers.

## Relationship labels (derived)

```text
stranger → acquaintance → familiar → trusted → strongly connected
```

Thresholds are in `observatory/affect.py` (`RelationshipLabelConfig`).

## Outputs

After `the-last-ai`:

```text
outputs/the_last_ai/observatory_seed0/
  agent_000_life.json   # authoritative
  agent_000_life.txt    # meticulously readable biography
  agent_000_life.html   # tabbed explorer (Overview/Life/Memory/…)
```

## CLI

```bash
last-ai the-last-ai --seed 0 --agents 100
last-ai inspect-agent outputs/the_last_ai/final_sim_seed0.json agent_000
last-ai inspect-agent outputs/the_last_ai/observatory_seed0/agent_000_life.json agent_000
```

Open the HTML file in a browser for the interactive observatory.

## Historical vs active

After disappearance, counterparts remain inspectable as:

```text
HISTORICAL — AGENT NO LONGER PRESENT
```

Their messages, relationship trajectories, and memories are retained.

## Source of truth

```text
Simulation → technical event log / observatory JSON → human TXT/HTML
```

Never treat the human-readable layer as authoritative.
