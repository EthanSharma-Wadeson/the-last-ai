# Computational Loss

## What this means in The Last AI

**Computational loss** is a research construct for measurable changes in an artificial agent’s internal state and behaviour after another entity that was previously represented as socially significant disappears permanently from the world.

It is **not** a claim that the agent feels grief, sadness, pain, or any subjective emotion.

> Observed computational response ≠ subjective emotional experience.

## What is measured

| Dimension | Source |
|-----------|--------|
| `social_loss` | Derived from relationship significance, residual presence expectation, prediction disruption, search pressure, memory retrieval |
| `prediction_disruption` | EMA of presence/position prediction mismatch for the absent entity |
| `memory_activation` / retrievals | LTM retrieve calls for the historical entity |
| `uncertainty`, `security`, `novelty`, `valence`, `activation`, `social_drive`, `goal_success` | Existing observatory `compute_affect` dimensions |
| Behaviour windows | Exploration, social actions, seek, communicate, investigate rates before/after |

## How `social_loss` is calculated

See equations in `src/the_last_ai/loss/model.py`.

**Significance** before disappearance combines strength, trust, interactions, memory strength, familiarity, communication count, and predicted utility.

**Per-entity loss** starts from significance × expectation factors, then mixes decaying baseline with ongoing prediction disruption and search pressure. Memory retrieval adds a small temporary boost.

**Agent aggregate**: `1 − Π(1 − L_i)` over historical entities.

## Disappearance vs internal model

When an entity disappears:

- **WORLD**: entity is removed from active positions.
- **INTERNAL**: relationship, memory, and world-model entries may remain.

Status labels: **ACTIVE** | **HISTORICAL** | **FORGOTTEN**.

## Prediction

World-model entity dynamics are **not** wiped on disappearance. Presence/position predictions can continue, producing scoring errors when the entity is not observed. The loss tracker records the adaptation trajectory.

## Memory

Entity memory decays by existing LTM rules. Disappearance does not delete representations. Retrieval while absent is counted.

## Behaviour

Seek toward last-known location is instrumented as search. Failed searches at empty expected locations raise search pressure. Pre/post behaviour windows are measured, not assumed.

## What “sadness-like” means here

Optional analogy language (e.g. “sadness-like computational response”) refers to **patterns** such as elevated `social_loss`, negative valence dimension, residual search. It never asserts felt emotion.

## Controls

Experiment 9 compares:

1. Minimal prior relationship + disappearance  
2. Moderate relationship + disappearance  
3. Strong relationship + disappearance  
4. Strong relationship, entity **remains** (time control)  
5. Strong formation + **resource shock** without social disappearance  

Plus a sequential multi-disappearance probe.

## Adaptation criterion (configurable)

Default (`AdaptationConfig`): after ≥10 ticks post-loss,

- `social_loss ≤ 0.35 × peak`
- `prediction_disruption ≤ 0.15`
- `search_pressure ≤ 0.15`

## How to run

```bash
last-ai experiment-9 --seed 0
last-ai experiment-9 --seeds 0,1,2,3,4,5,6,7,8,9
```

Outputs under `outputs/experiment_9/`:

- `technical_seed*.json` — raw measurements  
- `interpreted_seed*.json` / `.md` — derived narrative only  
- `report.json` — multi-seed aggregates  
- `observatory_seed*/` — Loss & Disappearance HTML tab for the strong condition  

Inspect a survivor after The Last AI:

```bash
last-ai the-last-ai --seed 0 --agents 10 --schedule 5,2,1 --ticks-between 15 --final-ticks 20
last-ai inspect-agent outputs/the_last_ai … agent_XXX
```

Open the HTML and use the **LOSS & DISAPPEARANCE** tab.

## Limitations

- Small grids and short formation windows limit ecological realism.  
- Seek/search depend on existing memory thresholds and RNG.  
- Communication-toward-absent is only counted when a failed location search coincides with prior communication history.  
- Multi-seed means are descriptive; do not overclaim significance.  
- Architecture is frozen — loss emerges from existing systems, so null results are possible and informative.

## Reproducibility

Fixed `--seed` / `--seeds` drive `ExperimentRNG`. Same seeds should reproduce technical JSON peaks within floating noise of identical code paths.
