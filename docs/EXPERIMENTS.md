# Experiments

## Experiment 0 — Random Baseline

### Purpose

Establish behaviour without learning.

### Setup

Multiple agents move randomly.

### Measurement

Record:

- movement
- interactions
- locations
- energy
- population

This validates the simulator.

---

## Experiment 1 — Learned Environment

### Purpose

Test whether an agent can learn useful environmental patterns.

### Setup

Provide stable resources and repeatable environmental structure.

### Measurement

Compare random and learning agents.

Expected result:

A learning agent should outperform random behaviour on the defined objective.

---

## Experiment 2 — Entity Representation

### Purpose

Test whether repeated interaction produces persistent internal representations.

### Setup

Two agents repeatedly interact.

Afterward, one disappears.

### Measurement

Track:

- representation strength
- predictions concerning the entity
- location preferences
- interaction attempts

Compare with an unfamiliar-agent control.

---

## Experiment 3 — Sudden Disappearance

### Setup

A highly familiar agent disappears suddenly.

### Variables

- interaction frequency
- relationship strength
- memory capacity
- memory decay

### Measurements

Plot behaviour before and after disappearance.

---

## Experiment 4 — Gradual Population Reduction

Start with many agents.

Remove agents at controlled intervals:

1000 -> 500 -> 250 -> 100 -> 50 -> 10 -> 1

The exact population should depend on simulation performance.

### Goal

Determine how the surviving agent's internal model changes as social information disappears.

---

## Experiment 5 — The Last AI

### Setup

A single agent remains after a long period of population interaction.

The agent's final internal state is captured.

### Questions

- What entity representations remain?
- Which locations retain behavioural importance?
- How much information about the previous population survives?
- How does the agent's policy differ from an equivalent agent that never experienced the population?

### Control

Create a new agent with the same architecture but without the previous history.

Compare internal representations and behaviour.

---

## Experiment 6 — Restoration

Because experiments should be reversible, restore a disappeared entity after a controlled period.

Measure:

- recognition
- behavioural adaptation
- prediction accuracy
- time to recover previous interaction patterns

This is particularly useful for distinguishing memory persistence from simple environmental conditioning.

---

## Experiment 8 — Primitive communication

### Setup

Agents may select `communicate` as a social action, construct short token messages, deliver them, and verify claims.

### Questions

- Can agents send and receive structured messages?
- Does useful communication alter subsequent behaviour?
- Does verification update information reliability?
- Does communication appear in relationships and finite communication memory?
- Does communication history remain after population collapse?
- How does this compare to agents with symbolic communication disabled?

### Non-claims

Do not interpret results as language understanding, consciousness, or emotion.

```bash
last-ai experiment-8 --seed 0
last-ai experiment-8 --seeds 0,1,2,3,4,5,6,7,8,9
```

---

## Experiment 9 — Computational Loss

### Purpose

Test whether disappearance of a socially significant entity produces a persistent, relationship-dependent change in computational state and behaviour — without hard-coded grief.

### Setup

Compare minimal / moderate / strong prior relationships, plus no-disappear and resource-shock controls. See `docs/LOSS.md`.

### Measurement

Peak `social_loss`, prediction disruption, memory retrievals, search attempts, behaviour windows, adaptation time.

```bash
last-ai experiment-9 --seed 0
last-ai experiment-9 --seeds 0,1,2,3,4,5,6,7,8,9
```

### Non-claims

Does not demonstrate subjective sadness, grief, or consciousness.

---

## Live Spectator

Watch the real simulation in a browser canvas (or terminal).

```bash
last-ai spectate --seed 0 --agents 20 --ticks 5000
last-ai spectate --seed 0 --agents 100 --schedule 50,25,10,5,2,1
last-ai spectate --terminal --seed 0 --agents 12 --ticks 200
```

See `docs/SPECTATOR.md`.

---

## Reproducibility

Every experiment must record:

- random seed
- world configuration
- agent configuration
- model parameters
- population
- event timeline
- metrics
- software version

Results should be reproducible from a saved configuration.