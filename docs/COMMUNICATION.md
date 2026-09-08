# Communication System

## Purpose

Agents can exchange **structured symbolic messages** as a social action.

This is a measurable information channel — not language understanding, consciousness, or emotion.

## What communication is

1. An agent **decides** to communicate (competes with move / cooperate / compete / share).
2. It **selects** a nearby recipient.
3. It **constructs** a short token list from current perception / internal state (max 2–3 tokens).
4. The message is **delivered** to the recipient.
5. The recipient **records** it, may update investigation targets using sender **information reliability**, and may change later actions.
6. Claims can be **verified** against the world; reliability rises or falls.
7. Records persist in **communication memory** (finite capacity) and can remain after counterparts disappear.

## Vocabulary (Version 1)

Fixed primitive tokens mapped to internal concepts:

`hi yes no come go food help stop here there you me wait where gone`

Tokens are symbols. Meanings are separate `Concept` values. Later phases can replace `"food"` with arbitrary tokens (e.g. `"ka"`) without rewriting the exchange pipeline.

Absence-conditioned construction (post-disappearance) may emit `you`/`where`/`gone`/`wait`/`no`/`come`/`here` from measurable `social_loss`, search pressure, and prediction disruption — never from a `sadness` variable.

## Messages

Messages are structured objects, not free-form strings:

```text
Message(sender_id, receiver_id, tokens=["food", "here"], tick=142, claimed_position=...)
```

## Reliability

`information_reliability` on a relationship is distinct from social `trust`.

- Useful verified claims → reliability ↑
- Falsified claims → reliability ↓

Receivers do **not** automatically obey senders; investigation is reliability-weighted.

## Historical vs active

After disappearance:

- communication **memory records** remain (flagged `historical`)
- relationships may still store reliability for absent IDs
- those are **not** counted as active counterparts in the live population

## What the system does NOT claim

- agents understand language
- agents are conscious
- agents feel loneliness / sadness
- emergent natural language (not yet)

Prefer: communication behaviour, symbol usage, information exchange, message–response association, information reliability.

## Experiment 8

```bash
last-ai experiment-8 --seed 0
last-ai experiment-8 --seeds 0,1,2,3,4,5,6,7,8,9
```

Outputs under `outputs/experiment_8/`:

- `communication_seed{N}.json` — technical
- `communication_seed{N}_interpreted.json` — cautious observations
- `communication_seed{N}.md` — human narrative
- `report.json` — multi-seed aggregates + disabled control

## Example interaction (illustrative)

```text
[TICK 142]
agent_001: food here → agent_004
agent_004: come → agent_001
```

Whether this occurs depends on state, adjacency, action selection, and reliability — it is not a scripted dialogue.
