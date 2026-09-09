# The Last AI

The Last AI is an experimental artificial-cognition project built around a persistent simulated world containing a population of artificial agents.

The central research question is:

> What happens to an artificial mind's internal model of a world when the entities and environments that created that model gradually disappear?

The project is not designed to create suffering or to claim that its agents are conscious. It studies measurable computational phenomena such as memory, prediction, social representation, adaptation, and forgetting.

## Core concept

A population of artificial agents lives in a simulated environment.

Agents can:

- perceive their environment
- move and interact
- learn from experience
- remember information
- form learned representations of other agents
- develop behavioural preferences
- predict future events
- adapt when the environment changes

Over a sequence of controlled experiments, agents disappear from the world.

Eventually, one agent remains.

The final agent's behaviour and internal representations are analysed to determine what remains of its learned model of the vanished world.

## Design principles

1. Build from simple mechanisms upward.
2. Keep the environment deterministic and reproducible when possible.
3. Separate simulation, cognition, learning, and measurement.
4. Prefer measurable behaviour over anthropomorphic interpretation.
5. Record enough information to reproduce experiments.
6. Keep experiments reversible.
7. Do not intentionally optimise agents for suffering or distress.

## Initial technology direction

The first implementation should favour transparency over performance.

Suggested stack:

- Python
- NumPy
- Standard library
- Matplotlib for analysis and visualisation

External ML frameworks should not be required for the initial cognitive architecture.

## Project status

This repository begins as a research prototype.

The first milestone is a minimal artificial world with multiple agents, memory, social interaction, and reproducible disappearance experiments.

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Run a short simulation:

```bash
last-ai run --seed 0 --ticks 50 --agents 5
```

Run Experiment 0 (random baseline):

```bash
last-ai experiment-0 --seed 0 --ticks 200 --agents 8
```

Run Experiment 1 (value learner vs random in a resource world):

```bash
last-ai experiment-1 --seed 0 --ticks 300 --agents 4 --resources 10
```

Run Experiment 2 (entity memory after disappearance):

```bash
last-ai experiment-2 --seed 0 --familiar-ticks 120 --unfamiliar-ticks 8 --post-ticks 80
```

Run Experiment 3 (bidirectional social relationship, then disappearance):

```bash
last-ai experiment-3 --seed 0 --familiar-ticks 100 --unfamiliar-ticks 6 --post-ticks 60
```

## Research mode (architecture frozen)

The cognitive stack is intentionally frozen for core cognition. **Symbolic communication** is an additive pathway on top of the existing social action `communicate`.

Centrepiece collapse (`100 → 50 → 25 → 10 → 5 → 2 → 1`):

```bash
last-ai the-last-ai --seed 0 --agents 100 --schedule 50,25,10,5,2,1
```

Restoration (Experiment 6):

```bash
last-ai experiment-6 --seed 0
```

Counterfactual reasoning (Experiment 7):

```bash
last-ai experiment-7 --seed 0
```

Primitive communication (Experiment 8):

```bash
last-ai experiment-8 --seed 0
last-ai experiment-8 --seeds 0,1,2,3,4,5,6,7,8,9
```

Architecture comparison across seeds:

```bash
last-ai compare-architectures --seeds 0,1,2,3,4 --agents 24 --schedule 12,6,3,1
```

Human-readable narratives from technical JSON:

```bash
last-ai interpret outputs/the_last_ai/report_seed0.json
last-ai interpret outputs/experiment_8/communication_seed0.json
```

Inspect one agent's life observatory:

```bash
last-ai inspect-agent outputs/the_last_ai/final_sim_seed0.json agent_000
# or open outputs/the_last_ai/observatory_seed0/agent_000_life.html
```

Core research question:

> When a population disappears, what structure of that population remains encoded in the final surviving artificial mind?

Outputs are written under `outputs/`.

## Package layout

```
src/the_last_ai/
  world/          # grid, world state, events
  perception/     # local observations
  agents/         # random → value → memory → social → predictive
  memory/         # short-term, long-term, entity memory
  social/         # interactions, relationships, graph, propagation
  communication/  # vocabulary, messages, memory, exchange/verify
  observatory/    # individual agent life history + biography export
  world_model/    # environmental + agent prediction, error, counterfactuals
  simulation/     # deterministic tick loop
  metrics/        # behavioural / population metrics
  persistence/    # save / load snapshots
  experiments/    # documented experiment runners
```

See `docs/` for specifications, experiments, and roadmap. See `docs/COMMUNICATION.md` for the communication pathway.
## License

MIT — see [`LICENSE`](LICENSE).
