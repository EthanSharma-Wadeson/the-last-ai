# Roadmap

## Phase 0 — Foundations

- [x] Create repository
- [x] Implement 2D grid
- [x] Implement deterministic simulation loop
- [x] Implement world state
- [x] Implement agent movement
- [x] Implement event system
- [x] Implement save/load
- [x] Implement experiment seeds

## Phase 1 — Basic Artificial Animals

- [x] Add perception
- [x] Add energy
- [x] Add simple motivation
- [x] Add action selection
- [x] Add basic learning
- [x] Add metrics
- [x] Create random baseline

## Phase 2 — Memory

- [x] Implement short-term memory
- [x] Implement long-term memory
- [x] Implement entity memory
- [x] Implement memory decay
- [x] Implement memory retrieval
- [x] Compare memory architectures

## Phase 3 — Social Cognition

- [x] Implement repeated interactions
- [x] Implement learned relationship representations
- [x] Implement social graph
- [x] Add cooperation
- [x] Measure relationship strength

## Phase 3.5 — Social Memory Propagation

- [x] Indirect association transfer via shared absent contacts
- [x] Secondhand entity representations (high uncertainty)
- [x] Controlled triangle experiment (A↔B↔C)
- [x] Control: weak chain
- [x] Control: A↔B only
- [x] Measure A→C effects after B disappears

## Phase 4 — World Model

- [x] Implement state prediction
- [x] Calculate prediction error
- [x] Update internal world model
- [x] Test prediction of environmental dynamics
- [x] Test prediction of other agents
- [x] Counterfactual query scaffold
- [x] Gradual population reduction starter (Experiment 5)

## Phase 5 — The Last AI Experiment

- [x] Implement reversible disappearance
- [x] Run unfamiliar-agent control
- [x] Run familiar-agent experiment
- [x] Run sudden disappearance
- [x] Run gradual population reduction
- [x] Run restoration experiment (Experiment 6)
- [x] Multi-step counterfactual reasoning (Experiment 7)
- [x] Centrepiece collapse 100→50→25→10→5→2→1 with mind snapshots
- [x] Compare final survivor vs inexperienced control
- [x] Multi-seed architecture comparison harness

## Architecture freeze

The prototype cognitive stack is frozen for research:

Random → Value → Memory → Social → Predictive / Last AI

**Additive:** primitive symbolic communication (`communication/`) extends the existing `communicate` social action without replacing the frozen core.

Prefer hundreds of seeded trials, statistics, and failure analysis over new features.

## Phase 5.5 — Emergent communication (additive)

- [x] Structured messages + vocabulary (token ≠ meaning)
- [x] Send / receive / memory / verification / reliability
- [x] Behavioural influence without hard-coded obedience
- [x] Collapse persistence + historical vs active distinction
- [x] Experiment 8 + multi-seed + interpret layer

## Phase 6 — Individual Agent Life Observatory

- [x] Chronological life-event instrumentation
- [x] Derived affective dimensions + causal change records
- [x] Relationship / communication / memory biography export
- [x] TXT + HTML tabbed explorer (JSON authoritative)
- [x] Survivor auto-export after The Last AI
- [x] `last-ai inspect-agent`

## Phase 6c — Computational Loss

- [x] Loss tracker (ACTIVE / HISTORICAL / FORGOTTEN)
- [x] Prediction disruption + memory persistence instrumentation
- [x] Behaviour windows + adaptation criteria
- [x] Observatory **LOSS & DISAPPEARANCE** tab (human-readable)
- [x] Experiment 9 (relationship-dependent + controls)
- [x] Technical vs interpreted JSON
- [x] Docs: `docs/LOSS.md`

## Phase 6b — Research publication loop

- [ ] Run statistically meaningful multi-seed trials
- [ ] Plot survivor structure across architectures
- [ ] Document failure cases
- [ ] Write technical report
- [ ] Release reproducible experiment configs

## Phase 7 — Advanced Cognition (only after research results)

Potential future work — do not start until Phase 6 produces findings:

- [ ] learned memory prioritisation
- [ ] hierarchical memory
- [ ] latent / neural world models
- [ ] neural policies
- [ ] recurrent architectures
- [ ] predictive coding
- [ ] multi-agent communication protocols (learned/arbitrary tokens beyond V1)
- [ ] developmental learning
- [ ] procedural worlds
- [ ] transfer to unseen environments
- [ ] multi-step planning beyond current internal rollout
