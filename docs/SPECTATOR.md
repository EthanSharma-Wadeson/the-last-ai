# Live Spectator

## Purpose

Watch the **real** Python simulation as a live artificial-world visualisation.

The browser is a **renderer only**. It does not invent agents, messages, emotions, or movement.

## Guided demos (recommended)

```bash
source .venv/bin/activate

# Best for “computational loss” readability:
# bond with a partner → remove them → watch PE / search / social-loss chain
last-ai spectate --demo loss --seed 0 --speed 6

# Friend vs stranger contrast (bonded partner removed, then low-bond stranger)
last-ai spectate --demo contrast --seed 0 --speed 6

# Short population collapse
last-ai spectate --demo collapse --seed 0 --agents 24 --speed 8

# Free ecology (no scheduled removals)
last-ai spectate --seed 0 --agents 20 --ticks 5000 --speed 5
```

## Full Last AI schedule

```bash
last-ai spectate \
  --seed 0 \
  --agents 100 \
  --schedule 50,25,10,5,2,1 \
  --ticks-between 30 \
  --speed 8
```

## What the upgraded UI shows

| Panel | Meaning |
|-------|---------|
| World canvas | Agents, resources, walls; spectated ring; dimmed others; **spotlight** on last-known locus |
| Computational chain | Expectation → disappearance → PE → memory → search → social-loss → adaptation |
| Metric deltas | Baseline → current after disappearance (↑/↓) |
| Sparklines | Rolling `social_loss` / prediction error / search pressure |
| Behaviour rates | Δseek / Δmessages / Δinvestigate since baseline |
| Contrast panel | Friend vs stranger removal comparison (`--demo contrast`) |
| Pattern pills | prediction-mismatch, social-loss state, residual search, … |
| Life so far | Rolling observatory moments for the spectated agent |
| Communication filters | All / Spectated / Food / **Absence** |
| Countdown chip | Ticks until next phase / disappearance |
| Cinema toasts | Real moments (first meet, trust jump, message, disappearance, search) |

## Absence-conditioned messaging

After a historically significant partner disappears, message construction can emit
tokens driven by measurable loss / search / prediction state, e.g.:

- `you where` — elevated social-loss / prediction mismatch
- `you gone` / `no gone` — residual absence / failed seek
- `come here` — search pressure toward last-known locus (claimed position)

Each message carries `construction_reason`, `absence_driven`, and optional
`about_entity_id` in the inspector. These are symbolic readouts — not claims of emotion.

## Architecture

```text
SpectatorSession (Python)
  → Simulation.step() / disappear()
  → compact snapshot + event bus (+ causal chain / deltas / spotlight / series)
  → HTTP + Server-Sent Events
  → HTML Canvas spectator UI
```

## Controls

| Control | Action |
|---------|--------|
| Pause / Resume / Step | Clock control |
| Speed | 0.25×–20× |
| Scroll / Drag | Zoom / pan |
| Follow | Keep spectated agent centred |
| Dim others | Cinema attention mode |
| Click agent | Spectate |
| Click message | Inspector + camera jump |

## Scientific language

UI may show `social-loss state` / affect labels / pattern pills / absence tokens.
These are **computational**. They are not claims of sadness, consciousness, or sentience.
