"""Terminal fallback spectator — simplified live view of the real simulation."""

from __future__ import annotations

import time

from the_last_ai.spectator.session import SpectatorSession


def _bar(value: float, width: int = 12) -> str:
    value = max(0.0, min(1.0, float(value)))
    n = int(round(value * width))
    return "█" * n + "░" * (width - n)


def render_terminal_frame(session: SpectatorSession) -> str:
    snap = session.snapshot()
    sim = session.sim
    aid = snap.get("spectating") or "agent_000"
    metrics = snap.get("metrics") or {}
    ascii_map = sim.world.grid.render_ascii(dict(sim.world.agent_positions))
    # Mark spectated agent with @
    if aid in sim.world.agent_positions:
        pos = sim.world.agent_positions[aid]
        lines = ascii_map.splitlines()
        row = list(lines[pos.y])
        if 0 <= pos.x < len(row):
            row[pos.x] = "@"
            lines[pos.y] = "".join(row)
        ascii_map = "\n".join(lines)

    recent = [
        e
        for e in session.event_buffer
        if e.get("type") == "communication"
    ][-8:]
    comm_lines = []
    for m in recent:
        tag = ""
        if m.get("highlight") == "from_spectated":
            tag = "[FROM] "
        elif m.get("highlight") == "to_spectated":
            tag = "[TO] "
        comm_lines.append(
            f"  {tag}{m.get('sender')} → {m.get('receiver')}: \"{m.get('text')}\""
        )
    if not comm_lines:
        comm_lines = ["  (no recent communication)"]

    events = [
        e for e in session.event_buffer if e.get("type") == "event"
    ][-5:]
    ev_lines = [f"  t{e.get('tick')} {e.get('summary')}" for e in events] or ["  (none)"]

    lines = [
        "══════════════════════════════════════════════",
        " THE LAST AI — TERMINAL SPECTATOR",
        f" Population {snap.get('population')} · Tick {snap.get('tick')} · "
        f"Speed {snap.get('speed')}× · Phase {snap.get('phase')}",
        f" Spectating {aid}" + (" [PAUSED]" if snap.get("paused") else ""),
        "══════════════════════════════════════════════",
        ascii_map,
        "──────────────────────────────────────────────",
        f" {aid.upper()} — LIVE",
        f" Energy {_bar((metrics.get('energy_pct') or 0)/100)} {metrics.get('energy_pct', 0):.0f}%",
        f" Social drive {metrics.get('social_drive', 0):.3f}  "
        f"Uncertainty {metrics.get('uncertainty', 0):.3f}",
        f" Prediction error {metrics.get('prediction_error', 0):.3f}  "
        f"Social loss {metrics.get('social_loss', 0):.3f}",
        f" Memories {metrics.get('memories', 0)}  "
        f"Relationships {metrics.get('relationships', 0)}  "
        f"Tracked {metrics.get('tracked_entities', 0)}",
        f" State: {metrics.get('affect_label', 'n/a')}",
        "──────────────────────────────────────────────",
        " LIVE COMMUNICATION",
        *comm_lines,
        " EVENTS",
        *ev_lines,
        "──────────────────────────────────────────────",
        " Controls: running automatically (Ctrl+C to stop)",
        " Disclaimer: computational labels ≠ subjective feelings.",
    ]
    return "\n".join(lines)


def run_terminal_spectator(session: SpectatorSession, *, refresh_hz: float = 4.0) -> None:
    session.start_background()
    delay = 1.0 / max(0.5, refresh_hz)
    try:
        while not session.finished:
            frame = render_terminal_frame(session)
            print("\033[2J\033[H" + frame, flush=True)
            time.sleep(delay)
        print(render_terminal_frame(session))
        if session.observatory_paths:
            print(f"\nObservatory: {session.observatory_paths}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        session.stop()
        if not session.finished:
            session.control("finish")
