"""CLI entry for live spectator mode."""

from __future__ import annotations

from pathlib import Path

from the_last_ai.spectator.server import run_spectator_server
from the_last_ai.spectator.session import SpectatorSession
from the_last_ai.spectator.terminal import run_terminal_spectator


def run_spectate_command(
    *,
    seed: int = 0,
    agents: int = 20,
    ticks: int | None = 5000,
    schedule: list[int] | None = None,
    ticks_between: int = 30,
    final_ticks: int = 40,
    width: int | None = None,
    height: int | None = None,
    resources: int | None = None,
    spectate: str = "agent_000",
    speed: float = 5.0,
    port: int = 8765,
    host: str = "127.0.0.1",
    terminal: bool = False,
    no_open: bool = False,
    output_dir: str | Path = "outputs/spectator",
    demo: str | None = None,
) -> SpectatorSession:
    """
    Start a live spectator session.

    demo:
      - loss: bond with partner, remove them, watch computational response
      - contrast: friend removal then stranger removal — compare responses
      - collapse: short population schedule
      - None / free: free-run
    """
    demo = (demo or "").strip().lower() or None
    if demo == "free":
        demo = None

    max_ticks = ticks
    if demo in {"loss", "contrast"} and ticks is None:
        max_ticks = 280 if demo == "contrast" else 200
    if schedule and ticks is None and demo not in {"loss", "contrast"}:
        max_ticks = None
    if schedule and ticks is not None:
        max_ticks = max(ticks, ticks_between * (len(schedule) + 2) + final_ticks + agents)
    if demo == "collapse" and not schedule:
        schedule = None  # session.create invents a short schedule

    session = SpectatorSession.create(
        seed=seed,
        n_agents=agents,
        width=width,
        height=height,
        n_resources=resources,
        schedule=list(schedule or []),
        ticks_between=ticks_between,
        final_ticks=final_ticks,
        max_ticks=max_ticks,
        spectate_id=spectate,
        speed=speed,
        output_dir=output_dir,
        demo=demo,
    )

    if terminal:
        run_terminal_spectator(session)
    else:
        run_spectator_server(
            session,
            host=host,
            port=port,
            open_browser=not no_open,
        )
    return session
