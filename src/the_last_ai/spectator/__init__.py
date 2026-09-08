"""Live spectator mode — watch the real Python simulation in a browser or terminal."""

from the_last_ai.spectator.server import run_spectator_server, serve_once_for_tests
from the_last_ai.spectator.session import SpectatorSession
from the_last_ai.spectator.snapshot import (
    build_live_snapshot,
    build_world_static,
    classify_message_for_spectator,
    format_communication_entry,
)
from the_last_ai.spectator.terminal import render_terminal_frame, run_terminal_spectator

__all__ = [
    "SpectatorSession",
    "build_live_snapshot",
    "build_world_static",
    "classify_message_for_spectator",
    "format_communication_entry",
    "render_terminal_frame",
    "run_spectator_server",
    "run_terminal_spectator",
    "serve_once_for_tests",
]
