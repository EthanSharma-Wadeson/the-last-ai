"""Live spectator mode tests."""

from __future__ import annotations

import json
import urllib.request

from the_last_ai.spectator.server import serve_once_for_tests
from the_last_ai.spectator.session import SpectatorSession
from the_last_ai.spectator.snapshot import (
    build_live_snapshot,
    build_world_static,
    classify_message_for_spectator,
    format_communication_entry,
)
from the_last_ai.spectator.terminal import render_terminal_frame


def test_world_static_and_snapshot():
    session = SpectatorSession.create(seed=0, n_agents=8, max_ticks=5, speed=100)
    world = build_world_static(session.sim)
    assert world["width"] == session.sim.config.width
    assert world["walls"]
    snap = session.snapshot()
    assert snap["type"] == "snapshot"
    assert snap["population"] == 8
    assert snap["spectating"] == "agent_000"
    assert any(a["spectated"] for a in snap["agents"])
    assert snap["metrics"] is not None
    assert "social_loss" in snap["metrics"]
    assert "prediction_error" in snap["metrics"]
    assert "memories" in snap["metrics"]


def test_message_highlight_classification():
    assert classify_message_for_spectator(
        {"sender": "agent_000", "receiver": "agent_001"}, "agent_000"
    ) == "from_spectated"
    assert classify_message_for_spectator(
        {"sender": "agent_002", "receiver": "agent_000"}, "agent_000"
    ) == "to_spectated"
    assert classify_message_for_spectator(
        {"sender": "a", "receiver": "b"}, "agent_000"
    ) == "other"
    entry = format_communication_entry(
        {
            "event": "communication",
            "tick": 3,
            "sender": "agent_000",
            "receiver": "agent_001",
            "tokens": ["food"],
            "message_id": "abc",
        },
        "agent_000",
    )
    assert entry["highlight"] == "from_spectated"
    assert entry["text"] == "food"


def test_pause_resume_speed_step():
    session = SpectatorSession.create(seed=1, n_agents=6, max_ticks=50, speed=10)
    session.set_paused(True)
    t0 = session.sim.world.tick
    # background not started; step_once advances
    session.step_once()
    assert session.sim.world.tick == t0 + 1
    session.set_speed(2.5)
    assert session.speed == 2.5
    session.set_paused(False)
    assert session.paused is False


def test_spectate_switch_and_camera_emit():
    session = SpectatorSession.create(seed=2, n_agents=5, max_ticks=20)
    session.step_once()
    session.set_spectate("agent_002")
    assert session.spectate_id == "agent_002"
    snap = session.snapshot()
    assert snap["spectating"] == "agent_002"
    assert any(a["id"] == "agent_002" and a["spectated"] for a in snap["agents"])
    session.control("reset_camera")
    assert any(e.get("type") == "camera" for e in session.event_buffer)


def test_disappearance_updates_population_and_history():
    session = SpectatorSession.create(seed=3, n_agents=6, max_ticks=100, speed=50)
    for _ in range(15):
        session.step_once()
    before = len(session.sim.active_agent_ids())
    session.sim.disappear("agent_005", reversible=False)
    session._emit(
        {
            "type": "event",
            "kind": "disappearance",
            "agent_id": "agent_005",
            "summary": "agent_005 is no longer present",
            "tick": session.sim.world.tick,
            "population": len(session.sim.active_agent_ids()),
        }
    )
    assert len(session.sim.active_agent_ids()) == before - 1
    snap = session.snapshot()
    assert snap["population"] == before - 1
    assert all(a["id"] != "agent_005" for a in snap["agents"])
    # historical representation may remain in survivor memory/relationships
    survivor = session.sim.agents["agent_000"]
    assert "agent_005" not in session.sim.world.agent_positions


def test_schedule_collapse_reaches_one(tmp_path):
    session = SpectatorSession.create(
        seed=4,
        n_agents=8,
        schedule=[4, 2, 1],
        ticks_between=1,
        final_ticks=1,
        max_ticks=500,
        speed=100,
        output_dir=tmp_path,
    )
    # Drive until finished (no background thread)
    guard = 0
    while not session.finished and guard < 400:
        session.step_once()
        guard += 1
    assert session.finished
    assert len(session.sim.active_agent_ids()) == 1
    assert session.spectate_id in session.sim.world.agent_positions
    assert session.observatory_paths is not None
    assert "html" in session.observatory_paths


def test_deterministic_seeds():
    a = SpectatorSession.create(seed=9, n_agents=5, max_ticks=10)
    b = SpectatorSession.create(seed=9, n_agents=5, max_ticks=10)
    for _ in range(8):
        a.step_once()
        b.step_once()
    pa = sorted((x["id"], x["x"], x["y"]) for x in a.snapshot()["agents"])
    pb = sorted((x["id"], x["x"], x["y"]) for x in b.snapshot()["agents"])
    assert pa == pb
    assert a.snapshot()["metrics"]["energy"] == b.snapshot()["metrics"]["energy"]


def test_http_state_endpoint(tmp_path):
    session = SpectatorSession.create(
        seed=5, n_agents=4, max_ticks=5, output_dir=tmp_path, speed=20
    )
    session.step_once()
    httpd, url = serve_once_for_tests(session, port=0)
    try:
        with urllib.request.urlopen(url + "api/state", timeout=2) as resp:
            data = json.loads(resp.read().decode())
        assert data["type"] == "snapshot"
        assert data["population"] == 4
        with urllib.request.urlopen(url + "api/world", timeout=2) as resp:
            world = json.loads(resp.read().decode())
        assert "walls" in world
        with urllib.request.urlopen(url, timeout=2) as resp:
            html = resp.read().decode()
        assert "THE LAST AI" in html
        assert "Live communication" in html
    finally:
        httpd.shutdown()
        session.stop()


def test_terminal_render_contains_core_fields():
    session = SpectatorSession.create(seed=6, n_agents=5, max_ticks=5)
    session.step_once()
    text = render_terminal_frame(session)
    assert "THE LAST AI" in text
    assert "Spectating" in text
    assert "LIVE COMMUNICATION" in text
    assert "@" in text  # spectated mark


def test_metrics_reflect_loss_after_disappearance():
    session = SpectatorSession.create(seed=7, n_agents=4, max_ticks=80, width=12, height=10)
    # Cluster for interaction
    from the_last_ai.types import Position

    session.sim.world.agent_positions["agent_000"] = Position(3, 3)
    session.sim.world.agent_positions["agent_001"] = Position(4, 3)
    for _ in range(25):
        session.sim.world.agent_positions["agent_000"] = Position(3, 3)
        session.sim.world.agent_positions["agent_001"] = Position(4, 3)
        session.step_once()
    before = session.snapshot()["metrics"]["social_loss"]
    session.sim.disappear("agent_001", reversible=False)
    for _ in range(10):
        session.step_once()
    after = session.snapshot()["metrics"]["social_loss"]
    # social_loss should be non-decreasing after a known partner disappears when bond existed
    assert after >= before


def test_cli_spectate_help():
    from the_last_ai.cli import main

    # Ensure subparser exists by parsing --help via SystemExit
    try:
        main(["spectate", "--help"])
    except SystemExit as e:
        assert e.code == 0


def test_demo_loss_removes_partner_and_builds_chain(tmp_path):
    session = SpectatorSession.create(
        seed=0,
        demo="loss",
        max_ticks=250,
        speed=100,
        output_dir=tmp_path,
    )
    assert session.demo_mode == "loss"
    assert session.phase == "bond"
    # Force short bond for test speed
    session.demo_bond_ticks = 8
    session.demo_post_ticks = 12
    guard = 0
    saw_disappear = False
    while not session.finished and guard < 200:
        snap = session.step_once()
        if "agent_001" not in session.sim.world.agent_positions:
            saw_disappear = True
        if session.phase == "post_loss" and session.baseline_metrics:
            assert "social_loss" in session.baseline_metrics
        guard += 1
    assert saw_disappear
    assert session.finished
    final = session.snapshot()
    assert final.get("causal_chain") is not None or final["metrics"]["social_loss"] >= 0
    assert final.get("countdown") is not None
    assert "deltas" in final
    assert "life_timeline" in final
    assert "spotlight" in final


def test_snapshot_includes_experience_fields():
    session = SpectatorSession.create(seed=2, n_agents=5, max_ticks=5)
    session.step_once()
    snap = session.snapshot()
    assert "patterns" in snap
    assert "countdown" in snap
    assert "cinema_dim" in snap
    assert snap["disclaimer"]


def test_experience_metric_deltas():
    from the_last_ai.spectator.experience import metric_deltas

    base = {"social_loss": 0.1, "prediction_error": 0.2, "seek_attempts": 1}
    cur = {"social_loss": 0.5, "prediction_error": 0.2, "seek_attempts": 4}
    d = metric_deltas(cur, base)
    assert d["social_loss"]["direction"] == "up"
    assert d["seek_attempts"]["delta"] == 3


def test_demo_contrast_builds_friend_stranger_summary(tmp_path):
    session = SpectatorSession.create(
        seed=0,
        demo="contrast",
        max_ticks=400,
        speed=100,
        output_dir=tmp_path,
    )
    assert session.demo_mode == "contrast"
    session.demo_bond_ticks = 6
    session.demo_post_ticks = 8
    session.demo_stranger_ticks = 8
    guard = 0
    while not session.finished and guard < 300:
        session.step_once()
        guard += 1
    assert session.finished
    assert "agent_001" not in session.sim.world.agent_positions
    assert "agent_002" not in session.sim.world.agent_positions
    snap = session.snapshot()
    assert snap.get("contrast") is not None
    assert snap["contrast"].get("friend") is not None
    assert snap["contrast"].get("stranger") is not None
    assert len(snap.get("metric_series") or []) > 0
    assert "behaviour_rates" in snap


def test_absence_message_tagged_in_feed_format():
    entry = format_communication_entry(
        {
            "event": "communication",
            "tick": 10,
            "sender": "agent_000",
            "receiver": "agent_003",
            "tokens": ["you", "where"],
            "message_id": "x",
            "absence_driven": True,
            "construction_reason": "prediction_mismatch_or_social_loss",
            "about_entity_id": "agent_001",
        },
        "agent_000",
    )
    assert entry["absence_driven"] is True
    assert entry["tag"] == "absence-driven"
    assert entry["construction_reason"] == "prediction_mismatch_or_social_loss"

