"""Spectator session: runs the real simulation with pause/speed/schedule/demo control."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from the_last_ai.agents.predictive_agent import PredictiveAgent
from the_last_ai.agents.social_agent import SocialConfig
from the_last_ai.experiments.snapshots import capture_agent_mind
from the_last_ai.observatory.export import export_agent_life
from the_last_ai.simulation.engine import Simulation, SimulationConfig
from the_last_ai.spectator.snapshot import (
    build_agent_metrics,
    build_live_snapshot,
    build_world_static,
    format_communication_entry,
)
from the_last_ai.spectator.experience import metric_deltas
from the_last_ai.types import JSONDict, Position


def _configure_agents(sim: Simulation, *, dense: bool = False) -> None:
    bias = 1.1 if dense else 0.55
    coop = 0.9 if dense else 0.45
    interact = 0.92 if dense else 0.65
    for agent in sim.agents.values():
        if isinstance(agent, PredictiveAgent):
            agent.social_config = SocialConfig(
                cooperate_bias=coop,
                interact_probability=interact,
                enable_indirect_memory_hook=True,
                enable_symbolic_communication=True,
                communication_bias=bias,
            )


def _cluster_pair(sim: Simulation, a: str, b: str, origin: Position) -> None:
    if a in sim.world.agent_positions:
        sim.world.agent_positions[a] = origin
    if b in sim.world.agent_positions:
        sim.world.agent_positions[b] = origin.offset(1, 0)


def _park_stranger(sim: Simulation, stranger_id: str) -> None:
    """Keep the stranger away from the bonded pair during friend formation."""
    if stranger_id not in sim.world.agent_positions:
        return
    w = sim.world.grid.width
    h = sim.world.grid.height
    sim.world.agent_positions[stranger_id] = Position(max(2, w - 3), max(2, h - 3))


@dataclass
class SpectatorSession:
    """
    Authoritative simulation + live event bus for the HTML spectator.

    The browser only renders snapshots/events; it never simulates.
    """

    sim: Simulation
    spectate_id: str = "agent_000"
    speed: float = 5.0
    paused: bool = False
    schedule: list[int] = field(default_factory=list)
    ticks_between: int = 30
    final_ticks: int = 40
    max_ticks: int | None = None
    output_dir: Path = field(default_factory=lambda: Path("outputs/spectator"))
    phase: str = "running"
    finished: bool = False
    tick_index: int = 0
    started_at: float = field(default_factory=time.time)
    event_buffer: deque = field(default_factory=lambda: deque(maxlen=300))
    subscribers: list = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)
    demo_mode: str | None = None
    demo_partner_id: str = "agent_001"
    demo_stranger_id: str = "agent_002"
    demo_bond_ticks: int = 45
    demo_post_ticks: int = 90
    demo_stranger_ticks: int = 50
    baseline_metrics: JSONDict | None = None
    contrast: JSONDict | None = None
    cinema_dim: bool = True
    metric_history: deque = field(default_factory=lambda: deque(maxlen=120))
    _comm_cursor: int = 0
    _schedule_idx: int = 0
    _phase_ticks: int = 0
    _dwell_active: bool = False
    _last_loss_alert_tick: int = -10_000
    _seen_partners: set[str] = field(default_factory=set)
    _last_trust: dict[str, float] = field(default_factory=dict)
    _last_search_attempts: int = 0
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    observatory_paths: dict[str, str] | None = None
    world_static: JSONDict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        seed: int = 0,
        n_agents: int = 20,
        width: int | None = None,
        height: int | None = None,
        n_resources: int | None = None,
        schedule: list[int] | None = None,
        ticks_between: int = 30,
        final_ticks: int = 40,
        max_ticks: int | None = 5000,
        spectate_id: str = "agent_000",
        speed: float = 5.0,
        output_dir: str | Path = "outputs/spectator",
        agent_type: str = "PredictiveAgent",
        demo: str | None = None,
    ) -> SpectatorSession:
        demo = (demo or "").strip().lower() or None
        schedule = list(schedule or [])

        if demo == "loss" or demo == "contrast":
            n_agents = min(n_agents, 12) if n_agents else 12
            n_agents = max(4 if demo == "loss" else 5, n_agents)
            width = width or 14
            height = height or 10
            n_resources = n_resources if n_resources is not None else 6
            if max_ticks is None:
                max_ticks = 280 if demo == "contrast" else 200
            speed = max(speed, 4.0)
        elif demo == "collapse":
            if not schedule:
                schedule = [max(2, n_agents // 2), max(1, n_agents // 4), 1]
                schedule = sorted({n for n in schedule if 1 <= n < n_agents})
            ticks_between = min(ticks_between, 20)
            if max_ticks is None:
                max_ticks = ticks_between * (len(schedule) + 3) + final_ticks + n_agents

        if width is None:
            width = max(20, min(60, int((n_agents * 2.2) ** 0.5) + 8))
        if height is None:
            height = max(16, min(45, int(width * 0.75)))
        if n_resources is None:
            n_resources = max(4, n_agents // 3)

        config = SimulationConfig(
            seed=seed,
            n_agents=n_agents,
            n_resources=n_resources,
            width=width,
            height=height,
            agent_type=agent_type,
            perception_radius=4,
            initial_energy=100.0,
            resource_energy=18.0,
            resource_regen_interval=12,
        )
        sim = Simulation.create(config)
        dense = demo in {"loss", "collapse", "contrast"}
        _configure_agents(sim, dense=dense)
        if spectate_id in sim.agents:
            sim.observatory.focus({spectate_id})

        partner = "agent_001" if "agent_001" in sim.agents else None
        stranger = "agent_002" if "agent_002" in sim.agents else None
        if demo in {"loss", "contrast"}:
            phase = "bond"
            if partner:
                _cluster_pair(sim, spectate_id, partner, Position(4, 4))
            if demo == "contrast" and stranger:
                _park_stranger(sim, stranger)
        elif schedule:
            phase = "formation"
        else:
            phase = "free_run"

        session = cls(
            sim=sim,
            spectate_id=spectate_id if spectate_id in sim.agents else "agent_000",
            speed=speed,
            schedule=schedule,
            ticks_between=ticks_between,
            final_ticks=final_ticks,
            max_ticks=max_ticks,
            output_dir=Path(output_dir),
            phase=phase,
            demo_mode=demo,
            demo_partner_id=partner or "agent_001",
            demo_stranger_id=stranger or "agent_002",
        )
        session.world_static = build_world_static(sim)
        session.output_dir.mkdir(parents=True, exist_ok=True)
        session._emit(
            {
                "type": "status",
                "message": (
                    f"Spectator started ({demo or 'free'} mode)"
                    if demo
                    else "Spectator session started"
                ),
                "tick": 0,
                "population": len(sim.active_agent_ids()),
                "demo_mode": demo,
            }
        )
        if demo == "loss":
            session._cinema(
                "demo_start",
                "Loss demo: bond with a partner, then remove them — watch computational response",
            )
        elif demo == "contrast":
            session._cinema(
                "demo_start",
                "Contrast demo: remove a bonded friend, then a stranger — compare computational response",
            )
        return session

    def subscribe(self) -> deque:
        q: deque = deque(maxlen=500)
        with self.lock:
            self.subscribers.append(q)
            q.append({"type": "world", **self.world_static})
            for ev in list(self.event_buffer)[-50:]:
                q.append(ev)
            q.append(self.snapshot())
        return q

    def unsubscribe(self, q: deque) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def _emit(self, event: JSONDict) -> None:
        self.event_buffer.append(event)
        for q in list(self.subscribers):
            q.append(event)

    def _cinema(self, kind: str, summary: str, **extra) -> None:
        self._emit(
            {
                "type": "cinema",
                "kind": kind,
                "summary": summary,
                "tick": self.sim.world.tick,
                **extra,
            }
        )
        self._emit(
            {
                "type": "event",
                "kind": kind,
                "summary": summary,
                "tick": self.sim.world.tick,
                **extra,
            }
        )

    def _snapshot_kwargs(self) -> dict:
        return {
            "spectate_id": self.spectate_id,
            "speed": self.speed,
            "paused": self.paused,
            "phase": self.phase,
            "elapsed_seconds": time.time() - self.started_at,
            "finished": self.finished,
            "tick_index": self.tick_index,
            "baseline_metrics": self.baseline_metrics,
            "schedule": self.schedule,
            "schedule_idx": self._schedule_idx,
            "phase_ticks": self._phase_ticks,
            "ticks_between": self.ticks_between,
            "final_ticks": self.final_ticks,
            "demo_mode": self.demo_mode,
            "demo_bond_ticks": self.demo_bond_ticks,
            "demo_post_ticks": self.demo_post_ticks,
            "demo_stranger_ticks": self.demo_stranger_ticks,
            "cinema_dim": self.cinema_dim,
            "metric_series": list(self.metric_history),
            "contrast": self.contrast,
        }

    def snapshot(self) -> JSONDict:
        with self.lock:
            return build_live_snapshot(self.sim, **self._snapshot_kwargs())

    def set_spectate(self, agent_id: str) -> None:
        with self.lock:
            if agent_id in self.sim.world.agent_positions:
                self.spectate_id = agent_id
                self.sim.observatory.focus({agent_id})
                self._emit(
                    {
                        "type": "spectate",
                        "agent_id": agent_id,
                        "tick": self.sim.world.tick,
                    }
                )

    def set_paused(self, paused: bool) -> None:
        with self.lock:
            self.paused = paused
            self._emit({"type": "control", "paused": paused, "speed": self.speed})

    def set_speed(self, speed: float) -> None:
        with self.lock:
            self.speed = max(0.1, min(60.0, float(speed)))
            self._emit({"type": "control", "paused": self.paused, "speed": self.speed})

    def step_once(self) -> JSONDict:
        with self.lock:
            if self.finished:
                return self.snapshot()
            self._advance_locked()
            return self.snapshot()

    def _drain_new_communications(self) -> None:
        log = self.sim.communication_log
        while self._comm_cursor < len(log):
            raw = log[self._comm_cursor]
            self._comm_cursor += 1
            if raw.get("event") not in {"communication", "communication_verification"}:
                continue
            entry = format_communication_entry(raw, self.spectate_id)
            self._emit({"type": "communication", **entry})
            if entry.get("absence_driven") and entry["highlight"] in {
                "from_spectated",
                "to_spectated",
            }:
                self._cinema(
                    "absence_message",
                    f"Absence-driven tokens: \"{entry['text']}\" "
                    f"({entry.get('construction_reason') or 'loss state'})",
                    message_id=entry.get("message_id"),
                    about_entity_id=entry.get("about_entity_id"),
                )
            elif entry["highlight"] in {"from_spectated", "to_spectated"}:
                if raw.get("event") == "communication":
                    direction = (
                        "sent"
                        if entry["highlight"] == "from_spectated"
                        else "received"
                    )
                    self._cinema(
                        "communication_moment",
                        f"Spectated agent {direction}: \"{entry['text']}\"",
                        message_id=entry.get("message_id"),
                    )

    def _capture_baseline(self) -> None:
        agent = self.sim.agents.get(self.spectate_id)
        if agent is None:
            return
        self.baseline_metrics = build_agent_metrics(
            agent, active_ids=set(self.sim.active_agent_ids())
        )
        self._cinema(
            "baseline",
            "Baseline metrics captured (pre-disappearance computational state)",
        )

    def _detect_cinema_moments(self) -> None:
        agent = self.sim.agents.get(self.spectate_id)
        if agent is None or not hasattr(agent, "relationships"):
            return
        # First encounters / trust jumps
        for oid, rel in agent.relationships.relationships.items():
            if oid not in self._seen_partners and rel.interaction_count > 0:
                self._seen_partners.add(oid)
                self._cinema(
                    "first_meet",
                    f"First recorded interactions with {oid}",
                    entity_id=oid,
                )
            prev = self._last_trust.get(oid)
            if prev is not None and rel.trust - prev >= 0.12:
                self._cinema(
                    "trust_jump",
                    f"Trust with {oid} rose {prev:.2f} → {rel.trust:.2f}",
                    entity_id=oid,
                )
            self._last_trust[oid] = float(rel.trust)

        # Failed / residual search after loss
        if hasattr(agent, "loss_tracker"):
            seeks = int(getattr(agent, "seek_attempts", 0))
            if seeks > self._last_search_attempts and self.baseline_metrics is not None:
                self._cinema(
                    "search_moment",
                    "Memory-/prediction-guided movement after disappearance",
                )
            self._last_search_attempts = seeks
            for rec in agent.loss_tracker.records.values():
                if rec.adapted and rec.adaptation_tick == self.sim.world.tick:
                    self._cinema(
                        "adaptation",
                        f"Computational adaptation criterion met for {rec.entity_id}",
                        entity_id=rec.entity_id,
                    )

    def _record_metric_point(self, snap: JSONDict | None = None) -> None:
        metrics = (snap or {}).get("metrics") if snap else None
        if metrics is None:
            agent = self.sim.agents.get(self.spectate_id)
            if agent is None:
                return
            metrics = build_agent_metrics(
                agent, active_ids=set(self.sim.active_agent_ids())
            )
        self.metric_history.append(
            {
                "tick": int(self.sim.world.tick),
                "social_loss": float(metrics.get("social_loss") or 0.0),
                "prediction_error": float(metrics.get("prediction_error") or 0.0),
                "search_pressure": float(metrics.get("search_pressure") or 0.0),
                "seek_attempts": int(metrics.get("seek_attempts") or 0),
            }
        )

    def _capture_response_summary(self, label: str) -> JSONDict:
        agent = self.sim.agents.get(self.spectate_id)
        if agent is None:
            return {"label": label}
        metrics = build_agent_metrics(
            agent, active_ids=set(self.sim.active_agent_ids())
        )
        return {
            "label": label,
            "entity_id": (
                self.demo_partner_id if "friend" in label else self.demo_stranger_id
            ),
            "metrics": {
                k: metrics.get(k)
                for k in (
                    "social_loss",
                    "prediction_error",
                    "search_pressure",
                    "seek_attempts",
                    "uncertainty",
                    "security",
                    "valence",
                )
            },
            "deltas": metric_deltas(metrics, self.baseline_metrics),
            "behaviour_rates": {
                "seek_attempts": int(metrics.get("seek_attempts") or 0),
                "delta_seek": int(metrics.get("seek_attempts") or 0)
                - int((self.baseline_metrics or {}).get("seek_attempts") or 0),
                "messages_sent": int(metrics.get("messages_sent") or 0),
                "delta_messages": int(metrics.get("messages_sent") or 0)
                - int((self.baseline_metrics or {}).get("messages_sent") or 0),
            },
            "baseline": {
                k: (self.baseline_metrics or {}).get(k)
                for k in ("social_loss", "prediction_error", "search_pressure")
            },
        }

    def _maybe_demo_loss(self) -> bool:
        """Bond → remove significant partner → observe. Returns True if beat consumed."""
        if self.demo_mode not in {"loss", "contrast"}:
            return False
        partner = self.demo_partner_id
        stranger = self.demo_stranger_id

        if self.phase == "bond":
            if partner in self.sim.world.agent_positions:
                _cluster_pair(
                    self.sim, self.spectate_id, partner, Position(4, 4)
                )
            if self.demo_mode == "contrast" and stranger:
                _park_stranger(self.sim, stranger)
            if self._phase_ticks >= self.demo_bond_ticks:
                self._capture_baseline()
                if partner in self.sim.world.agent_positions:
                    self.sim.disappear(partner, reversible=False)
                    self._cinema(
                        "disappearance",
                        f"{partner} (bonded partner) is no longer present",
                        agent_id=partner,
                        population=len(self.sim.active_agent_ids()),
                        role="friend",
                    )
                    self._emit(
                        {
                            "type": "population",
                            "population": len(self.sim.active_agent_ids()),
                            "tick": self.sim.world.tick,
                        }
                    )
                self.phase = "post_friend" if self.demo_mode == "contrast" else "post_loss"
                self._phase_ticks = 0
                return True
            return False

        if self.phase == "post_loss":
            if self._phase_ticks >= self.demo_post_ticks:
                self._finish_locked()
            return False

        if self.phase == "post_friend":
            if self._phase_ticks >= self.demo_post_ticks:
                friend_summary = self._capture_response_summary("friend")
                self.contrast = {"friend": friend_summary, "stranger": None}
                self._cinema(
                    "contrast_friend_done",
                    "Friend-removal response recorded — removing low-bond stranger next",
                )
                # Fresh baseline just before stranger removal
                self._capture_baseline()
                if stranger in self.sim.world.agent_positions:
                    self.sim.disappear(stranger, reversible=False)
                    self._cinema(
                        "disappearance",
                        f"{stranger} (low-bond stranger) is no longer present",
                        agent_id=stranger,
                        population=len(self.sim.active_agent_ids()),
                        role="stranger",
                    )
                    self._emit(
                        {
                            "type": "population",
                            "population": len(self.sim.active_agent_ids()),
                            "tick": self.sim.world.tick,
                        }
                    )
                self.phase = "post_stranger"
                self._phase_ticks = 0
                return True
            return False

        if self.phase == "post_stranger":
            if self._phase_ticks >= self.demo_stranger_ticks:
                stranger_summary = self._capture_response_summary("stranger")
                friend = (self.contrast or {}).get("friend")
                self.contrast = {
                    "friend": friend,
                    "stranger": stranger_summary,
                    "note": (
                        "Friend had prior bonding; stranger was kept distant. "
                        "Compare social_loss / PE / seek deltas — not claimed feelings."
                    ),
                }
                self._cinema(
                    "contrast_ready",
                    "Friend vs stranger computational contrast ready",
                )
                self._finish_locked()
            return False
        return False

    def _maybe_collapse(self) -> bool:
        if self.demo_mode in {"loss", "contrast"}:
            return self._maybe_demo_loss()
        if not self.schedule:
            return False
        if self.phase == "formation":
            if self._phase_ticks >= self.ticks_between:
                self._capture_baseline()
                self.phase = "collapse"
                self._phase_ticks = 0
                self._schedule_idx = 0
                self._cinema("phase", "Collapse schedule begins")
            return False

        if self.phase == "collapse":
            if self._schedule_idx >= len(self.schedule):
                self.phase = "final"
                self._phase_ticks = 0
                self._dwell_active = False
                self._cinema("phase", "Final survivor phase")
                return False
            target = self.schedule[self._schedule_idx]
            pop = len(self.sim.active_agent_ids())
            if pop > target:
                self._dwell_active = False
                victims = [
                    aid
                    for aid in sorted(self.sim.active_agent_ids(), reverse=True)
                    if aid != self.spectate_id
                ]
                if not victims:
                    victims = [
                        aid
                        for aid in sorted(self.sim.active_agent_ids(), reverse=True)
                        if aid != "agent_000"
                    ]
                if victims:
                    gone = victims[0]
                    self.sim.disappear(gone, reversible=False)
                    self._cinema(
                        "disappearance",
                        f"{gone} is no longer present",
                        agent_id=gone,
                        population=len(self.sim.active_agent_ids()),
                    )
                    self._emit(
                        {
                            "type": "population",
                            "population": len(self.sim.active_agent_ids()),
                            "tick": self.sim.world.tick,
                        }
                    )
                    active = self.sim.active_agent_ids()
                    if self.spectate_id not in active and active:
                        self.spectate_id = (
                            "agent_000" if "agent_000" in active else sorted(active)[0]
                        )
                        self.sim.observatory.focus({self.spectate_id})
                        self._emit(
                            {
                                "type": "spectate",
                                "agent_id": self.spectate_id,
                                "tick": self.sim.world.tick,
                                "reason": "spectated_agent_absent",
                            }
                        )
                    return True
            if not self._dwell_active:
                self._dwell_active = True
                self._phase_ticks = 0
            if self._phase_ticks >= self.ticks_between:
                self.sim.observatory.on_collapse_stage(
                    self.sim,
                    survivor_id=self.spectate_id
                    if self.spectate_id in self.sim.world.agent_positions
                    else "agent_000",
                    population=len(self.sim.active_agent_ids()),
                    label=f"collapse_to_{target}",
                )
                self._cinema(
                    "collapse_stage",
                    f"Population stage reached: {target}",
                    population=len(self.sim.active_agent_ids()),
                )
                self._schedule_idx += 1
                self._phase_ticks = 0
                self._dwell_active = False
            return False

        if self.phase == "final":
            if self._phase_ticks >= self.final_ticks:
                self._finish_locked()
            return False
        return False

    def _finish_locked(self) -> None:
        if self.finished:
            return
        self.finished = True
        self.paused = True
        self.phase = "finished"
        active = self.sim.active_agent_ids()
        if active:
            survivor = (
                "agent_000"
                if "agent_000" in active
                else (self.spectate_id if self.spectate_id in active else active[0])
            )
            self.spectate_id = survivor
            self.sim.observatory.finalize_survivor(self.sim, survivor)
            hist = self.sim.observatory.histories.get(survivor)
            if hist is not None:
                mind = capture_agent_mind(
                    self.sim.agents[survivor], sim=self.sim, label="spectator_final"
                )
                paths = export_agent_life(
                    hist,
                    self.output_dir / "observatory",
                    mind=mind,
                    active_ids=set(active),
                    meta={
                        "mode": "spectator",
                        "demo": self.demo_mode,
                        "seed": self.sim.config.seed,
                        "schedule": list(self.schedule),
                    },
                )
                self.observatory_paths = paths
        self._cinema("finished", "Run complete — open life observatory for full history")
        self._emit(
            {
                "type": "finished",
                "tick": self.sim.world.tick,
                "spectating": self.spectate_id,
                "population": len(active),
                "observatory": self.observatory_paths,
            }
        )

    def _emit_snapshot(self) -> JSONDict:
        snap = build_live_snapshot(self.sim, **self._snapshot_kwargs())
        self._record_metric_point(snap)
        # Refresh series into a follow-up emit via updated kwargs next time;
        # include current history on this snap for UI.
        snap["metric_series"] = list(self.metric_history)
        self._emit(snap)
        self._emit_metric_alerts(snap)
        return snap

    def _advance_locked(self) -> None:
        if self._maybe_collapse():
            self._drain_new_communications()
            self._detect_cinema_moments()
            self._emit_snapshot()
            return

        before_pop = len(self.sim.active_agent_ids())
        # Keep demo pair clustered during bonding
        if self.demo_mode in {"loss", "contrast"} and self.phase == "bond":
            if self.demo_partner_id in self.sim.world.agent_positions:
                _cluster_pair(
                    self.sim, self.spectate_id, self.demo_partner_id, Position(4, 4)
                )
            if self.demo_mode == "contrast":
                _park_stranger(self.sim, self.demo_stranger_id)
        self.sim.step()
        self.tick_index += 1
        self._phase_ticks += 1
        self._drain_new_communications()
        self._detect_cinema_moments()

        after_pop = len(self.sim.active_agent_ids())
        if after_pop < before_pop:
            self._emit(
                {
                    "type": "population",
                    "population": after_pop,
                    "tick": self.sim.world.tick,
                }
            )

        if self.spectate_id not in self.sim.world.agent_positions:
            active = self.sim.active_agent_ids()
            if active:
                self.spectate_id = (
                    "agent_000" if "agent_000" in active else sorted(active)[0]
                )
                self._emit(
                    {
                        "type": "spectate",
                        "agent_id": self.spectate_id,
                        "tick": self.sim.world.tick,
                        "reason": "spectated_agent_absent",
                    }
                )

        self._emit_snapshot()

        if self.max_ticks is not None and self.tick_index >= self.max_ticks:
            if self.demo_mode in {"loss", "contrast"} and self.phase in {
                "bond",
                "post_loss",
                "post_friend",
                "post_stranger",
            }:
                if self.tick_index >= self.max_ticks + 80:
                    self._finish_locked()
            elif not self.schedule or self.phase in {"final", "finished", "free_run"}:
                self._finish_locked()
            elif self.phase == "collapse" and self._schedule_idx >= len(self.schedule):
                self._finish_locked()

        if self.phase == "final" and self._phase_ticks >= self.final_ticks:
            self._finish_locked()
        if self.demo_mode == "loss" and self.phase == "post_loss":
            if self._phase_ticks >= self.demo_post_ticks:
                self._finish_locked()
        if self.demo_mode == "contrast" and self.phase == "post_stranger":
            if self._phase_ticks >= self.demo_stranger_ticks:
                # Handled in _maybe_demo_loss on next collapse check; finish here as backup
                stranger_summary = self._capture_response_summary("stranger")
                friend = (self.contrast or {}).get("friend")
                self.contrast = {
                    "friend": friend,
                    "stranger": stranger_summary,
                    "note": (
                        "Friend had prior bonding; stranger was kept distant. "
                        "Compare social_loss / PE / seek deltas — not claimed feelings."
                    ),
                }
                self._finish_locked()

    def _emit_metric_alerts(self, snap: JSONDict) -> None:
        metrics = snap.get("metrics") or {}
        social_loss = float(metrics.get("social_loss") or 0.0)
        pe = float(metrics.get("prediction_error") or 0.0)
        tick = int(snap.get("tick") or 0)
        if social_loss >= 0.45 and tick - self._last_loss_alert_tick >= 25:
            self._last_loss_alert_tick = tick
            self._cinema(
                "social_loss_state",
                "Elevated computational social-loss state associated with "
                "absence of historically significant entities",
                social_loss=social_loss,
                prediction_error=pe,
            )

    def start_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.is_set():
                with self.lock:
                    if self.finished:
                        break
                    paused = self.paused
                    speed = self.speed
                if paused:
                    time.sleep(0.05)
                    continue
                with self.lock:
                    if not self.finished:
                        self._advance_locked()
                delay = 1.0 / max(0.1, speed)
                time.sleep(delay)

        self._thread = threading.Thread(target=_loop, name="spectator-sim", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def control(self, action: str, **kwargs) -> JSONDict:
        action = action.lower()
        if action == "pause":
            self.set_paused(True)
        elif action == "resume":
            self.set_paused(False)
        elif action == "toggle_pause":
            self.set_paused(not self.paused)
        elif action == "step":
            self.set_paused(True)
            return self.step_once()
        elif action == "speed":
            self.set_speed(float(kwargs.get("value", self.speed)))
        elif action == "spectate":
            self.set_spectate(str(kwargs.get("agent_id", self.spectate_id)))
        elif action == "reset_camera":
            self._emit({"type": "camera", "action": "reset"})
        elif action == "toggle_dim":
            self.cinema_dim = not self.cinema_dim
        elif action == "finish":
            with self.lock:
                self._finish_locked()
        return self.snapshot()
