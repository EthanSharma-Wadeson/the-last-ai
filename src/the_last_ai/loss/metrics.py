"""Behaviour windows and adaptation criteria for computational loss."""

from __future__ import annotations

from dataclasses import dataclass

from the_last_ai.types import JSONDict


@dataclass
class AdaptationConfig:
    """When post-loss computational state is considered adapted toward baseline."""

    social_loss_fraction_of_peak: float = 0.35  # loss <= peak * this
    prediction_disruption_max: float = 0.15
    search_pressure_max: float = 0.15
    min_ticks_after_loss: int = 10


@dataclass
class BehaviourWindow:
    label: str
    tick_start: int
    tick_end: int
    exploration_unique_cells: int = 0
    social_actions: int = 0
    communicate_actions: int = 0
    approach_actions: int = 0
    avoid_actions: int = 0
    seek_attempts: int = 0
    investigation_actions: int = 0
    resources_collected_delta: int = 0
    ticks: int = 0

    def rates(self) -> JSONDict:
        t = max(1, self.ticks)
        return {
            "exploration_cells_per_tick": self.exploration_unique_cells / t,
            "social_actions_per_tick": self.social_actions / t,
            "communicate_per_tick": self.communicate_actions / t,
            "approach_per_tick": self.approach_actions / t,
            "avoid_per_tick": self.avoid_actions / t,
            "seek_per_tick": self.seek_attempts / t,
            "investigation_per_tick": self.investigation_actions / t,
            "resource_collect_per_tick": self.resources_collected_delta / t,
        }

    def to_dict(self) -> JSONDict:
        return {
            "label": self.label,
            "tick_start": self.tick_start,
            "tick_end": self.tick_end,
            "ticks": self.ticks,
            "exploration_unique_cells": self.exploration_unique_cells,
            "social_actions": self.social_actions,
            "communicate_actions": self.communicate_actions,
            "approach_actions": self.approach_actions,
            "avoid_actions": self.avoid_actions,
            "seek_attempts": self.seek_attempts,
            "investigation_actions": self.investigation_actions,
            "resources_collected_delta": self.resources_collected_delta,
            "rates": self.rates(),
        }


def measure_behaviour_window(
    agent,
    *,
    label: str,
    tick_start: int,
    tick_end: int,
    snapshot_before: JSONDict,
    snapshot_after: JSONDict,
) -> BehaviourWindow:
    """Diff counters between two behavioural snapshots."""
    def _g(d: JSONDict, key: str, default: int = 0) -> int:
        return int(d.get(key, default) or 0)

    return BehaviourWindow(
        label=label,
        tick_start=tick_start,
        tick_end=tick_end,
        ticks=max(1, tick_end - tick_start),
        exploration_unique_cells=max(
            0, _g(snapshot_after, "unique_cells") - _g(snapshot_before, "unique_cells")
        ),
        social_actions=max(
            0, _g(snapshot_after, "social_total") - _g(snapshot_before, "social_total")
        ),
        communicate_actions=max(
            0,
            _g(snapshot_after, "communicate") - _g(snapshot_before, "communicate"),
        ),
        approach_actions=max(
            0, _g(snapshot_after, "approach") - _g(snapshot_before, "approach")
        ),
        avoid_actions=max(0, _g(snapshot_after, "avoid") - _g(snapshot_before, "avoid")),
        seek_attempts=max(0, _g(snapshot_after, "seek") - _g(snapshot_before, "seek")),
        investigation_actions=max(
            0, _g(snapshot_after, "investigate") - _g(snapshot_before, "investigate")
        ),
        resources_collected_delta=max(
            0, _g(snapshot_after, "resources") - _g(snapshot_before, "resources")
        ),
    )


def behaviour_snapshot(agent) -> JSONDict:
    social = dict(getattr(agent, "social_action_counts", {}) or {})
    return {
        "unique_cells": len(getattr(agent, "visit_counts", {}) or {}),
        "social_total": int(sum(social.values())),
        "communicate": int(social.get("communicate", 0)),
        "approach": int(getattr(agent, "approach_actions", 0)),
        "avoid": int(getattr(agent, "avoid_actions", 0)),
        "seek": int(getattr(agent, "seek_attempts", 0)),
        "investigate": int(getattr(agent, "investigation_actions", 0)),
        "resources": int(getattr(agent, "resources_collected", 0)),
    }


def assess_adaptation(
    *,
    social_loss: float,
    peak_social_loss: float,
    prediction_disruption: float,
    search_pressure: float,
    ticks_since_loss: int,
    config: AdaptationConfig | None = None,
) -> bool:
    cfg = config or AdaptationConfig()
    if ticks_since_loss < cfg.min_ticks_after_loss:
        return False
    peak = max(peak_social_loss, 1e-6)
    return (
        social_loss <= peak * cfg.social_loss_fraction_of_peak
        and prediction_disruption <= cfg.prediction_disruption_max
        and search_pressure <= cfg.search_pressure_max
    )
