"""Metrics recording for behavioural and population statistics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from the_last_ai.types import Action, JSONDict, Position


@dataclass
class TickRecord:
    tick: int
    population: int
    positions: dict[str, tuple[int, int]]
    actions: dict[str, str]
    energies: dict[str, float]
    interactions: int
    rewards: dict[str, float] = field(default_factory=dict)
    resources_collected: int = 0


@dataclass
class MetricsRecorder:
    records: list[TickRecord] = field(default_factory=list)
    location_visits: Counter[tuple[int, int]] = field(default_factory=Counter)
    action_counts: Counter[str] = field(default_factory=Counter)
    interaction_events: int = 0
    total_resources_collected: int = 0
    _previous_positions: dict[str, Position] = field(default_factory=dict)

    def record_tick(
        self,
        *,
        tick: int,
        positions: dict[str, Position],
        actions: dict[str, Action],
        energies: dict[str, float],
        interactions: int = 0,
        rewards: dict[str, float] | None = None,
        resources_collected: int = 0,
    ) -> None:
        rewards = rewards or {}
        for pos in positions.values():
            self.location_visits[pos.as_tuple()] += 1
        for action in actions.values():
            self.action_counts[action.value] += 1
        self.interaction_events += interactions
        self.total_resources_collected += resources_collected

        self.records.append(
            TickRecord(
                tick=tick,
                population=len(positions),
                positions={aid: pos.as_tuple() for aid, pos in positions.items()},
                actions={aid: action.value for aid, action in actions.items()},
                energies=dict(energies),
                interactions=interactions,
                rewards=dict(rewards),
                resources_collected=resources_collected,
            )
        )
        self._previous_positions = dict(positions)

    def population_series(self) -> list[tuple[int, int]]:
        return [(record.tick, record.population) for record in self.records]

    def mean_energy_series(self) -> list[tuple[int, float]]:
        series: list[tuple[int, float]] = []
        for record in self.records:
            if not record.energies:
                series.append((record.tick, 0.0))
            else:
                mean = sum(record.energies.values()) / len(record.energies)
                series.append((record.tick, mean))
        return series

    def mean_reward_series(self) -> list[tuple[int, float]]:
        series: list[tuple[int, float]] = []
        for record in self.records:
            if not record.rewards:
                series.append((record.tick, 0.0))
            else:
                mean = sum(record.rewards.values()) / len(record.rewards)
                series.append((record.tick, mean))
        return series

    def exploration_entropy(self) -> float:
        """Shannon entropy over visited cells (nats)."""
        import math

        total = sum(self.location_visits.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in self.location_visits.values():
            p = count / total
            entropy -= p * math.log(p)
        return entropy

    def social_interaction_rate(self) -> float:
        if not self.records:
            return 0.0
        return self.interaction_events / len(self.records)

    def final_mean_energy(self) -> float:
        if not self.records or not self.records[-1].energies:
            return 0.0
        energies = self.records[-1].energies
        return sum(energies.values()) / len(energies)

    def mean_reward(self) -> float:
        if not self.records:
            return 0.0
        totals = [sum(r.rewards.values()) / len(r.rewards) for r in self.records if r.rewards]
        if not totals:
            return 0.0
        return sum(totals) / len(totals)

    def summary(self) -> JSONDict:
        return {
            "ticks_recorded": len(self.records),
            "exploration_entropy": self.exploration_entropy(),
            "social_interaction_rate": self.social_interaction_rate(),
            "action_counts": dict(self.action_counts),
            "unique_cells_visited": len(self.location_visits),
            "final_population": self.records[-1].population if self.records else 0,
            "final_mean_energy": self.final_mean_energy(),
            "mean_reward": self.mean_reward(),
            "total_resources_collected": self.total_resources_collected,
        }

    def to_dict(self) -> JSONDict:
        return {
            "records": [
                {
                    "tick": r.tick,
                    "population": r.population,
                    "positions": r.positions,
                    "actions": r.actions,
                    "energies": r.energies,
                    "interactions": r.interactions,
                    "rewards": r.rewards,
                    "resources_collected": r.resources_collected,
                }
                for r in self.records
            ],
            "location_visits": {f"{x},{y}": count for (x, y), count in self.location_visits.items()},
            "action_counts": dict(self.action_counts),
            "interaction_events": self.interaction_events,
            "total_resources_collected": self.total_resources_collected,
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> MetricsRecorder:
        recorder = cls()
        for item in data.get("records", []):
            recorder.records.append(
                TickRecord(
                    tick=int(item["tick"]),
                    population=int(item["population"]),
                    positions={k: tuple(v) for k, v in item["positions"].items()},
                    actions=dict(item["actions"]),
                    energies={k: float(v) for k, v in item["energies"].items()},
                    interactions=int(item.get("interactions", 0)),
                    rewards={k: float(v) for k, v in item.get("rewards", {}).items()},
                    resources_collected=int(item.get("resources_collected", 0)),
                )
            )
        for key, count in data.get("location_visits", {}).items():
            x_str, y_str = key.split(",")
            recorder.location_visits[(int(x_str), int(y_str))] = int(count)
        recorder.action_counts = Counter(data.get("action_counts", {}))
        recorder.interaction_events = int(data.get("interaction_events", 0))
        recorder.total_resources_collected = int(data.get("total_resources_collected", 0))
        return recorder
