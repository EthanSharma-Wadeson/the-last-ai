"""Equations for computational loss associated with disappeared entities.

Documentation of update rules
-----------------------------

**Entity significance** (before disappearance), bounded [0, 1]:

    S = 0.28·strength
      + 0.18·trust
      + 0.14·min(1, interactions/40)
      + 0.14·memory_strength
      + 0.10·familiarity
      + 0.08·min(1, communicate_count/12)
      + 0.08·clip(predicted_utility, 0, 1)

**Initial social-loss component** at disappearance:

    L0 = S · (0.45 + 0.35·presence_expectation + 0.20·min(1, info_reliability))

**Per-tick update** (entity still historical):

    prediction_disruption_t = 0.85·prediction_disruption_{t-1}
                              + 0.15·presence_prediction_error_t

    search_pressure_t decays 0.92× each tick without a search; +0.25 on failed search

    memory_drive_t = decayed_memory_strength_t

    L_t = clip(
        0.55·L0·exp(-λ_adapt · age_since_loss)
      + 0.25·S·prediction_disruption_t
      + 0.20·S·search_pressure_t
      , 0, 1
    )
    optionally boosted by +0.05·S when memory is retrieved this tick (capped at 1)

**Agent aggregate social_loss**:

    social_loss = 1 - Π_i (1 - L_i)   over historical (non-forgotten) entities
    clipped to [0, 1]

This is a computational aggregate of unresolved expectations about absent entities,
not a sadness variable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from the_last_ai.types import JSONDict


class EntityStatus(str, Enum):
    ACTIVE = "ACTIVE"
    HISTORICAL = "HISTORICAL"
    FORGOTTEN = "FORGOTTEN"


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def compute_entity_significance(
    *,
    strength: float,
    trust: float,
    interaction_count: int,
    memory_strength: float,
    familiarity: float,
    communicate_count: int = 0,
    predicted_utility: float = 0.0,
) -> float:
    """Weighted importance of an entity to the agent before/at disappearance."""
    return _clip(
        0.28 * strength
        + 0.18 * trust
        + 0.14 * min(1.0, interaction_count / 40.0)
        + 0.14 * memory_strength
        + 0.10 * familiarity
        + 0.08 * min(1.0, communicate_count / 12.0)
        + 0.08 * _clip(predicted_utility, 0.0, 1.0)
    )


def compute_entity_social_loss(
    *,
    significance: float,
    presence_expectation: float,
    information_reliability: float,
    ticks_since_disappearance: int,
    prediction_disruption: float,
    search_pressure: float,
    memory_retrieved_this_tick: bool,
    adapt_lambda: float = 0.004,
) -> float:
    """Per-entity social-loss component at a given tick after disappearance."""
    l0 = significance * (
        0.45 + 0.35 * _clip(presence_expectation) + 0.20 * _clip(information_reliability)
    )
    age = max(0, ticks_since_disappearance)
    base = 0.55 * l0 * math.exp(-adapt_lambda * age)
    dynamic = 0.25 * significance * _clip(prediction_disruption) + 0.20 * significance * _clip(
        search_pressure
    )
    boost = 0.05 * significance if memory_retrieved_this_tick else 0.0
    return _clip(base + dynamic + boost)


def aggregate_social_loss(components: list[float]) -> float:
    """Combine per-entity losses without simple averaging saturation tricks."""
    product = 1.0
    for c in components:
        product *= 1.0 - _clip(c)
    return _clip(1.0 - product)


@dataclass
class EntityLossRecord:
    """Full computational record for one disappeared (or tracked) entity."""

    entity_id: str
    status: EntityStatus = EntityStatus.ACTIVE
    disappearance_tick: int | None = None
    significance: float = 0.0
    presence_expectation_at_loss: float = 0.0

    # Snapshots at disappearance
    relationship_before: JSONDict = field(default_factory=dict)
    memory_before: JSONDict = field(default_factory=dict)
    prediction_before: JSONDict = field(default_factory=dict)

    # Live / evolving measures
    social_loss: float = 0.0
    peak_social_loss: float = 0.0
    prediction_disruption: float = 0.0
    peak_prediction_disruption: float = 0.0
    search_pressure: float = 0.0
    search_attempts: int = 0
    failed_searches: int = 0
    communication_attempts_after: int = 0
    memory_retrievals_after: int = 0
    memory_strength_current: float = 0.0
    presence_prediction_error_ema: float = 0.0

    # Trajectories: list of {tick, ...}
    loss_trajectory: list[JSONDict] = field(default_factory=list)
    prediction_error_trajectory: list[JSONDict] = field(default_factory=list)
    memory_trajectory: list[JSONDict] = field(default_factory=list)
    causal_events: list[JSONDict] = field(default_factory=list)

    # Adaptation
    adapted: bool = False
    adaptation_tick: int | None = None

    def to_dict(self) -> JSONDict:
        return {
            "entity_id": self.entity_id,
            "status": self.status.value,
            "disappearance_tick": self.disappearance_tick,
            "significance": self.significance,
            "presence_expectation_at_loss": self.presence_expectation_at_loss,
            "relationship_before": dict(self.relationship_before),
            "memory_before": dict(self.memory_before),
            "prediction_before": dict(self.prediction_before),
            "social_loss": self.social_loss,
            "peak_social_loss": self.peak_social_loss,
            "prediction_disruption": self.prediction_disruption,
            "peak_prediction_disruption": self.peak_prediction_disruption,
            "search_pressure": self.search_pressure,
            "search_attempts": self.search_attempts,
            "failed_searches": self.failed_searches,
            "communication_attempts_after": self.communication_attempts_after,
            "memory_retrievals_after": self.memory_retrievals_after,
            "memory_strength_current": self.memory_strength_current,
            "presence_prediction_error_ema": self.presence_prediction_error_ema,
            "loss_trajectory": list(self.loss_trajectory),
            "prediction_error_trajectory": list(self.prediction_error_trajectory),
            "memory_trajectory": list(self.memory_trajectory),
            "causal_events": list(self.causal_events),
            "adapted": self.adapted,
            "adaptation_tick": self.adaptation_tick,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> EntityLossRecord:
        rec = cls(entity_id=str(data["entity_id"]))
        rec.status = EntityStatus(str(data.get("status", "ACTIVE")))
        rec.disappearance_tick = data.get("disappearance_tick")
        for key in (
            "significance",
            "presence_expectation_at_loss",
            "social_loss",
            "peak_social_loss",
            "prediction_disruption",
            "peak_prediction_disruption",
            "search_pressure",
            "memory_strength_current",
            "presence_prediction_error_ema",
        ):
            if key in data:
                setattr(rec, key, float(data[key]))
        for key in (
            "search_attempts",
            "failed_searches",
            "communication_attempts_after",
            "memory_retrievals_after",
        ):
            if key in data:
                setattr(rec, key, int(data[key]))
        rec.relationship_before = dict(data.get("relationship_before") or {})
        rec.memory_before = dict(data.get("memory_before") or {})
        rec.prediction_before = dict(data.get("prediction_before") or {})
        rec.loss_trajectory = list(data.get("loss_trajectory") or [])
        rec.prediction_error_trajectory = list(data.get("prediction_error_trajectory") or [])
        rec.memory_trajectory = list(data.get("memory_trajectory") or [])
        rec.causal_events = list(data.get("causal_events") or [])
        rec.adapted = bool(data.get("adapted", False))
        rec.adaptation_tick = data.get("adaptation_tick")
        return rec
