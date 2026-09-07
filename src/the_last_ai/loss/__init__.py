"""Computational loss — measurable responses to permanent entity disappearance.

Scientific constraint:
  Observed computational response ≠ subjective emotional experience.
  No hard-coded grief/sadness. Values derive from relationship, memory,
  prediction, and behaviour already present in the architecture.
"""

from the_last_ai.loss.metrics import (
    AdaptationConfig,
    BehaviourWindow,
    assess_adaptation,
    measure_behaviour_window,
)
from the_last_ai.loss.model import (
    EntityLossRecord,
    EntityStatus,
    compute_entity_significance,
    compute_entity_social_loss,
)
from the_last_ai.loss.tracker import LossTracker

__all__ = [
    "AdaptationConfig",
    "BehaviourWindow",
    "EntityLossRecord",
    "EntityStatus",
    "LossTracker",
    "assess_adaptation",
    "compute_entity_significance",
    "compute_entity_social_loss",
    "measure_behaviour_window",
]
