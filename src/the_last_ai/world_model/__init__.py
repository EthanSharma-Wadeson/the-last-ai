"""World-model package."""

from the_last_ai.world_model.counterfactual import (
    CounterfactualResult,
    simulate_if_entity_absent,
    simulate_if_i_move,
)
from the_last_ai.world_model.model import WorldModel
from the_last_ai.world_model.planning import (
    evaluate_action_options,
    simulate_action_sequence,
    simulate_disappearance_then_act,
)
from the_last_ai.world_model.types import Prediction, PredictionError, PredictionKind

__all__ = [
    "CounterfactualResult",
    "Prediction",
    "PredictionError",
    "PredictionKind",
    "WorldModel",
    "evaluate_action_options",
    "simulate_action_sequence",
    "simulate_disappearance_then_act",
    "simulate_if_entity_absent",
    "simulate_if_i_move",
]
