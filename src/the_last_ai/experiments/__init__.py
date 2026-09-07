"""Experiment runners."""

from the_last_ai.experiments.architecture_compare import run_architecture_comparison
from the_last_ai.experiments.communication_experiment import run_experiment_8
from the_last_ai.experiments.computational_loss import run_experiment_9
from the_last_ai.experiments.counterfactual_reasoning import run_counterfactual_reasoning
from the_last_ai.experiments.entity_representation import run_entity_representation
from the_last_ai.experiments.gradual_population import (
    run_gradual_population_reduction,
    run_the_last_ai,
)
from the_last_ai.experiments.interpret import interpret_json_file, write_interpretation
from the_last_ai.experiments.learned_environment import run_learned_environment
from the_last_ai.experiments.random_baseline import run_random_baseline
from the_last_ai.experiments.restoration import run_restoration
from the_last_ai.experiments.social_memory_propagation import run_social_memory_propagation
from the_last_ai.experiments.sudden_disappearance import run_sudden_disappearance
from the_last_ai.experiments.world_model_prediction import run_world_model_experiment

__all__ = [
    "interpret_json_file",
    "run_architecture_comparison",
    "run_counterfactual_reasoning",
    "run_entity_representation",
    "run_experiment_8",
    "run_experiment_9",
    "run_gradual_population_reduction",
    "run_learned_environment",
    "run_random_baseline",
    "run_restoration",
    "run_social_memory_propagation",
    "run_sudden_disappearance",
    "run_the_last_ai",
    "run_world_model_experiment",
    "write_interpretation",
]
