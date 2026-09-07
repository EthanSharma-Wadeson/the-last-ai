"""Predictive social agent with an internal world model."""

from __future__ import annotations

from the_last_ai.agents.motivation import MotivationConfig
from the_last_ai.agents.social_agent import SocialAgent, SocialConfig
from the_last_ai.agents.value_learner import LearningConfig
from the_last_ai.memory.long_term import MemoryConfig
from the_last_ai.rng import ExperimentRNG
from the_last_ai.types import Action, JSONDict, Observation
from the_last_ai.world_model.counterfactual import simulate_if_entity_absent, simulate_if_i_move
from the_last_ai.world_model.model import WorldModel
from the_last_ai.world_model.planning import (
    evaluate_action_options,
    simulate_action_sequence,
    simulate_disappearance_then_act,
)


class PredictiveAgent(SocialAgent):
    """
    Extends social cognition with predict → error → update.

    Can learn an internal predictive model of a changing world, including other
    agents, and continue using that model when parts of the world disappear.
    """

    def __init__(
        self,
        agent_id: str,
        *,
        energy: float = 100.0,
        perception_radius: int = 3,
        stm_capacity: int = 32,
        motivation: MotivationConfig | None = None,
        learning: LearningConfig | None = None,
        memory_config: MemoryConfig | None = None,
        social_config: SocialConfig | None = None,
    ) -> None:
        super().__init__(
            agent_id,
            energy=energy,
            perception_radius=perception_radius,
            stm_capacity=stm_capacity,
            motivation=motivation,
            learning=learning,
            memory_config=memory_config,
            social_config=social_config,
        )
        self.world_model = WorldModel()
        self.last_prediction_errors: list = []
        self.prediction_cycles: int = 0

    def observe(self, world, *, cognitive: bool = True) -> Observation:
        observation = super().observe(world, cognitive=cognitive)
        if not cognitive:
            return observation
        # Verify last cycle's predictions, ingest observation, form next predictions.
        self.last_prediction_errors = self.world_model.observe_and_update(observation)
        self.world_model.form_predictions(observation)
        self.prediction_cycles += 1
        return observation

    def select_action(self, observation: Observation, rng: ExperimentRNG) -> Action:
        # Communication-influenced investigation can compete with prediction-guided moves.
        investigate = self._maybe_investigate_from_communication(observation, rng)
        if investigate is not None:
            return investigate

        # Prefer moving toward predicted positions of high-confidence allies.
        predicted_targets = []
        for pred in self.world_model.pending:
            if pred.kind.value != "agent_position":
                continue
            if pred.confidence < 0.35:
                continue
            rel = self.relationships.get(pred.subject_id)
            if rel is None or rel.predicted_utility < 0.05:
                continue
            if isinstance(pred.value, list) and len(pred.value) == 2:
                predicted_targets.append((pred.confidence + rel.predicted_utility, pred))
        if predicted_targets and rng.random() < 0.35:
            predicted_targets.sort(key=lambda item: (-item[0], item[1].subject_id))
            _, best = predicted_targets[0]
            from the_last_ai.agents.memory_agent import _direction_to_target
            from the_last_ai.agents.features import action_toward
            from the_last_ai.types import MOVEMENT_ACTIONS, Position

            direction = _direction_to_target(
                observation.position, (int(best.value[0]), int(best.value[1]))
            )
            action = action_toward(direction)
            if action is not None and action in MOVEMENT_ACTIONS:
                return action

        return super().select_action(observation, rng)

    def counterfactual_move(self, direction: str) -> JSONDict:
        obs = self.last_observation
        if obs is None:
            return {"error": "no observation"}
        return simulate_if_i_move(self.world_model, direction, obs.position).to_dict()

    def counterfactual_absence(self, entity_id: str) -> JSONDict:
        return simulate_if_entity_absent(self.world_model, entity_id).to_dict()

    def imagine_sequence(self, actions: list[str]) -> JSONDict:
        obs = self.last_observation
        if obs is None:
            return {"error": "no observation"}
        return simulate_action_sequence(
            self.world_model, start=obs.position, actions=actions
        ).to_dict()

    def plan(self, *, horizon: int = 2, goal_entity: str | None = None) -> JSONDict:
        obs = self.last_observation
        if obs is None:
            return {"error": "no observation"}
        return evaluate_action_options(
            self.world_model,
            start=obs.position,
            horizon=horizon,
            goal_entity=goal_entity,
        )

    def imagine_absent_then_act(self, entity_id: str, actions: list[str]) -> JSONDict:
        obs = self.last_observation
        if obs is None:
            return {"error": "no observation"}
        return simulate_disappearance_then_act(
            self.world_model,
            start=obs.position,
            missing_entity=entity_id,
            actions=actions,
        ).to_dict()

    def world_model_summary(self) -> JSONDict:
        summary = self.world_model.summary()
        summary["prediction_cycles"] = self.prediction_cycles
        summary["last_error_count"] = len(self.last_prediction_errors)
        return summary

    def to_dict(self) -> JSONDict:
        data = super().to_dict()
        data["world_model"] = self.world_model.to_dict()
        data["prediction_cycles"] = self.prediction_cycles
        return data

    def load_state(self, data: JSONDict) -> None:
        super().load_state(data)
        self.world_model = WorldModel.from_dict(data.get("world_model", {}))
        self.prediction_cycles = int(data.get("prediction_cycles", 0))
