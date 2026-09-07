"""Deterministic simulation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from the_last_ai.agents.base import Agent, create_agent
from the_last_ai.agents.motivation import MotivationConfig, compute_reward_with_blocked
from the_last_ai.communication.system import exchange_communication, verify_pending_claims
from the_last_ai.metrics.recorder import MetricsRecorder
from the_last_ai.observatory.recorder import ObservatoryRecorder
from the_last_ai.persistence.serialize import (
    load_snapshot,
    restore_agents,
    restore_metrics,
    restore_world,
    save_snapshot,
    snapshot_to_dict,
)
from the_last_ai.rng import ExperimentRNG
from the_last_ai.social.graph import SocialGraph
from the_last_ai.social.propagation import PropagationConfig, PropagationLog, propagate_after_interaction
from the_last_ai.social.resolve import resolve_pair
from the_last_ai.social.types import ACTION_TO_INTERACTION, SOCIAL_ACTIONS, InteractionOutcome
from the_last_ai.types import Action, JSONDict, Observation, Position
from the_last_ai.world.grid import Grid
from the_last_ai.world.world import World


@dataclass
class SimulationConfig:
    width: int = 20
    height: int = 12
    seed: int = 0
    n_agents: int = 5
    perception_radius: int = 3
    initial_energy: float = 100.0
    agent_type: str = "RandomAgent"
    bordered: bool = True
    n_resources: int = 0
    resource_energy: float = 20.0
    resource_regen_interval: int = 15
    energy_cost_per_tick: float = 0.1

    def to_dict(self) -> JSONDict:
        return {
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "n_agents": self.n_agents,
            "perception_radius": self.perception_radius,
            "initial_energy": self.initial_energy,
            "agent_type": self.agent_type,
            "bordered": self.bordered,
            "n_resources": self.n_resources,
            "resource_energy": self.resource_energy,
            "resource_regen_interval": self.resource_regen_interval,
            "energy_cost_per_tick": self.energy_cost_per_tick,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> SimulationConfig:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass
class Simulation:
    config: SimulationConfig
    world: World
    agents: dict[str, Agent]
    rng: ExperimentRNG
    metrics: MetricsRecorder = field(default_factory=MetricsRecorder)
    social_graph: SocialGraph = field(default_factory=SocialGraph)
    interaction_log: list[InteractionOutcome] = field(default_factory=list)
    communication_log: list[JSONDict] = field(default_factory=list)
    observatory: ObservatoryRecorder = field(default_factory=ObservatoryRecorder)
    propagation_log: PropagationLog = field(default_factory=PropagationLog)
    propagation_config: PropagationConfig = field(default_factory=PropagationConfig)

    @classmethod
    def create(cls, config: SimulationConfig | None = None) -> Simulation:
        config = config or SimulationConfig()
        rng = ExperimentRNG.from_seed(config.seed)
        grid = Grid.empty(config.width, config.height, bordered=config.bordered)
        world = World(
            grid=grid,
            resource_energy=config.resource_energy,
            resource_regen_interval=config.resource_regen_interval,
        )

        free = grid.free_positions()
        rng.shuffle(free)
        needed = config.n_agents + config.n_resources
        if needed > len(free):
            raise ValueError(
                f"Need {needed} free cells for agents/resources; only {len(free)} available"
            )

        agents: dict[str, Agent] = {}
        for i in range(config.n_agents):
            agent_id = f"agent_{i:03d}"
            agent = create_agent(
                config.agent_type,
                agent_id,
                energy=config.initial_energy,
                perception_radius=config.perception_radius,
            )
            agent.state.energy_cost_per_tick = config.energy_cost_per_tick
            world.place_agent(agent_id, free[i])
            agents[agent_id] = agent

        resource_start = config.n_agents
        for offset in range(config.n_resources):
            world.place_resource(free[resource_start + offset])

        sim = cls(config=config, world=world, agents=agents, rng=rng)
        sim.observatory.record_births(sim)
        return sim

    def active_agent_ids(self) -> list[str]:
        return sorted(self.world.agent_positions.keys())

    def _resolve_social_interactions(
        self,
        actions: dict[str, Action],
        positions: dict[str, Position],
    ) -> list[InteractionOutcome]:
        """Resolve pairwise social actions for agents adjacent at decision time."""
        outcomes: list[InteractionOutcome] = []
        ids = sorted(positions.keys())
        handled: set[tuple[str, str]] = set()

        for i, a in enumerate(ids):
            pa = positions[a]
            for b in ids[i + 1 :]:
                pb = positions[b]
                if pa.manhattan(pb) > 1:
                    continue
                key = (a, b)
                if key in handled:
                    continue

                intent_a = ACTION_TO_INTERACTION.get(actions.get(a, Action.STAY))
                intent_b = ACTION_TO_INTERACTION.get(actions.get(b, Action.STAY))
                if actions.get(a) == Action.INTERACT:
                    intent_a = intent_a or ACTION_TO_INTERACTION[Action.COOPERATE]
                if actions.get(b) == Action.INTERACT:
                    intent_b = intent_b or ACTION_TO_INTERACTION[Action.COOPERATE]

                if intent_a is None and intent_b is None:
                    continue

                outcome = resolve_pair(
                    tick=self.world.tick,
                    agent_a=a,
                    agent_b=b,
                    intent_a=intent_a,
                    intent_b=intent_b,
                    energy_a=self.agents[a].state.energy,
                    energy_b=self.agents[b].state.energy,
                )
                handled.add(key)
                outcomes.append(outcome)

                for agent_id in (a, b):
                    agent = self.agents[agent_id]
                    if hasattr(agent, "apply_social_outcome"):
                        agent.apply_social_outcome(outcome)
                    else:
                        _, energy_delta, _, _ = outcome.for_agent(agent_id)
                        if energy_delta:
                            agent.apply_energy_gain(energy_delta)

                # Symbolic message exchange when communicate was selected.
                if "communicate" in outcome.resolved_kind:
                    cost_a = 0.5
                    cost_b = 0.5
                    if hasattr(self.agents[a], "social_config"):
                        cost_a = float(
                            getattr(
                                self.agents[a].social_config,
                                "communication_energy_cost",
                                0.5,
                            )
                        )
                    if hasattr(self.agents[b], "social_config"):
                        cost_b = float(
                            getattr(
                                self.agents[b].social_config,
                                "communication_energy_cost",
                                0.5,
                            )
                        )
                    max_tokens = 2
                    if hasattr(self.agents[a], "social_config"):
                        max_tokens = int(
                            getattr(self.agents[a].social_config, "max_message_tokens", 2)
                        )
                    events = exchange_communication(
                        self.agents[a],
                        self.agents[b],
                        intent_a=intent_a,
                        intent_b=intent_b,
                        tick=self.world.tick,
                        positions=positions,
                        max_tokens=max_tokens,
                        energy_cost=max(cost_a, cost_b),
                    )
                    message_dicts = [e.to_dict() for e in events]
                    outcome.messages = message_dicts
                    self.communication_log.extend(message_dicts)

                # Indirect association / secondhand transfer via shared absent contacts.
                absent_ids = set(self.world.removed_agents.keys())
                if absent_ids:
                    propagate_after_interaction(
                        self.agents[a],
                        self.agents[b],
                        outcome,
                        absent_ids=absent_ids,
                        config=self.propagation_config,
                        log=self.propagation_log,
                    )

        return outcomes

    def step(self) -> None:
        """Advance one simulation tick following the world-spec order."""
        positions_before = dict(self.world.agent_positions)
        self.observatory.on_step_begin(self)
        comm_before = len(self.communication_log)

        # 1–2. Observe
        observations: dict[str, Observation] = {}
        for agent_id in self.active_agent_ids():
            observations[agent_id] = self.agents[agent_id].observe(self.world)
        self.observatory.on_observations(self, observations)

        # 3–4. Choose actions
        actions: dict[str, Action] = {}
        for agent_id in self.active_agent_ids():
            agent = self.agents[agent_id]
            action = agent.select_action(observations[agent_id], self.rng)
            agent.last_action = action
            actions[agent_id] = action
        self.observatory.on_actions(self, actions, observations)

        nearby_pairs = self._count_nearby_pairs()

        # 5. Apply movement actions
        moved: dict[str, bool] = {}
        for agent_id in self.active_agent_ids():
            action = actions[agent_id]
            if action in SOCIAL_ACTIONS or action == Action.INTERACT:
                moved[agent_id] = False
                continue
            did_move = self.world.apply_action(agent_id, action)
            moved[agent_id] = did_move

        # 5b. Resolve social interactions using pre-move adjacency (A↔B)
        social_outcomes = self._resolve_social_interactions(actions, positions_before)
        self.interaction_log.extend(social_outcomes)
        self.observatory.on_social_outcomes(self, social_outcomes)
        if len(self.communication_log) > comm_before:
            self.observatory.on_communication_events(
                self, self.communication_log[comm_before:]
            )

        # 6. World dynamics: collect then regenerate, then advance tick
        energy_gains = self.world.collect_resources()
        for agent_id, amount in energy_gains.items():
            self.agents[agent_id].apply_energy_gain(amount)
        self.observatory.on_resource_collected(self, energy_gains)

        self.world.advance_tick()
        self.world.regenerate_resources()

        # Verify communicated claims against the post-move / post-collect world.
        verify_start = len(self.communication_log)
        for agent_id in self.active_agent_ids():
            agent = self.agents[agent_id]
            if hasattr(agent, "pending_claims") and agent.pending_claims:
                for event in verify_pending_claims(agent, self.world):
                    self.communication_log.append(event.to_dict())
        if len(self.communication_log) > verify_start:
            self.observatory.on_communication_events(
                self, self.communication_log[verify_start:]
            )

        # Sync population social graph from relationship stores.
        self.social_graph.sync_population(self.agents)

        # Post-action observations for learning (no second cognitive update)
        next_observations: dict[str, Observation] = {}
        for agent_id in self.active_agent_ids():
            next_observations[agent_id] = self.agents[agent_id].observe(
                self.world, cognitive=False
            )

        # 7. Learning
        rewards: dict[str, float] = {}
        motivation = MotivationConfig()
        social_bonus = {
            outcome.agent_a: outcome.valence_a * 0.3 + outcome.energy_delta_a / 25.0
            for outcome in social_outcomes
        }
        for outcome in social_outcomes:
            social_bonus[outcome.agent_b] = (
                social_bonus.get(outcome.agent_b, 0.0)
                + outcome.valence_b * 0.3
                + outcome.energy_delta_b / 25.0
            )

        for agent_id in self.active_agent_ids():
            agent = self.agents[agent_id]
            action = actions[agent_id]
            obs_before = observations[agent_id]
            obs_after = next_observations[agent_id]
            gain = energy_gains.get(agent_id, 0.0)
            did_move = moved.get(agent_id, False)
            blocked = (
                action != Action.STAY
                and action not in SOCIAL_ACTIONS
                and action != Action.INTERACT
                and not did_move
                and positions_before.get(agent_id) == self.world.agent_positions.get(agent_id)
            )
            before_visits = getattr(agent, "visit_counts", None)
            if before_visits is not None:
                new_cell = before_visits[obs_after.position.as_tuple()] == 0
            else:
                new_cell = obs_before.position != obs_after.position

            agent_motivation = getattr(agent, "motivation_config", motivation)
            reward = compute_reward_with_blocked(
                obs_before,
                obs_after,
                energy_gained=gain,
                moved=did_move,
                new_cell=new_cell,
                blocked=blocked,
                config=agent_motivation,
            )
            reward += social_bonus.get(agent_id, 0.0)
            agent.learn(
                obs_before,
                action,
                reward=reward,
                next_observation=obs_after,
                energy_gained=gain,
                moved=did_move,
                blocked=blocked,
            )
            rewards[agent_id] = reward
            agent.update_after_tick()

        # Computational-loss update for survivors (historical entities remain represented).
        active_set = set(self.active_agent_ids())
        for agent_id in self.active_agent_ids():
            agent = self.agents[agent_id]
            if hasattr(agent, "update_loss_tracker"):
                agent.update_loss_tracker(tick=self.world.tick, active_ids=active_set)

        self.observatory.on_prediction_errors(self)
        self.observatory.on_affect(self)

        # 8. Metrics
        energies = {
            agent_id: self.agents[agent_id].state.energy
            for agent_id in self.world.agent_positions
        }
        self.metrics.record_tick(
            tick=self.world.tick,
            positions=dict(self.world.agent_positions),
            actions=actions,
            energies=energies,
            interactions=nearby_pairs + len(social_outcomes),
            rewards=rewards,
            resources_collected=len(energy_gains),
        )

    def run(self, n_ticks: int) -> MetricsRecorder:
        for _ in range(n_ticks):
            self.step()
        return self.metrics

    def disappear(self, agent_id: str, *, reversible: bool = True) -> None:
        self.world.remove_agent(agent_id, reversible=reversible)
        absent = {agent_id}
        tick = self.world.tick
        for other_id, agent in self.agents.items():
            if other_id == agent_id:
                continue
            if hasattr(agent, "mark_absent_communicators"):
                agent.mark_absent_communicators(absent)
            # Snapshot pre-loss relationship/memory/prediction; do not wipe internal models.
            if hasattr(agent, "loss_tracker"):
                knew = (
                    (hasattr(agent, "relationships") and agent.relationships.get(agent_id) is not None)
                    or (hasattr(agent, "ltm") and agent.ltm.get(agent_id) is not None)
                    or (
                        hasattr(agent, "world_model")
                        and agent_id in getattr(agent.world_model.agents, "entities", {})
                    )
                )
                if knew:
                    agent.loss_tracker.record_disappearance(agent, agent_id, tick=tick, sim=self)
        self.observatory.on_disappear(self, agent_id)

    def restore(self, agent_id: str, position: Position | None = None) -> Position:
        return self.world.restore_agent(agent_id, position)

    def _count_nearby_pairs(self, radius: int = 1) -> int:
        ids = self.active_agent_ids()
        count = 0
        for i, a in enumerate(ids):
            pa = self.world.agent_positions[a]
            for b in ids[i + 1 :]:
                pb = self.world.agent_positions[b]
                if pa.manhattan(pb) <= radius:
                    count += 1
        return count

    def recent_communications(self, n: int = 8) -> list[JSONDict]:
        return [e for e in self.communication_log if e.get("event") == "communication"][-n:]

    def render(self) -> str:
        grid = self.world.render_ascii()
        lines = [f"[TICK {self.world.tick}]", grid]
        recent = self.recent_communications(6)
        if recent:
            lines.append("")
            for event in recent:
                tokens = " ".join(event.get("tokens") or [])
                lines.append(
                    f"{event.get('sender')}: {tokens} → {event.get('receiver')}"
                )
        return "\n".join(lines)

    def agent_learning_summary(self) -> JSONDict:
        summary: JSONDict = {}
        for agent_id, agent in self.agents.items():
            entry: JSONDict = {
                "energy": agent.state.energy,
                "total_reward": agent.total_reward,
                "resources_collected": agent.resources_collected,
                "age": agent.state.age,
            }
            if hasattr(agent, "epsilon"):
                entry["epsilon"] = agent.epsilon
            if hasattr(agent, "update_count"):
                entry["update_count"] = agent.update_count
            if hasattr(agent, "q_values"):
                entry["q_states"] = len(agent.q_values)
            if hasattr(agent, "memory_summary"):
                entry["memory"] = agent.memory_summary()
            if hasattr(agent, "relationship_summary"):
                entry["social"] = agent.relationship_summary()
            if hasattr(agent, "world_model_summary"):
                entry["world_model"] = agent.world_model_summary()
            summary[agent_id] = entry
        return summary

    def save(self, path: str | Path) -> Path:
        data = snapshot_to_dict(
            world=self.world,
            agents=self.agents,
            seed=self.config.seed,
            metrics=self.metrics,
            config=self.config.to_dict(),
        )
        data["social_graph"] = self.social_graph.to_dict()
        data["interaction_log"] = [outcome.to_dict() for outcome in self.interaction_log[-500:]]
        data["communication_log"] = list(self.communication_log[-500:])
        data["observatory"] = self.observatory.to_dict()
        data["propagation_log"] = self.propagation_log.to_dict()
        return save_snapshot(path, data)

    @classmethod
    def load(cls, path: str | Path) -> Simulation:
        data = load_snapshot(path)
        config = SimulationConfig.from_dict(data.get("config") or {"seed": data["seed"]})
        world = restore_world(data)
        agents = restore_agents(data)
        metrics = restore_metrics(data) or MetricsRecorder()
        rng = ExperimentRNG.from_seed(int(data["seed"]))
        graph = SocialGraph.from_dict(data.get("social_graph", {}))
        sim = cls(
            config=config,
            world=world,
            agents=agents,
            rng=rng,
            metrics=metrics,
            social_graph=graph,
        )
        sim.communication_log = list(data.get("communication_log", []))
        if data.get("observatory"):
            sim.observatory = ObservatoryRecorder.from_dict(data["observatory"])
        return sim