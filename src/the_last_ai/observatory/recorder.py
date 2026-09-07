"""Record significant agent experiences into life histories during simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

from the_last_ai.observatory.affect import (
    AffectiveState,
    affect_label,
    compute_affect,
    significant_affect_delta,
)
from the_last_ai.observatory.event import LifeEvent, LifeEventType
from the_last_ai.observatory.history import AgentLifeHistory
from the_last_ai.social.types import InteractionKind
from the_last_ai.types import Action, JSONDict, SOCIAL_ACTION_VALUES


@dataclass
class ObservatoryRecorder:
    """
    Instrumentation layer. Does not alter cognition — only observes and records.

    full_history_ids: agents that keep dense histories (survivor focus).
    Others get lighter recording (still capture social/comm/affect changes).
    If empty, all agents get full detail.
    """

    histories: dict[str, AgentLifeHistory] = field(default_factory=dict)
    full_history_ids: set[str] = field(default_factory=set)
    enabled: bool = True
    affect_sample_interval: int = 5
    _last_affect: dict[str, AffectiveState] = field(default_factory=dict)
    _seen_resources: dict[str, set[tuple[int, int]]] = field(default_factory=dict)
    _last_population: int | None = None

    def ensure(self, agent_id: str, *, tick: int = 0) -> AgentLifeHistory:
        if agent_id not in self.histories:
            full = (not self.full_history_ids) or (agent_id in self.full_history_ids)
            hist = AgentLifeHistory(agent_id=agent_id, birth_tick=tick, full_detail=full)
            if not full:
                hist.max_events = 800
            self.histories[agent_id] = hist
        return self.histories[agent_id]

    def record_births(self, sim) -> None:
        if not self.enabled:
            return
        tick = sim.world.tick
        for agent_id in sim.agents:
            hist = self.ensure(agent_id, tick=tick)
            if any(e.event_type == LifeEventType.BIRTH for e in hist.events):
                continue
            hist.birth_tick = tick
            hist.record(
                LifeEvent(
                    tick=tick,
                    agent_id=agent_id,
                    event_type=LifeEventType.BIRTH,
                    summary="Born.",
                    context={"population": len(sim.active_agent_ids())},
                )
            )
            state, factors = compute_affect(sim.agents[agent_id], active_ids=set(sim.active_agent_ids()))
            hist.record_affect(tick, state)
            self._last_affect[agent_id] = state
            hist.record(
                LifeEvent(
                    tick=tick,
                    agent_id=agent_id,
                    event_type=LifeEventType.AFFECTIVE_CHANGE,
                    state_after=state.to_dict(),
                    context={"factors": factors, "label": affect_label(state)[0], "initial": True},
                    summary=f"Initial affective state: {affect_label(state)[0]}.",
                )
            )
        self._last_population = len(sim.active_agent_ids())

    def focus(self, agent_ids: set[str] | list[str]) -> None:
        self.full_history_ids = set(agent_ids)
        for aid, hist in self.histories.items():
            hist.full_detail = aid in self.full_history_ids or not self.full_history_ids

    def _is_dense(self, agent_id: str) -> bool:
        return (not self.full_history_ids) or (agent_id in self.full_history_ids)

    def on_step_begin(self, sim) -> None:
        if not self.enabled:
            return
        pop = len(sim.active_agent_ids())
        if self._last_population is not None and pop != self._last_population:
            for agent_id in sim.active_agent_ids():
                hist = self.ensure(agent_id, tick=sim.world.tick)
                hist.record(
                    LifeEvent(
                        tick=sim.world.tick,
                        agent_id=agent_id,
                        event_type=LifeEventType.POPULATION_CHANGE,
                        context={
                            "population_before": self._last_population,
                            "population_after": pop,
                        },
                        summary=f"Population changed {self._last_population} → {pop}.",
                    )
                )
        self._last_population = pop

    def on_observations(self, sim, observations: dict) -> None:
        if not self.enabled:
            return
        active = set(sim.active_agent_ids())
        tick = sim.world.tick
        for agent_id, obs in observations.items():
            agent = sim.agents[agent_id]
            hist = self.ensure(agent_id, tick=tick)
            # New entity encounters
            for visible in obs.visible_agents:
                oid = visible.agent_id
                if oid not in hist.known_entities:
                    hist.known_entities.add(oid)
                    hist.record(
                        LifeEvent(
                            tick=tick,
                            agent_id=agent_id,
                            event_type=LifeEventType.ENTITY_ENCOUNTERED,
                            context={
                                "other_id": oid,
                                "distance": visible.distance,
                                "position": list(
                                    obs.position.offset(*visible.relative_position).as_tuple()
                                ),
                            },
                            summary=f"Encountered {oid}.",
                        )
                    )
            # Resources discovered (first time at cell)
            seen = self._seen_resources.setdefault(agent_id, set())
            from the_last_ai.types import CellType

            radius = obs.perception_radius
            for dy, row in enumerate(obs.local_cells):
                for dx, cell in enumerate(row):
                    if cell != CellType.RESOURCE:
                        continue
                    abs_pos = obs.position.offset(dx - radius, dy - radius).as_tuple()
                    if abs_pos not in seen:
                        seen.add(abs_pos)
                        hist.record(
                            LifeEvent(
                                tick=tick,
                                agent_id=agent_id,
                                event_type=LifeEventType.RESOURCE_DISCOVERED,
                                context={"position": list(abs_pos)},
                                summary=f"Resource discovered at {list(abs_pos)}.",
                            )
                        )
            # Expected entity absent (memory seek targets not visible)
            if hasattr(agent, "last_retrieved_ids") and agent.last_retrieved_ids:
                visible_ids = {v.agent_id for v in obs.visible_agents}
                for eid in agent.last_retrieved_ids[:3]:
                    if eid not in visible_ids and eid not in active:
                        mem = agent.ltm.get(eid) if hasattr(agent, "ltm") else None
                        pe = (
                            float(agent.world_model.mean_error_ema)
                            if hasattr(agent, "world_model")
                            else None
                        )
                        # Record at most once per entity per 10 ticks to avoid spam
                        recent = [
                            e
                            for e in hist.events[-20:]
                            if e.event_type == LifeEventType.ENTITY_ABSENT_EXPECTED
                            and e.context.get("other_id") == eid
                        ]
                        if recent and tick - recent[-1].tick < 10:
                            continue
                        hist.record(
                            LifeEvent(
                                tick=tick,
                                agent_id=agent_id,
                                event_type=LifeEventType.ENTITY_ABSENT_EXPECTED,
                                context={
                                    "other_id": eid,
                                    "observed": "absent",
                                    "prediction_error": pe,
                                    "memory_strength": (
                                        float(mem.strength_at(tick, agent.ltm.config.decay_lambda))
                                        if mem is not None
                                        else None
                                    ),
                                    "expected_location": (
                                        list(mem.expected_location) if mem and mem.expected_location else None
                                    ),
                                },
                                summary=f"Expected entity {eid} absent.",
                            )
                        )
                        if mem is not None:
                            hist.record(
                                LifeEvent(
                                    tick=tick,
                                    agent_id=agent_id,
                                    event_type=LifeEventType.MEMORY_ACTIVATED,
                                    context={
                                        "other_id": eid,
                                        "strength": float(
                                            mem.strength_at(tick, agent.ltm.config.decay_lambda)
                                        ),
                                    },
                                    summary=f"Memory of {eid} activated.",
                                )
                            )

    def on_actions(self, sim, actions: dict, observations: dict) -> None:
        if not self.enabled:
            return
        tick = sim.world.tick
        active = set(sim.active_agent_ids())
        for agent_id, action in actions.items():
            agent = sim.agents[agent_id]
            hist = self.ensure(agent_id, tick=tick)
            dense = self._is_dense(agent_id)
            notable = (
                action in SOCIAL_ACTION_VALUES
                or action == Action.INTERACT
                or getattr(agent, "investigation_actions", 0) > 0
                and getattr(agent, "pending_investigation", None) is not None
            )
            # Decision record for notable / dense agents
            if notable or (dense and tick % 10 == 0):
                info = self._decision_information(agent, observations.get(agent_id), active)
                decision = {
                    "tick": tick,
                    "selected_action": action.value,
                    "available_information": info,
                }
                hist.decisions.append(decision)
                if len(hist.decisions) > 500:
                    hist.decisions = hist.decisions[-500:]
                if notable:
                    hist.record(
                        LifeEvent(
                            tick=tick,
                            agent_id=agent_id,
                            event_type=LifeEventType.DECISION,
                            context=decision,
                            summary=f"Selected action: {action.value}.",
                        )
                    )
            if action in SOCIAL_ACTION_VALUES or (dense and action != Action.STAY):
                hist.record(
                    LifeEvent(
                        tick=tick,
                        agent_id=agent_id,
                        event_type=LifeEventType.ACTION,
                        context={"action": action.value},
                        summary=f"Action: {action.value}.",
                    )
                )
            if getattr(agent, "pending_investigation", None) is not None and action.value.startswith(
                "move_"
            ):
                hist.record(
                    LifeEvent(
                        tick=tick,
                        agent_id=agent_id,
                        event_type=LifeEventType.INVESTIGATION,
                        context={
                            "target": list(agent.pending_investigation.as_tuple()),
                            "source": getattr(agent, "investigation_source", None),
                            "priority": getattr(agent, "investigation_priority", 0.0),
                        },
                        summary="Investigation movement toward communicated/claimed location.",
                    )
                )

    def _decision_information(self, agent, observation, active_ids: set[str]) -> JSONDict:
        info: JSONDict = {}
        if observation is not None:
            from the_last_ai.types import CellType

            info["nearby_resources"] = any(
                cell == CellType.RESOURCE for row in observation.local_cells for cell in row
            )
            info["visible_agents"] = [v.agent_id for v in observation.visible_agents]
            info["energy"] = observation.energy
        if hasattr(agent, "last_retrieved_ids"):
            info["remembered_entities"] = list(agent.last_retrieved_ids[:5])
        if hasattr(agent, "world_model"):
            info["prediction_error"] = float(agent.world_model.mean_error_ema)
        if hasattr(agent, "last_motivation") and agent.last_motivation:
            info["social_drive"] = float(agent.last_motivation.social_drive)
        if getattr(agent, "last_received_message", None):
            info["recent_communication"] = agent.last_received_message
        info["active_population"] = sorted(active_ids)[:20]
        return info

    def on_social_outcomes(self, sim, outcomes: list) -> None:
        if not self.enabled:
            return
        tick = sim.world.tick
        for outcome in outcomes:
            for agent_id in (outcome.agent_a, outcome.agent_b):
                if agent_id not in sim.agents:
                    continue
                other, energy_delta, valence, intent = outcome.for_agent(agent_id)
                hist = self.ensure(agent_id, tick=tick)
                hist.known_entities.add(other)
                traj = hist.get_or_create_traj(other)
                if traj.first_interaction_tick is None:
                    traj.first_interaction_tick = tick
                traj.last_interaction_tick = tick
                traj.interaction_count += 1
                if intent == InteractionKind.COOPERATE:
                    traj.cooperation_count += 1
                elif intent == InteractionKind.COMPETE:
                    traj.competition_count += 1
                elif intent == InteractionKind.COMMUNICATE:
                    traj.communication_count += 1
                agent = sim.agents[agent_id]
                rel = agent.relationships.get(other) if hasattr(agent, "relationships") else None
                if rel is not None:
                    traj.snapshot(
                        tick=tick,
                        trust=rel.trust,
                        strength=rel.strength,
                        utility=rel.predicted_utility,
                        reliability=getattr(rel, "information_reliability", None),
                    )
                    traj.successful_messages = rel.successful_messages
                    traj.failed_messages = rel.failed_messages
                hist.record(
                    LifeEvent(
                        tick=tick,
                        agent_id=agent_id,
                        event_type=LifeEventType.SOCIAL_INTERACTION,
                        context={
                            "other_id": other,
                            "intent": intent.value if intent else None,
                            "valence": valence,
                            "energy_delta": energy_delta,
                            "resolved_kind": outcome.resolved_kind,
                        },
                        summary=f"Social interaction with {other} ({outcome.resolved_kind}).",
                    )
                )
                if rel is not None:
                    hist.record(
                        LifeEvent(
                            tick=tick,
                            agent_id=agent_id,
                            event_type=LifeEventType.RELATIONSHIP_UPDATED,
                            context={
                                "other_id": other,
                                "trust": rel.trust,
                                "strength": rel.strength,
                                "predicted_utility": rel.predicted_utility,
                                "information_reliability": getattr(
                                    rel, "information_reliability", 0.5
                                ),
                            },
                            summary=f"Relationship with {other} updated.",
                        )
                    )

    def on_communication_events(self, sim, events: list[JSONDict]) -> None:
        if not self.enabled:
            return
        for event in events:
            etype = event.get("event")
            tick = int(event.get("tick", sim.world.tick))
            sender = event.get("sender")
            receiver = event.get("receiver")
            tokens = list(event.get("tokens") or [])
            if etype == "communication":
                for agent_id, role in ((sender, "sent"), (receiver, "received")):
                    if agent_id not in sim.agents and agent_id not in self.histories:
                        continue
                    hist = self.ensure(str(agent_id), tick=tick)
                    msg = {
                        **event,
                        "role_for_agent": role,
                        "tokens": tokens,  # exact tokens preserved
                    }
                    hist.messages.append(msg)
                    if len(hist.messages) > 1000:
                        hist.messages = hist.messages[-1000:]
                    kind = (
                        LifeEventType.COMMUNICATION_SENT
                        if role == "sent"
                        else LifeEventType.COMMUNICATION_RECEIVED
                    )
                    hist.record(
                        LifeEvent(
                            tick=tick,
                            agent_id=str(agent_id),
                            event_type=kind,
                            context=msg,
                            summary=(
                                f'Communicated {tokens!r} → {receiver}.'
                                if role == "sent"
                                else f'Received {tokens!r} from {sender}.'
                            ),
                        )
                    )
                    other = receiver if role == "sent" else sender
                    if other:
                        traj = hist.get_or_create_traj(str(other))
                        traj.communication_count += 1
            elif etype == "communication_verification":
                receiver = str(event.get("receiver"))
                sender = str(event.get("sender"))
                hist = self.ensure(receiver, tick=tick)
                hist.record(
                    LifeEvent(
                        tick=tick,
                        agent_id=receiver,
                        event_type=LifeEventType.COMMUNICATION_VERIFIED,
                        context=dict(event),
                        summary=(
                            f"Communication from {sender} verified: "
                            f"{'SUCCESS' if event.get('verified') else 'FAILURE'} "
                            f"({event.get('outcome')})."
                        ),
                    )
                )
                traj = hist.get_or_create_traj(sender)
                traj.verification_events.append(dict(event))
                if event.get("verified"):
                    traj.successful_messages += 1
                else:
                    traj.failed_messages += 1
                if receiver in sim.agents:
                    rel = sim.agents[receiver].relationships.get(sender)
                    if rel is not None:
                        traj.snapshot(
                            tick=tick,
                            trust=rel.trust,
                            strength=rel.strength,
                            utility=rel.predicted_utility,
                            reliability=rel.information_reliability,
                        )

    def on_resource_collected(self, sim, energy_gains: dict[str, float]) -> None:
        if not self.enabled:
            return
        tick = sim.world.tick
        for agent_id, amount in energy_gains.items():
            if amount <= 0:
                continue
            hist = self.ensure(agent_id, tick=tick)
            pos = sim.world.agent_positions.get(agent_id)
            hist.record(
                LifeEvent(
                    tick=tick,
                    agent_id=agent_id,
                    event_type=LifeEventType.RESOURCE_COLLECTED,
                    context={
                        "energy_gained": amount,
                        "position": list(pos.as_tuple()) if pos else None,
                    },
                    summary=f"Collected resource (+{amount:.1f} energy).",
                )
            )

    def on_prediction_errors(self, sim) -> None:
        if not self.enabled:
            return
        tick = sim.world.tick
        for agent_id in sim.active_agent_ids():
            agent = sim.agents[agent_id]
            if not hasattr(agent, "world_model"):
                continue
            pe = float(agent.world_model.mean_error_ema)
            # Record spikes only
            hist = self.ensure(agent_id, tick=tick)
            prev = None
            for e in reversed(hist.events[-15:]):
                if e.event_type == LifeEventType.PREDICTION_ERROR:
                    prev = e.context.get("prediction_error_ema")
                    break
            if prev is None or abs(pe - float(prev)) >= 0.08 or pe >= 0.25:
                if prev is not None and abs(pe - float(prev)) < 0.08 and pe < 0.25:
                    continue
                hist.record(
                    LifeEvent(
                        tick=tick,
                        agent_id=agent_id,
                        event_type=LifeEventType.PREDICTION_ERROR,
                        context={"prediction_error_ema": pe},
                        summary=f"Prediction error EMA {pe:.3f}.",
                    )
                )

    def on_affect(self, sim) -> None:
        if not self.enabled:
            return
        tick = sim.world.tick
        active = set(sim.active_agent_ids())
        for agent_id in sim.active_agent_ids():
            agent = sim.agents[agent_id]
            state, factors = compute_affect(agent, active_ids=active)
            hist = self.ensure(agent_id, tick=tick)
            sample = tick % self.affect_sample_interval == 0 or self._is_dense(agent_id)
            prev = self._last_affect.get(agent_id)
            changed = prev is None or significant_affect_delta(prev, state)
            if sample:
                hist.record_affect(tick, state)
            if changed and prev is not None:
                label, reasons = affect_label(state)
                hist.record(
                    LifeEvent(
                        tick=tick,
                        agent_id=agent_id,
                        event_type=LifeEventType.AFFECTIVE_CHANGE,
                        state_before=prev.to_dict(),
                        state_after=state.to_dict(),
                        context={
                            "factors": factors,
                            "label": label,
                            "reasons": reasons,
                            "contributing_recent_events": [
                                e.to_dict()
                                for e in hist.events[-8:]
                                if e.event_type
                                in {
                                    LifeEventType.ENTITY_ABSENT_EXPECTED,
                                    LifeEventType.DISAPPEARANCE_OTHER,
                                    LifeEventType.COMMUNICATION_VERIFIED,
                                    LifeEventType.PREDICTION_ERROR,
                                    LifeEventType.SOCIAL_INTERACTION,
                                }
                            ],
                        },
                        summary=f"Affective state changed → {label}.",
                    )
                )
            self._last_affect[agent_id] = state

    def on_disappear(self, sim, disappeared_id: str) -> None:
        if not self.enabled:
            return
        tick = sim.world.tick
        # Record for the disappeared agent
        if disappeared_id in self.histories or disappeared_id in sim.agents:
            hist = self.ensure(disappeared_id, tick=tick)
            hist.record(
                LifeEvent(
                    tick=tick,
                    agent_id=disappeared_id,
                    event_type=LifeEventType.DISAPPEARANCE_SELF,
                    summary="This agent disappeared from the active population.",
                    context={"population_after": len(sim.active_agent_ids())},
                )
            )
        # Record for survivors who knew them
        for agent_id in list(sim.agents.keys()):
            hist = self.ensure(agent_id, tick=tick)
            if disappeared_id in hist.known_entities or (
                hasattr(sim.agents[agent_id], "relationships")
                and sim.agents[agent_id].relationships.get(disappeared_id) is not None
            ):
                hist.mark_historical({disappeared_id})
                loss_ctx: JSONDict = {
                    "other_id": disappeared_id,
                    "status": "HISTORICAL — AGENT NO LONGER PRESENT",
                }
                agent = sim.agents[agent_id]
                if hasattr(agent, "loss_tracker"):
                    rec = agent.loss_tracker.get(disappeared_id)
                    if rec is not None:
                        loss_ctx["loss_record"] = {
                            "significance": rec.significance,
                            "social_loss": rec.social_loss,
                            "presence_expectation": rec.presence_expectation_at_loss,
                            "relationship_before": rec.relationship_before,
                            "memory_before": rec.memory_before,
                        }
                hist.record(
                    LifeEvent(
                        tick=tick,
                        agent_id=agent_id,
                        event_type=LifeEventType.DISAPPEARANCE_OTHER,
                        context=loss_ctx,
                        summary=f"{disappeared_id} no longer present (historical).",
                    )
                )

    def on_collapse_stage(self, sim, *, survivor_id: str, population: int, label: str) -> None:
        if not self.enabled:
            return
        tick = sim.world.tick
        self.focus({survivor_id})
        hist = self.ensure(survivor_id, tick=tick)
        hist.record(
            LifeEvent(
                tick=tick,
                agent_id=survivor_id,
                event_type=LifeEventType.COLLAPSE_STAGE,
                context={"population": population, "label": label},
                summary=f"Collapse stage: population {population} ({label}).",
            )
        )

    def finalize_survivor(self, sim, survivor_id: str) -> None:
        if not self.enabled:
            return
        tick = sim.world.tick
        active = set(sim.active_agent_ids())
        hist = self.ensure(survivor_id, tick=tick)
        agent = sim.agents[survivor_id]
        state, factors = compute_affect(agent, active_ids=active)
        label, reasons = affect_label(state)
        hist.record_affect(tick, state)
        hist.record(
            LifeEvent(
                tick=tick,
                agent_id=survivor_id,
                event_type=LifeEventType.FINAL_STATE,
                state_after=state.to_dict(),
                context={
                    "population": len(active),
                    "affect_label": label,
                    "affect_reasons": reasons,
                    "factors": factors,
                    "overview": hist.overview(active_ids=active, final_tick=tick),
                },
                summary="Final survivor state recorded.",
            )
        )

    def to_dict(self) -> JSONDict:
        return {
            "enabled": self.enabled,
            "full_history_ids": sorted(self.full_history_ids),
            "affect_sample_interval": self.affect_sample_interval,
            "histories": {aid: h.to_dict() for aid, h in self.histories.items()},
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> ObservatoryRecorder:
        rec = cls(
            enabled=bool(data.get("enabled", True)),
            full_history_ids=set(data.get("full_history_ids") or []),
            affect_sample_interval=int(data.get("affect_sample_interval", 5)),
        )
        for aid, hd in (data.get("histories") or {}).items():
            rec.histories[aid] = AgentLifeHistory.from_dict(hd)
            if rec.histories[aid].affect_series:
                rec._last_affect[aid] = rec.histories[aid].affect_series[-1][1]
        return rec
