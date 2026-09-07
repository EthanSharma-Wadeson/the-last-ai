"""Per-agent tracker for computational responses to entity disappearance."""

from __future__ import annotations

from dataclasses import dataclass, field

from the_last_ai.loss.metrics import AdaptationConfig, assess_adaptation
from the_last_ai.loss.model import (
    EntityLossRecord,
    EntityStatus,
    aggregate_social_loss,
    compute_entity_significance,
    compute_entity_social_loss,
)
from the_last_ai.types import JSONDict


@dataclass
class LossTracker:
    """
    Tracks ACTIVE / HISTORICAL / FORGOTTEN entities and loss trajectories.

    Attached to SocialAgent (and subclasses). Does not inject grief —
    updates from relationship, memory, prediction, and search observations.
    """

    records: dict[str, EntityLossRecord] = field(default_factory=dict)
    aggregate_social_loss: float = 0.0
    dimension_changes: list[JSONDict] = field(default_factory=list)
    adaptation_config: AdaptationConfig = field(default_factory=AdaptationConfig)
    forget_memory_threshold: float = 0.02
    sample_interval: int = 1

    def get(self, entity_id: str) -> EntityLossRecord | None:
        return self.records.get(entity_id)

    def ensure_active(self, entity_id: str) -> EntityLossRecord:
        if entity_id not in self.records:
            self.records[entity_id] = EntityLossRecord(
                entity_id=entity_id, status=EntityStatus.ACTIVE
            )
        return self.records[entity_id]

    def record_disappearance(self, agent, entity_id: str, *, tick: int, sim=None) -> EntityLossRecord:
        """Snapshot pre-loss state and open a HISTORICAL loss record."""
        rel = None
        if hasattr(agent, "relationships"):
            rel = agent.relationships.get(entity_id)
        mem = agent.ltm.get(entity_id) if hasattr(agent, "ltm") else None
        mem_strength = (
            float(mem.strength_at(tick, agent.ltm.config.decay_lambda)) if mem is not None else 0.0
        )
        familiarity = float(mem.familiarity) if mem is not None else 0.0

        strength = float(rel.strength) if rel else 0.0
        trust = float(rel.trust) if rel else 0.5
        interactions = int(rel.interaction_count) if rel else 0
        communicate = int(rel.communicate_count) if rel else 0
        utility = float(rel.predicted_utility) if rel else 0.0
        reliability = float(getattr(rel, "information_reliability", 0.5)) if rel else 0.5

        presence_expectation = 0.0
        pred_before: JSONDict = {}
        if hasattr(agent, "world_model"):
            entities = getattr(getattr(agent.world_model, "agents", None), "entities", {}) or {}
            entity_model = entities.get(entity_id)
            if entity_model is not None:
                conf = float(entity_model.confidence())
                total_pres = entity_model.presence_when_expected + entity_model.absence_when_expected
                if total_pres > 0:
                    rate = entity_model.presence_when_expected / total_pres
                else:
                    rate = 0.7 if entity_model.observations > 0 else 0.3
                presence_expectation = min(1.0, 0.5 * conf + 0.5 * rate)
                pred_before = {
                    "confidence": conf,
                    "presence_rate": rate,
                    "last_position": list(entity_model.last_absolute_position)
                    if entity_model.last_absolute_position
                    else None,
                    "observations": int(entity_model.observations),
                }
            # Fallback: memory still expects a location → moderate residual expectation
            if presence_expectation <= 0.0 and mem is not None and mem.expected_location:
                presence_expectation = 0.55

        significance = compute_entity_significance(
            strength=strength,
            trust=trust,
            interaction_count=interactions,
            memory_strength=mem_strength,
            familiarity=familiarity,
            communicate_count=communicate,
            predicted_utility=utility,
        )

        rec = self.records.get(entity_id) or EntityLossRecord(entity_id=entity_id)
        rec.status = EntityStatus.HISTORICAL
        rec.disappearance_tick = tick
        rec.significance = significance
        rec.presence_expectation_at_loss = presence_expectation
        rec.relationship_before = rel.to_dict() if rel else {}
        rec.memory_before = {
            "strength": mem_strength,
            "familiarity": familiarity,
            "expected_location": list(mem.expected_location) if mem and mem.expected_location else None,
            "last_seen": mem.last_seen if mem else None,
            "interaction_value": float(mem.interaction_value) if mem else 0.0,
            "uncertainty": float(mem.uncertainty) if mem else 1.0,
        }
        rec.prediction_before = pred_before
        rec.memory_strength_current = mem_strength
        rec.social_loss = compute_entity_social_loss(
            significance=significance,
            presence_expectation=presence_expectation,
            information_reliability=reliability,
            ticks_since_disappearance=0,
            prediction_disruption=0.0,
            search_pressure=0.0,
            memory_retrieved_this_tick=False,
        )
        rec.peak_social_loss = rec.social_loss
        rec.causal_events.append(
            {
                "tick": tick,
                "cause": "disappearance_of_entity",
                "entity_id": entity_id,
                "associated_with": [
                    "relationship_representation_retained",
                    "memory_representation_retained",
                    "expectation_may_persist",
                ],
                "significance": significance,
                "initial_social_loss": rec.social_loss,
            }
        )
        rec.loss_trajectory.append({"tick": tick, "social_loss": rec.social_loss})
        rec.memory_trajectory.append({"tick": tick, "memory_strength": mem_strength})
        self.records[entity_id] = rec
        self._refresh_aggregate(tick)
        self.dimension_changes.append(
            {
                "dimension": "social_loss",
                "old_value": 0.0,
                "new_value": self.aggregate_social_loss,
                "cause": "disappearance_of_significant_entity",
                "entity_id": entity_id,
                "tick": tick,
            }
        )
        return rec

    def on_tick(self, agent, *, tick: int, active_ids: set[str]) -> None:
        """Update historical entities from prediction / memory / search."""
        from the_last_ai.types import ACTION_DELTA, MOVEMENT_ACTIONS, Position

        retrieved = set(getattr(agent, "last_retrieved_ids", []) or [])
        pos = None
        if hasattr(agent, "last_observation") and agent.last_observation is not None:
            pos = agent.last_observation.position

        for entity_id, rec in list(self.records.items()):
            if rec.status == EntityStatus.ACTIVE:
                if entity_id not in active_ids:
                    # Disappearance may have been missed; open record lightly
                    self.record_disappearance(agent, entity_id, tick=tick)
                continue
            if rec.status == EntityStatus.FORGOTTEN:
                continue
            if rec.disappearance_tick is None:
                continue

            age = tick - rec.disappearance_tick
            mem = agent.ltm.get(entity_id) if hasattr(agent, "ltm") else None
            mem_strength = (
                float(mem.strength_at(tick, agent.ltm.config.decay_lambda)) if mem else 0.0
            )
            rec.memory_strength_current = mem_strength

            # Presence prediction error: model may still expect the entity while world lacks it
            presence_error = 0.0
            pending_about = False
            if hasattr(agent, "world_model"):
                entities = getattr(getattr(agent.world_model, "agents", None), "entities", {}) or {}
                entity_model = entities.get(entity_id)
                pending_about = any(
                    p.subject_id == entity_id for p in agent.world_model.pending
                )
                if entity_model is not None:
                    # Score recent presence predictions about this subject if available
                    recent = [
                        e
                        for e in agent.world_model.errors[-40:]
                        if getattr(e.prediction, "subject_id", None) == entity_id
                        and e.prediction.kind.value in {"agent_presence", "agent_position"}
                    ]
                    if recent:
                        presence_error = sum(e.error for e in recent) / len(recent)
                    else:
                        # Residual: still tracked + not visible → disruption proportional to confidence
                        conf = float(entity_model.confidence())
                        total_pres = (
                            entity_model.presence_when_expected + entity_model.absence_when_expected
                        )
                        rate = (
                            entity_model.presence_when_expected / total_pres
                            if total_pres > 0
                            else (0.7 if entity_model.observations > 0 else 0.3)
                        )
                        presence_error = conf * rate
                else:
                    presence_error = max(
                        0.0, rec.presence_expectation_at_loss * math_exp_decay(age)
                    )
            else:
                presence_error = max(0.0, rec.presence_expectation_at_loss * math_exp_decay(age))

            rec.presence_prediction_error_ema = (
                0.85 * rec.presence_prediction_error_ema + 0.15 * presence_error
            )
            rec.prediction_disruption = (
                0.85 * rec.prediction_disruption + 0.15 * presence_error
            )
            rec.peak_prediction_disruption = max(
                rec.peak_prediction_disruption, rec.prediction_disruption
            )

            retrieved_now = entity_id in retrieved or pending_about
            if entity_id in retrieved:
                rec.memory_retrievals_after += 1
                rec.causal_events.append(
                    {
                        "tick": tick,
                        "cause": "memory_retrieved",
                        "entity_id": entity_id,
                        "followed_by": "possible_search_or_expectation_maintenance",
                    }
                )
            elif pending_about and tick % 5 == 0:
                rec.memory_retrievals_after += 1
                rec.causal_events.append(
                    {
                        "tick": tick,
                        "cause": "world_model_still_predicts_entity",
                        "entity_id": entity_id,
                        "associated_with": "expectation_persistence",
                    }
                )

            # Search toward last expected location (memory-guided or prediction-guided)
            moved_toward = False
            target_pos = None
            if mem is not None and mem.expected_location is not None:
                target_pos = Position(*mem.expected_location)
            elif hasattr(agent, "world_model"):
                entities = getattr(getattr(agent.world_model, "agents", None), "entities", {}) or {}
                em = entities.get(entity_id)
                if em is not None and em.last_absolute_position is not None:
                    target_pos = Position(*em.last_absolute_position)

            if pos is not None and target_pos is not None:
                dist = pos.manhattan(target_pos)
                last_action = getattr(agent, "last_action", None)
                if last_action is not None and last_action in MOVEMENT_ACTIONS:
                    dx, dy = ACTION_DELTA[last_action]
                    nxt = pos.offset(dx, dy)
                    if nxt.manhattan(target_pos) < dist:
                        moved_toward = True

                pursuing = moved_toward or (entity_id in retrieved) or pending_about
                if dist <= 1 and (moved_toward or entity_id in retrieved or pending_about):
                    rec.search_attempts += 1
                    rec.failed_searches += 1
                    rec.search_pressure = min(1.0, rec.search_pressure + 0.25)
                    rec.causal_events.append(
                        {
                            "tick": tick,
                            "cause": "search_failed",
                            "entity_id": entity_id,
                            "associated_with": "expected_location_empty",
                        }
                    )
                    if int((rec.relationship_before or {}).get("communicate_count", 0) or 0) > 0:
                        self.note_communication_toward_absent(entity_id, tick=tick)
                elif pursuing and moved_toward:
                    rec.search_pressure = min(1.0, rec.search_pressure + 0.08)
                    if tick % 3 == 0:
                        rec.search_attempts += 1
                        rec.causal_events.append(
                            {
                                "tick": tick,
                                "cause": "prediction_guided_pursuit",
                                "entity_id": entity_id,
                                "associated_with": "movement_toward_expected_locus",
                            }
                        )
                else:
                    rec.search_pressure *= 0.92
            else:
                rec.search_pressure *= 0.92

            reliability = float(
                (rec.relationship_before or {}).get("information_reliability", 0.5)
            )
            old_loss = rec.social_loss
            rec.social_loss = compute_entity_social_loss(
                significance=rec.significance,
                presence_expectation=rec.presence_expectation_at_loss,
                information_reliability=reliability,
                ticks_since_disappearance=age,
                prediction_disruption=rec.prediction_disruption,
                search_pressure=rec.search_pressure,
                memory_retrieved_this_tick=retrieved_now,
            )
            if rec.social_loss > rec.peak_social_loss:
                rec.peak_social_loss = rec.social_loss
                self.dimension_changes.append(
                    {
                        "dimension": "social_loss",
                        "old_value": old_loss,
                        "new_value": rec.social_loss,
                        "cause": "elevated_unresolved_expectation",
                        "entity_id": entity_id,
                        "tick": tick,
                    }
                )

            if tick % max(1, self.sample_interval) == 0:
                rec.loss_trajectory.append({"tick": tick, "social_loss": rec.social_loss})
                rec.prediction_error_trajectory.append(
                    {"tick": tick, "prediction_disruption": rec.prediction_disruption}
                )
                rec.memory_trajectory.append(
                    {"tick": tick, "memory_strength": mem_strength}
                )
                # Cap trajectories
                for traj in (
                    rec.loss_trajectory,
                    rec.prediction_error_trajectory,
                    rec.memory_trajectory,
                ):
                    if len(traj) > 400:
                        del traj[:-400]

            if not rec.adapted and assess_adaptation(
                social_loss=rec.social_loss,
                peak_social_loss=rec.peak_social_loss,
                prediction_disruption=rec.prediction_disruption,
                search_pressure=rec.search_pressure,
                ticks_since_loss=age,
                config=self.adaptation_config,
            ):
                rec.adapted = True
                rec.adaptation_tick = tick
                rec.causal_events.append(
                    {
                        "tick": tick,
                        "cause": "computational_adaptation",
                        "entity_id": entity_id,
                        "definition": (
                            "social_loss and prediction_disruption returned toward "
                            "baseline relative to peak (configurable AdaptationConfig)"
                        ),
                    }
                )

            # Forgotten: memory below threshold and loss near zero
            if mem is None or mem_strength < self.forget_memory_threshold:
                if rec.social_loss < 0.05 and age > 20:
                    rec.status = EntityStatus.FORGOTTEN
                    rec.causal_events.append(
                        {
                            "tick": tick,
                            "cause": "memory_below_forget_threshold",
                            "entity_id": entity_id,
                            "status": "FORGOTTEN",
                        }
                    )

        old_agg = self.aggregate_social_loss
        self._refresh_aggregate(tick)
        if abs(self.aggregate_social_loss - old_agg) >= 0.05:
            self.dimension_changes.append(
                {
                    "dimension": "social_loss",
                    "old_value": old_agg,
                    "new_value": self.aggregate_social_loss,
                    "cause": "aggregate_update",
                    "tick": tick,
                }
            )

    def note_communication_toward_absent(self, entity_id: str, *, tick: int) -> None:
        rec = self.records.get(entity_id)
        if rec is None or rec.status != EntityStatus.HISTORICAL:
            return
        rec.communication_attempts_after += 1
        rec.causal_events.append(
            {
                "tick": tick,
                "cause": "communication_attempt_toward_absent_entity",
                "entity_id": entity_id,
            }
        )

    def _refresh_aggregate(self, tick: int) -> None:
        comps = [
            r.social_loss
            for r in self.records.values()
            if r.status == EntityStatus.HISTORICAL
        ]
        self.aggregate_social_loss = aggregate_social_loss(comps)

    def summary(self, *, active_ids: set[str] | None = None) -> JSONDict:
        active_ids = active_ids or set()
        by_status = {s.value: [] for s in EntityStatus}
        for eid, rec in self.records.items():
            if rec.status == EntityStatus.ACTIVE and eid not in active_ids:
                # Stale active label → treat as historical for display
                status = EntityStatus.HISTORICAL.value
            else:
                status = rec.status.value
            by_status[status].append(rec.to_dict())
        return {
            "aggregate_social_loss": self.aggregate_social_loss,
            "by_status": by_status,
            "records": {eid: r.to_dict() for eid, r in self.records.items()},
            "dimension_changes": list(self.dimension_changes[-50:]),
            "counts": {
                "active": len(by_status["ACTIVE"]),
                "historical": len(by_status["HISTORICAL"]),
                "forgotten": len(by_status["FORGOTTEN"]),
            },
        }

    def to_dict(self) -> JSONDict:
        return {
            "aggregate_social_loss": self.aggregate_social_loss,
            "forget_memory_threshold": self.forget_memory_threshold,
            "sample_interval": self.sample_interval,
            "records": {eid: r.to_dict() for eid, r in self.records.items()},
            "dimension_changes": list(self.dimension_changes),
            "adaptation_config": {
                "social_loss_fraction_of_peak": self.adaptation_config.social_loss_fraction_of_peak,
                "prediction_disruption_max": self.adaptation_config.prediction_disruption_max,
                "search_pressure_max": self.adaptation_config.search_pressure_max,
                "min_ticks_after_loss": self.adaptation_config.min_ticks_after_loss,
            },
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> LossTracker:
        cfg = AdaptationConfig()
        ac = data.get("adaptation_config") or {}
        for k, v in ac.items():
            if hasattr(cfg, k):
                setattr(cfg, k, type(getattr(cfg, k))(v))
        tracker = cls(
            aggregate_social_loss=float(data.get("aggregate_social_loss", 0.0)),
            forget_memory_threshold=float(data.get("forget_memory_threshold", 0.02)),
            sample_interval=int(data.get("sample_interval", 1)),
            adaptation_config=cfg,
            dimension_changes=list(data.get("dimension_changes") or []),
        )
        for eid, rd in (data.get("records") or {}).items():
            tracker.records[eid] = EntityLossRecord.from_dict(rd)
        return tracker


def math_exp_decay(age: int, lam: float = 0.01) -> float:
    import math

    return math.exp(-lam * max(0, age))
