"""Per-agent chronological life history."""

from __future__ import annotations

from dataclasses import dataclass, field

from the_last_ai.observatory.affect import AffectiveState, relationship_label
from the_last_ai.observatory.event import LifeEvent, LifeEventType
from the_last_ai.types import JSONDict


@dataclass
class RelationshipTrajectory:
    other_id: str
    first_interaction_tick: int | None = None
    last_interaction_tick: int | None = None
    interaction_count: int = 0
    cooperation_count: int = 0
    competition_count: int = 0
    communication_count: int = 0
    successful_messages: int = 0
    failed_messages: int = 0
    trust_history: list[tuple[int, float]] = field(default_factory=list)
    strength_history: list[tuple[int, float]] = field(default_factory=list)
    utility_history: list[tuple[int, float]] = field(default_factory=list)
    reliability_history: list[tuple[int, float]] = field(default_factory=list)
    verification_events: list[JSONDict] = field(default_factory=list)
    historical: bool = False

    def snapshot(
        self,
        *,
        tick: int,
        trust: float,
        strength: float,
        utility: float,
        reliability: float | None = None,
    ) -> None:
        self.trust_history.append((tick, float(trust)))
        self.strength_history.append((tick, float(strength)))
        self.utility_history.append((tick, float(utility)))
        if reliability is not None:
            self.reliability_history.append((tick, float(reliability)))
        # Cap trajectory length for performance
        for hist in (
            self.trust_history,
            self.strength_history,
            self.utility_history,
            self.reliability_history,
        ):
            if len(hist) > 200:
                del hist[:-200]

    def to_dict(self) -> JSONDict:
        return {
            "other_id": self.other_id,
            "first_interaction_tick": self.first_interaction_tick,
            "last_interaction_tick": self.last_interaction_tick,
            "interaction_count": self.interaction_count,
            "cooperation_count": self.cooperation_count,
            "competition_count": self.competition_count,
            "communication_count": self.communication_count,
            "successful_messages": self.successful_messages,
            "failed_messages": self.failed_messages,
            "trust_history": [[t, v] for t, v in self.trust_history],
            "strength_history": [[t, v] for t, v in self.strength_history],
            "utility_history": [[t, v] for t, v in self.utility_history],
            "reliability_history": [[t, v] for t, v in self.reliability_history],
            "verification_events": list(self.verification_events),
            "historical": self.historical,
            "label": relationship_label(
                interactions=self.interaction_count,
                strength=self.strength_history[-1][1] if self.strength_history else 0.0,
                trust=self.trust_history[-1][1] if self.trust_history else 0.5,
            ),
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> RelationshipTrajectory:
        traj = cls(
            other_id=str(data["other_id"]),
            first_interaction_tick=data.get("first_interaction_tick"),
            last_interaction_tick=data.get("last_interaction_tick"),
            interaction_count=int(data.get("interaction_count", 0)),
            cooperation_count=int(data.get("cooperation_count", 0)),
            competition_count=int(data.get("competition_count", 0)),
            communication_count=int(data.get("communication_count", 0)),
            successful_messages=int(data.get("successful_messages", 0)),
            failed_messages=int(data.get("failed_messages", 0)),
            verification_events=list(data.get("verification_events") or []),
            historical=bool(data.get("historical", False)),
        )
        traj.trust_history = [(int(t), float(v)) for t, v in data.get("trust_history") or []]
        traj.strength_history = [(int(t), float(v)) for t, v in data.get("strength_history") or []]
        traj.utility_history = [(int(t), float(v)) for t, v in data.get("utility_history") or []]
        traj.reliability_history = [
            (int(t), float(v)) for t, v in data.get("reliability_history") or []
        ]
        return traj


@dataclass
class AgentLifeHistory:
    """Complete inspectable computational biography for one agent."""

    agent_id: str
    birth_tick: int = 0
    events: list[LifeEvent] = field(default_factory=list)
    affect_series: list[tuple[int, AffectiveState]] = field(default_factory=list)
    relationship_trajectories: dict[str, RelationshipTrajectory] = field(default_factory=dict)
    messages: list[JSONDict] = field(default_factory=list)
    decisions: list[JSONDict] = field(default_factory=list)
    known_entities: set[str] = field(default_factory=set)
    max_events: int = 5000
    full_detail: bool = True

    def record(self, event: LifeEvent) -> None:
        self.events.append(event)
        if len(self.events) > self.max_events:
            # Keep birth + most recent
            birth = [e for e in self.events if e.event_type == LifeEventType.BIRTH][:1]
            self.events = birth + self.events[-(self.max_events - len(birth)) :]

    def record_affect(self, tick: int, state: AffectiveState) -> None:
        self.affect_series.append((tick, state))
        if len(self.affect_series) > 1000:
            # downsample older half
            keep = self.affect_series[::2][-400:] + self.affect_series[-200:]
            self.affect_series = keep

    def get_or_create_traj(self, other_id: str) -> RelationshipTrajectory:
        if other_id not in self.relationship_trajectories:
            self.relationship_trajectories[other_id] = RelationshipTrajectory(other_id=other_id)
        return self.relationship_trajectories[other_id]

    def mark_historical(self, absent_ids: set[str]) -> None:
        for oid in absent_ids:
            if oid in self.relationship_trajectories:
                self.relationship_trajectories[oid].historical = True

    def overview(self, *, active_ids: set[str] | None = None, final_tick: int | None = None) -> JSONDict:
        active_ids = active_ids or set()
        last_affect = self.affect_series[-1][1].to_dict() if self.affect_series else {}
        rels = self.relationship_trajectories
        active_rels = sum(1 for oid, t in rels.items() if oid in active_ids and not t.historical)
        hist_rels = sum(1 for oid, t in rels.items() if oid not in active_ids or t.historical)
        return {
            "agent_id": self.agent_id,
            "birth_tick": self.birth_tick,
            "age_ticks": (final_tick - self.birth_tick) if final_tick is not None else None,
            "experience_count": len(self.events),
            "message_count": len(self.messages),
            "decision_count": len(self.decisions),
            "relationships_active": active_rels,
            "relationships_historical": hist_rels,
            "known_entities": sorted(self.known_entities),
            "affect_latest": last_affect,
            "affect_samples": len(self.affect_series),
        }

    def to_dict(self) -> JSONDict:
        return {
            "agent_id": self.agent_id,
            "birth_tick": self.birth_tick,
            "full_detail": self.full_detail,
            "max_events": self.max_events,
            "events": [e.to_dict() for e in self.events],
            "affect_series": [[t, s.to_dict()] for t, s in self.affect_series],
            "relationship_trajectories": {
                oid: traj.to_dict() for oid, traj in self.relationship_trajectories.items()
            },
            "messages": list(self.messages),
            "decisions": list(self.decisions),
            "known_entities": sorted(self.known_entities),
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> AgentLifeHistory:
        hist = cls(
            agent_id=str(data["agent_id"]),
            birth_tick=int(data.get("birth_tick", 0)),
            full_detail=bool(data.get("full_detail", True)),
            max_events=int(data.get("max_events", 5000)),
            known_entities=set(data.get("known_entities") or []),
            messages=list(data.get("messages") or []),
            decisions=list(data.get("decisions") or []),
        )
        hist.events = [LifeEvent.from_dict(e) for e in data.get("events") or []]
        hist.affect_series = [
            (int(t), AffectiveState.from_dict(s)) for t, s in data.get("affect_series") or []
        ]
        hist.relationship_trajectories = {
            oid: RelationshipTrajectory.from_dict(td)
            for oid, td in (data.get("relationship_trajectories") or {}).items()
        }
        return hist
