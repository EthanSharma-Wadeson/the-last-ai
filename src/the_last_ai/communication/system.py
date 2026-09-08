"""Communication exchange: construct → transmit → receive → verify → learn."""

from __future__ import annotations

from dataclasses import dataclass, field

from the_last_ai.communication.memory import CommunicationRecord
from the_last_ai.communication.message import Message
from the_last_ai.communication.vocabulary import (
    DEFAULT_VOCABULARY,
    Concept,
    Vocabulary,
)
from the_last_ai.loss.model import EntityStatus
from the_last_ai.types import CellType, JSONDict, Observation, Position


@dataclass
class PendingClaim:
    """Receiver-side claim awaiting environmental verification."""

    message_id: str
    sender_id: str
    claimed_position: tuple[int, int]
    tick_received: int
    concepts: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | verified | falsified

    def to_dict(self) -> JSONDict:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "claimed_position": list(self.claimed_position),
            "tick_received": self.tick_received,
            "concepts": list(self.concepts),
            "status": self.status,
        }


@dataclass
class CommunicationEvent:
    """Technical log entry for one delivered message (+ optional verification later)."""

    tick: int
    event: str
    sender: str
    receiver: str
    tokens: list[str]
    message_id: str
    sender_position: list[int] | None = None
    receiver_position: list[int] | None = None
    claimed_position: list[int] | None = None
    sender_reliability: float | None = None
    receiver_response: str | None = None
    verified: bool | None = None
    outcome: str | None = None
    concepts: list[str] = field(default_factory=list)
    construction_reason: str | None = None
    absence_driven: bool = False
    about_entity_id: str | None = None

    def to_dict(self) -> JSONDict:
        return {
            "tick": self.tick,
            "event": self.event,
            "sender": self.sender,
            "receiver": self.receiver,
            "tokens": list(self.tokens),
            "message_id": self.message_id,
            "sender_position": self.sender_position,
            "receiver_position": self.receiver_position,
            "claimed_position": self.claimed_position,
            "sender_reliability": self.sender_reliability,
            "receiver_response": self.receiver_response,
            "verified": self.verified,
            "outcome": self.outcome,
            "concepts": list(self.concepts),
            "construction_reason": self.construction_reason,
            "absence_driven": bool(self.absence_driven),
            "about_entity_id": self.about_entity_id,
        }


def _nearest_resource_absolute(observation: Observation) -> Position | None:
    best: tuple[int, Position] | None = None
    radius = observation.perception_radius
    for dy, row in enumerate(observation.local_cells):
        for dx, cell in enumerate(row):
            if cell != CellType.RESOURCE:
                continue
            rel_x = dx - radius
            rel_y = dy - radius
            pos = observation.position.offset(rel_x, rel_y)
            dist = abs(rel_x) + abs(rel_y)
            if best is None or dist < best[0]:
                best = (dist, pos)
    return best[1] if best else None


def _strongest_historical_absence(agent) -> tuple[object | None, float]:
    """Return (loss record, pressure score) for the strongest historical entity."""
    if not hasattr(agent, "loss_tracker"):
        return None, 0.0
    best = None
    best_score = 0.0
    for rec in agent.loss_tracker.records.values():
        if rec.status != EntityStatus.HISTORICAL:
            continue
        score = (
            0.45 * float(rec.social_loss)
            + 0.30 * float(rec.search_pressure)
            + 0.15 * float(rec.prediction_disruption)
            + 0.10 * float(rec.significance)
        )
        if score > best_score:
            best_score = score
            best = rec
    return best, best_score


def _expected_locus(agent, entity_id: str) -> tuple[int, int] | None:
    if hasattr(agent, "ltm"):
        mem = agent.ltm.get(entity_id)
        if mem is not None and mem.expected_location is not None:
            return tuple(mem.expected_location)  # type: ignore[return-value]
    if hasattr(agent, "loss_tracker"):
        rec = agent.loss_tracker.records.get(entity_id)
        if rec is not None:
            loc = (rec.memory_before or {}).get("expected_location")
            if loc:
                return (int(loc[0]), int(loc[1]))
            loc = (rec.prediction_before or {}).get("last_position")
            if loc:
                return (int(loc[0]), int(loc[1]))
    return None


def _absence_message_plan(
    agent,
    *,
    observation: Observation,
    vocab: Vocabulary,
    max_tokens: int,
) -> tuple[list[str], tuple[int, int] | None, str, str] | None:
    """
    Map measurable loss / search state → absence-conditioned tokens.

    Returns (tokens, claimed_position, reason, about_entity_id) or None.
    """
    rec, pressure = _strongest_historical_absence(agent)
    if rec is None or pressure < 0.18:
        return None

    # Rate-limit absence talk so the feed stays readable.
    last_tick = int(getattr(agent, "last_absence_message_tick", -10_000))
    if observation.tick - last_tick < 4:
        return None

    about = rec.entity_id
    locus = _expected_locus(agent, about)
    tokens: list[str] = []
    reason = "absence_signal"

    if float(rec.search_pressure) >= 0.35 and float(rec.failed_searches) > 0:
        tokens = [
            vocab.primary_token(Concept.NEGATE, "no"),
            vocab.primary_token(Concept.ABSENCE, "gone"),
        ]
        reason = "failed_seek_at_expected_locus"
    elif float(rec.search_pressure) >= 0.25:
        tokens = [
            vocab.primary_token(Concept.APPROACH, "come"),
            vocab.primary_token(Concept.LOCATION_HERE, "here"),
        ]
        reason = "search_pressure_toward_last_known"
    elif float(rec.social_loss) >= 0.28 or float(rec.prediction_disruption) >= 0.35:
        tokens = [
            vocab.primary_token(Concept.ADDRESSEE, "you"),
            vocab.primary_token(Concept.QUERY_LOCATION, "where"),
        ]
        reason = "prediction_mismatch_or_social_loss"
    elif float(rec.social_loss) >= 0.18:
        tokens = [
            vocab.primary_token(Concept.ADDRESSEE, "you"),
            vocab.primary_token(Concept.ABSENCE, "gone"),
        ]
        reason = "elevated_social_loss"
    else:
        tokens = [
            vocab.primary_token(Concept.DELAY, "wait"),
            vocab.primary_token(Concept.ADDRESSEE, "you"),
        ]
        reason = "residual_absence_expectation"

    tokens = tokens[: max(1, max_tokens)]
    return tokens, locus, reason, about


def construct_message(
    agent,
    *,
    receiver_id: str,
    observation: Observation | None,
    vocabulary: Vocabulary | None = None,
    max_tokens: int = 2,
) -> Message | None:
    """
    Build a short token list from current internal/perceptual state.

    Not a grammar — heuristic mapping from measurable state → symbols.
    Absence-conditioned tokens fire from loss / search / prediction signals.
    """
    vocab = vocabulary or DEFAULT_VOCABULARY
    if observation is None:
        return None

    tokens: list[str] = []
    claimed: tuple[int, int] | None = None
    sender_pos = observation.position.as_tuple()
    construction_reason = "default_greeting"
    absence_driven = False
    about_entity_id: str | None = None

    resource_pos = _nearest_resource_absolute(observation)
    energy = observation.energy
    rel = None
    if hasattr(agent, "relationships"):
        rel = agent.relationships.get(receiver_id)

    absence_plan = _absence_message_plan(
        agent, observation=observation, vocab=vocab, max_tokens=max_tokens
    )
    # Critical energy / visible food still wins; otherwise absence talk can fire.
    prefer_resource = resource_pos is not None and (
        energy < 35.0 or absence_plan is None
    )

    if prefer_resource and resource_pos is not None and max_tokens >= 1:
        tokens.append(vocab.primary_token(Concept.RESOURCE, "food"))
        if max_tokens >= 2:
            tokens.append(vocab.primary_token(Concept.LOCATION_HERE, "here"))
        claimed = resource_pos.as_tuple()
        construction_reason = "visible_resource"
    elif absence_plan is not None:
        tokens, claimed, construction_reason, about_entity_id = absence_plan
        absence_driven = True
        agent.last_absence_message_tick = int(observation.tick)
    elif energy < 40.0 and max_tokens >= 1:
        tokens.append(vocab.primary_token(Concept.ASSIST, "help"))
        construction_reason = "low_energy"
    elif rel is not None and rel.uncertainty > 0.55 and max_tokens >= 1:
        tokens.append(vocab.primary_token(Concept.GREETING, "hi"))
        construction_reason = "relationship_uncertainty"
    elif rel is not None and rel.trust >= 0.6 and max_tokens >= 1:
        tokens.append(vocab.primary_token(Concept.APPROACH, "come"))
        construction_reason = "high_trust_approach"
    elif hasattr(agent, "pending_claims") and agent.pending_claims and max_tokens >= 1:
        tokens.append(vocab.primary_token(Concept.AFFIRM, "yes"))
        construction_reason = "pending_claim_echo"
    else:
        tokens.append(vocab.primary_token(Concept.GREETING, "hi"))
        construction_reason = "default_greeting"

    tokens = tokens[: max(1, max_tokens)]
    concepts = [c.value for c in vocab.concepts(tokens)]

    return Message(
        sender_id=agent.agent_id,
        receiver_id=receiver_id,
        tokens=tokens,
        tick=observation.tick,
        sender_position=sender_pos,
        claimed_position=claimed,
        concepts=concepts,
        construction_reason=construction_reason,
        absence_driven=absence_driven,
        about_entity_id=about_entity_id,
    )


def deliver_message(
    receiver,
    message: Message,
    *,
    vocabulary: Vocabulary | None = None,
    investigate_threshold: float = 0.35,
) -> str:
    """
    Receiver processes a message as uncertain information.

    Returns a response label for technical logging (not dialogue text).
    """
    vocab = vocabulary or DEFAULT_VOCABULARY
    if not hasattr(receiver, "communication_memory"):
        return "ignored"

    reliability = 0.5
    if hasattr(receiver, "relationships"):
        rel = receiver.relationships.get(message.sender_id)
        if rel is not None:
            reliability = float(getattr(rel, "information_reliability", rel.trust))

    receiver.communication_memory.add(
        CommunicationRecord.from_message(message, role="receiver", reliability=reliability)
    )

    concepts = set(vocab.concepts(message.tokens))
    response = "recorded"

    # Absence / query tokens may redirect investigation to the claimed last-known locus.
    if (
        (Concept.ABSENCE in concepts or Concept.QUERY_LOCATION in concepts)
        and message.claimed_position is not None
        and reliability >= investigate_threshold * 0.7
    ):
        strength = reliability * 0.75
        existing_priority = float(getattr(receiver, "investigation_priority", 0.0))
        if strength >= existing_priority:
            receiver.pending_investigation = Position(*message.claimed_position)
            receiver.investigation_priority = strength
            receiver.investigation_source = message.sender_id
            receiver.investigation_message_id = message.message_id
            response = "investigate_absence"
    # Resource + location: may raise investigate utility (not hard-coded obedience).
    elif Concept.RESOURCE in concepts and reliability >= investigate_threshold:
        target = message.claimed_position or message.sender_position
        if target is not None:
            strength = reliability
            existing_priority = float(getattr(receiver, "investigation_priority", 0.0))
            if strength >= existing_priority:
                receiver.pending_investigation = Position(target[0], target[1])
                receiver.investigation_priority = strength
                receiver.investigation_source = message.sender_id
                receiver.investigation_message_id = message.message_id
                if not hasattr(receiver, "pending_claims"):
                    receiver.pending_claims = []
                receiver.pending_claims.append(
                    PendingClaim(
                        message_id=message.message_id,
                        sender_id=message.sender_id,
                        claimed_position=target,
                        tick_received=message.tick,
                        concepts=[c.value for c in concepts],
                    )
                )
                response = "investigate"
    elif Concept.APPROACH in concepts and reliability >= investigate_threshold:
        target = message.claimed_position or message.sender_position
        if target is not None:
            receiver.pending_investigation = Position(*target)
            receiver.investigation_priority = reliability * 0.8
            receiver.investigation_source = message.sender_id
            response = "approach_sender"
    elif Concept.GREETING in concepts:
        response = "acknowledge"
    elif Concept.ASSIST in concepts and reliability >= investigate_threshold:
        if message.sender_position is not None:
            receiver.pending_investigation = Position(*message.sender_position)
            receiver.investigation_priority = reliability * 0.7
            receiver.investigation_source = message.sender_id
            response = "assist_move"

    if hasattr(receiver, "messages_received_count"):
        receiver.messages_received_count += 1
    if hasattr(receiver, "last_received_message"):
        receiver.last_received_message = message.to_dict()

    return response


def exchange_communication(
    agent_a,
    agent_b,
    *,
    intent_a,
    intent_b,
    tick: int,
    positions: dict[str, Position],
    vocabulary: Vocabulary | None = None,
    max_tokens: int = 2,
    energy_cost: float = 0.5,
) -> list[CommunicationEvent]:
    """
    If either agent intended COMMUNICATE, construct and deliver symbolic messages.

    Returns technical communication events for the simulation log.
    """
    from the_last_ai.social.types import InteractionKind

    vocab = vocabulary or DEFAULT_VOCABULARY
    events: list[CommunicationEvent] = []

    speakers: list[tuple[object, object]] = []
    if intent_a == InteractionKind.COMMUNICATE:
        speakers.append((agent_a, agent_b))
    if intent_b == InteractionKind.COMMUNICATE:
        speakers.append((agent_b, agent_a))

    for speaker, listener in speakers:
        if not getattr(speaker, "symbolic_communication_enabled", True):
            continue
        if not hasattr(speaker, "communication_memory"):
            continue

        obs = getattr(speaker, "last_observation", None)
        message = construct_message(
            speaker,
            receiver_id=listener.agent_id,
            observation=obs,
            vocabulary=vocab,
            max_tokens=max_tokens,
        )
        if message is None:
            continue

        sp = positions.get(speaker.agent_id)
        rp = positions.get(listener.agent_id)
        if sp is not None:
            message.sender_position = sp.as_tuple()
        if rp is not None:
            message.receiver_position = rp.as_tuple()
        message.tick = tick

        if energy_cost > 0 and hasattr(speaker, "state"):
            speaker.state.energy = max(0.0, speaker.state.energy - energy_cost)

        reliability = 0.5
        if hasattr(listener, "relationships"):
            rel = listener.relationships.get(speaker.agent_id)
            if rel is not None:
                reliability = float(getattr(rel, "information_reliability", 0.5))

        speaker.communication_memory.add(
            CommunicationRecord.from_message(message, role="sender", reliability=reliability)
        )
        if hasattr(speaker, "messages_sent_count"):
            speaker.messages_sent_count += 1
        if hasattr(speaker, "last_sent_message"):
            speaker.last_sent_message = message.to_dict()

        response = deliver_message(
            listener, message, vocabulary=vocab, investigate_threshold=0.35
        )

        if hasattr(speaker, "relationships"):
            speaker.relationships.get_or_create(listener.agent_id)
        if hasattr(listener, "relationships"):
            listener.relationships.get_or_create(speaker.agent_id)

        events.append(
            CommunicationEvent(
                tick=tick,
                event="communication",
                sender=speaker.agent_id,
                receiver=listener.agent_id,
                tokens=list(message.tokens),
                message_id=message.message_id,
                sender_position=list(message.sender_position) if message.sender_position else None,
                receiver_position=list(message.receiver_position)
                if message.receiver_position
                else None,
                claimed_position=list(message.claimed_position)
                if message.claimed_position
                else None,
                sender_reliability=reliability,
                receiver_response=response,
                concepts=list(message.concepts),
                construction_reason=message.construction_reason,
                absence_driven=bool(message.absence_driven),
                about_entity_id=message.about_entity_id,
            )
        )

    return events


def verify_pending_claims(agent, world) -> list[CommunicationEvent]:
    """
    Check pending resource claims against the current world.

    Updates information_reliability on the receiver's relationship toward the sender.
    """
    events: list[CommunicationEvent] = []
    claims = getattr(agent, "pending_claims", None)
    if not claims:
        return events

    pos = world.agent_positions.get(agent.agent_id)
    if pos is None:
        return events

    remaining = []
    for claim in claims:
        if claim.status != "pending":
            remaining.append(claim)
            continue
        target = Position(*claim.claimed_position)
        if pos.manhattan(target) > 1:
            remaining.append(claim)
            continue

        cell = world.grid.get(target)
        success = cell == CellType.RESOURCE
        claim.status = "verified" if success else "falsified"
        outcome = "resource_found" if success else "resource_absent"

        if hasattr(agent, "relationships"):
            rel = agent.relationships.get_or_create(claim.sender_id)
            if hasattr(rel, "update_information_reliability"):
                rel.update_information_reliability(success=success)
            else:
                lr = 0.2
                target_r = 1.0 if success else 0.0
                rel.information_reliability = (1 - lr) * getattr(
                    rel, "information_reliability", 0.5
                ) + lr * target_r
                if success:
                    rel.successful_messages = getattr(rel, "successful_messages", 0) + 1
                else:
                    rel.failed_messages = getattr(rel, "failed_messages", 0) + 1

        if hasattr(agent, "communication_memory"):
            for record in agent.communication_memory.records:
                if record.message_id == claim.message_id:
                    record.verified = success
                    record.outcome = outcome

        if getattr(agent, "investigation_message_id", None) == claim.message_id:
            agent.pending_investigation = None
            agent.investigation_priority = 0.0
            agent.investigation_source = None
            agent.investigation_message_id = None

        if hasattr(agent, "verification_count"):
            agent.verification_count += 1
        if success and hasattr(agent, "successful_verification_count"):
            agent.successful_verification_count += 1

        events.append(
            CommunicationEvent(
                tick=world.tick,
                event="communication_verification",
                sender=claim.sender_id,
                receiver=agent.agent_id,
                tokens=[],
                message_id=claim.message_id,
                claimed_position=list(claim.claimed_position),
                verified=success,
                outcome=outcome,
                receiver_response="verified" if success else "falsified",
            )
        )

    agent.pending_claims = remaining
    return events
