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
    """
    vocab = vocabulary or DEFAULT_VOCABULARY
    if observation is None:
        return None

    tokens: list[str] = []
    claimed: tuple[int, int] | None = None
    sender_pos = observation.position.as_tuple()

    resource_pos = _nearest_resource_absolute(observation)
    energy = observation.energy
    rel = None
    if hasattr(agent, "relationships"):
        rel = agent.relationships.get(receiver_id)

    # Resource signalling: if a resource is visible, prefer ["food","here"].
    if resource_pos is not None and max_tokens >= 1:
        tokens.append(vocab.primary_token(Concept.RESOURCE, "food"))
        if max_tokens >= 2:
            tokens.append(vocab.primary_token(Concept.LOCATION_HERE, "here"))
        claimed = resource_pos.as_tuple()
    elif energy < 40.0 and max_tokens >= 1:
        tokens.append(vocab.primary_token(Concept.ASSIST, "help"))
    elif rel is not None and rel.uncertainty > 0.55 and max_tokens >= 1:
        tokens.append(vocab.primary_token(Concept.GREETING, "hi"))
    elif rel is not None and rel.trust >= 0.6 and max_tokens >= 1:
        tokens.append(vocab.primary_token(Concept.APPROACH, "come"))
    elif hasattr(agent, "pending_claims") and agent.pending_claims and max_tokens >= 1:
        # Echo affirmation / negation if verifying context is active.
        tokens.append(vocab.primary_token(Concept.AFFIRM, "yes"))
    else:
        tokens.append(vocab.primary_token(Concept.GREETING, "hi"))

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

    # Resource + location: may raise investigate utility (not hard-coded obedience).
    if Concept.RESOURCE in concepts and reliability >= investigate_threshold:
        target = message.claimed_position or message.sender_position
        if target is not None:
            # Weight by reliability — low-reliability senders rarely redirect behaviour.
            strength = reliability
            existing = getattr(receiver, "pending_investigation", None)
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
        if message.sender_position is not None:
            receiver.pending_investigation = Position(*message.sender_position)
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

        # Attach positions from world if observation lacked them.
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

        # Sender stores outbound record.
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

        # Relationship counters for communication attempts.
        if hasattr(speaker, "relationships"):
            rel_out = speaker.relationships.get_or_create(listener.agent_id)
            rel_out.communicate_count += 0  # already counted via apply_social_outcome
        if hasattr(listener, "relationships"):
            # Ensure relationship exists even if listener had no prior bond.
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
        # Only verify once the agent is at/near the claimed cell.
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
                # Fallback if relationship lacks helper.
                lr = 0.2
                target_r = 1.0 if success else 0.0
                rel.information_reliability = (1 - lr) * getattr(
                    rel, "information_reliability", 0.5
                ) + lr * target_r
                if success:
                    rel.successful_messages = getattr(rel, "successful_messages", 0) + 1
                else:
                    rel.failed_messages = getattr(rel, "failed_messages", 0) + 1

        # Update matching communication memory records.
        if hasattr(agent, "communication_memory"):
            for record in agent.communication_memory.records:
                if record.message_id == claim.message_id:
                    record.verified = success
                    record.outcome = outcome

        # Clear investigation if this was the active target.
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
        # Keep falsified/verified claims briefly for audit, then drop.
        # (not appended to remaining)

    agent.pending_claims = remaining
    return events
