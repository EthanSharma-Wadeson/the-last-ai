"""Resolve pairwise social interactions into energy and valence outcomes."""

from __future__ import annotations

from the_last_ai.social.types import InteractionKind, InteractionOutcome


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def resolve_pair(
    *,
    tick: int,
    agent_a: str,
    agent_b: str,
    intent_a: InteractionKind | None,
    intent_b: InteractionKind | None,
    energy_a: float,
    energy_b: float,
    share_amount: float = 8.0,
    cooperate_bonus: float = 6.0,
    compete_steal: float = 5.0,
    communicate_value: float = 0.15,
) -> InteractionOutcome:
    """
    Map joint intents to outcomes.

    Trust is not assigned here — only raw outcomes. Agents learn trust from valence.
    """
    a = intent_a
    b = intent_b

    # Default: one-sided social attempt with a passive neighbour.
    if a is None and b is None:
        return InteractionOutcome(
            tick=tick,
            agent_a=agent_a,
            agent_b=agent_b,
            intent_a=None,
            intent_b=None,
            energy_delta_a=0.0,
            energy_delta_b=0.0,
            valence_a=0.0,
            valence_b=0.0,
            resolved_kind="none",
        )

    # Communicate (either side): information exchange, no energy transfer.
    if a == InteractionKind.COMMUNICATE or b == InteractionKind.COMMUNICATE:
        if a == InteractionKind.COMMUNICATE and b == InteractionKind.COMMUNICATE:
            return InteractionOutcome(
                tick=tick,
                agent_a=agent_a,
                agent_b=agent_b,
                intent_a=a,
                intent_b=b,
                energy_delta_a=0.0,
                energy_delta_b=0.0,
                valence_a=communicate_value,
                valence_b=communicate_value,
                resolved_kind="communicate_mutual",
            )
        speaker = a if a == InteractionKind.COMMUNICATE else b
        listener_is_a = speaker == b
        # Listener gets slightly more informational value; speaker gets mild positive valence.
        return InteractionOutcome(
            tick=tick,
            agent_a=agent_a,
            agent_b=agent_b,
            intent_a=a,
            intent_b=b,
            energy_delta_a=0.0,
            energy_delta_b=0.0,
            valence_a=communicate_value if not listener_is_a else communicate_value * 0.7,
            valence_b=communicate_value if listener_is_a else communicate_value * 0.7,
            resolved_kind="communicate",
        )

    # Share: donor transfers energy if able.
    if a == InteractionKind.SHARE or b == InteractionKind.SHARE:
        # If both share, each transfers a smaller amount.
        if a == InteractionKind.SHARE and b == InteractionKind.SHARE:
            amount = share_amount * 0.5
            da = -amount if energy_a >= amount else 0.0
            db = -amount if energy_b >= amount else 0.0
            # Net: each loses `amount` and gains the other's amount.
            return InteractionOutcome(
                tick=tick,
                agent_a=agent_a,
                agent_b=agent_b,
                intent_a=a,
                intent_b=b,
                energy_delta_a=da - db,  # -own + partner's
                energy_delta_b=db - da,
                valence_a=0.35 if da < 0 or db < 0 else -0.05,
                valence_b=0.35 if da < 0 or db < 0 else -0.05,
                resolved_kind="share_mutual",
            )
        if a == InteractionKind.SHARE:
            amount = share_amount if energy_a >= share_amount else 0.0
            return InteractionOutcome(
                tick=tick,
                agent_a=agent_a,
                agent_b=agent_b,
                intent_a=a,
                intent_b=b,
                energy_delta_a=-amount,
                energy_delta_b=amount,
                valence_a=0.25 if amount > 0 else -0.1,
                valence_b=0.45 if amount > 0 else 0.0,
                resolved_kind="share",
            )
        amount = share_amount if energy_b >= share_amount else 0.0
        return InteractionOutcome(
            tick=tick,
            agent_a=agent_a,
            agent_b=agent_b,
            intent_a=a,
            intent_b=b,
            energy_delta_a=amount,
            energy_delta_b=-amount,
            valence_a=0.45 if amount > 0 else 0.0,
            valence_b=0.25 if amount > 0 else -0.1,
            resolved_kind="share",
        )

    # Compete involved.
    if a == InteractionKind.COMPETE or b == InteractionKind.COMPETE:
        if a == InteractionKind.COMPETE and b == InteractionKind.COMPETE:
            # Costly conflict: both lose a little.
            return InteractionOutcome(
                tick=tick,
                agent_a=agent_a,
                agent_b=agent_b,
                intent_a=a,
                intent_b=b,
                energy_delta_a=-compete_steal * 0.4,
                energy_delta_b=-compete_steal * 0.4,
                valence_a=-0.35,
                valence_b=-0.35,
                resolved_kind="compete_mutual",
            )
        if a == InteractionKind.COMPETE:
            # Exploitation if partner cooperates or is passive.
            steal = min(compete_steal, max(0.0, energy_b * 0.2 + compete_steal * 0.5))
            partner_valence = -0.55 if b == InteractionKind.COOPERATE else -0.25
            return InteractionOutcome(
                tick=tick,
                agent_a=agent_a,
                agent_b=agent_b,
                intent_a=a,
                intent_b=b,
                energy_delta_a=steal,
                energy_delta_b=-steal,
                valence_a=0.35,
                valence_b=partner_valence,
                resolved_kind="compete",
            )
        steal = min(compete_steal, max(0.0, energy_a * 0.2 + compete_steal * 0.5))
        partner_valence = -0.55 if a == InteractionKind.COOPERATE else -0.25
        return InteractionOutcome(
            tick=tick,
            agent_a=agent_a,
            agent_b=agent_b,
            intent_a=a,
            intent_b=b,
            energy_delta_a=-steal,
            energy_delta_b=steal,
            valence_a=partner_valence,
            valence_b=0.35,
            resolved_kind="compete",
        )

    # Cooperate (one or both; no compete/share/communicate above).
    if a == InteractionKind.COOPERATE and b == InteractionKind.COOPERATE:
        return InteractionOutcome(
            tick=tick,
            agent_a=agent_a,
            agent_b=agent_b,
            intent_a=a,
            intent_b=b,
            energy_delta_a=cooperate_bonus,
            energy_delta_b=cooperate_bonus,
            valence_a=0.7,
            valence_b=0.7,
            resolved_kind="cooperate_mutual",
        )
    if a == InteractionKind.COOPERATE or b == InteractionKind.COOPERATE:
        # Unilateral cooperation with a passive partner: small positive for initiator.
        if a == InteractionKind.COOPERATE:
            return InteractionOutcome(
                tick=tick,
                agent_a=agent_a,
                agent_b=agent_b,
                intent_a=a,
                intent_b=b,
                energy_delta_a=cooperate_bonus * 0.25,
                energy_delta_b=cooperate_bonus * 0.1,
                valence_a=0.2,
                valence_b=0.15,
                resolved_kind="cooperate_unilateral",
            )
        return InteractionOutcome(
            tick=tick,
            agent_a=agent_a,
            agent_b=agent_b,
            intent_a=a,
            intent_b=b,
            energy_delta_a=cooperate_bonus * 0.1,
            energy_delta_b=cooperate_bonus * 0.25,
            valence_a=0.15,
            valence_b=0.2,
            resolved_kind="cooperate_unilateral",
        )

    return InteractionOutcome(
        tick=tick,
        agent_a=agent_a,
        agent_b=agent_b,
        intent_a=a,
        intent_b=b,
        energy_delta_a=0.0,
        energy_delta_b=0.0,
        valence_a=0.0,
        valence_b=0.0,
        resolved_kind="noop",
    )


def valence_from_energy(delta: float, scale: float = 10.0) -> float:
    return _clamp(delta / scale)
