"""Affective dimensions derived from existing agent internals — not scripted emotions."""

from __future__ import annotations

from dataclasses import dataclass

from the_last_ai.types import JSONDict


# Relationship-label thresholds (documented; configurable via RelationshipLabelConfig).
@dataclass
class RelationshipLabelConfig:
    stranger_max_interactions: int = 0
    acquaintance_min_interactions: int = 1
    familiar_min_interactions: int = 5
    familiar_min_strength: float = 0.35
    trusted_min_trust: float = 0.6
    trusted_min_strength: float = 0.55
    strongly_connected_min_trust: float = 0.7
    strongly_connected_min_strength: float = 0.75
    strongly_connected_min_interactions: int = 15


DEFAULT_RELATIONSHIP_LABELS = RelationshipLabelConfig()


def relationship_label(
    *,
    interactions: int,
    strength: float,
    trust: float,
    config: RelationshipLabelConfig | None = None,
) -> str:
    """Derive a cautious human-readable label from measurable relationship vars."""
    cfg = config or DEFAULT_RELATIONSHIP_LABELS
    if interactions <= cfg.stranger_max_interactions:
        return "stranger"
    if (
        interactions >= cfg.strongly_connected_min_interactions
        and trust >= cfg.strongly_connected_min_trust
        and strength >= cfg.strongly_connected_min_strength
    ):
        return "strongly connected"
    if trust >= cfg.trusted_min_trust and strength >= cfg.trusted_min_strength:
        return "trusted"
    if interactions >= cfg.familiar_min_interactions and strength >= cfg.familiar_min_strength:
        return "familiar"
    if interactions >= cfg.acquaintance_min_interactions:
        return "acquaintance"
    return "stranger"


@dataclass
class AffectiveState:
    """
    Bounded computational dimensions in roughly [-1, 1] or [0, 1] as noted.

    Derived from motivation, prediction error, relationships, memory, and outcomes.
    Not a claim of subjective emotion.
    """

    valence: float = 0.0  # [-1, 1]
    activation: float = 0.0  # [0, 1]
    social_drive: float = 0.0  # [0, 1]
    uncertainty: float = 0.5  # [0, 1]
    security: float = 0.5  # [0, 1]
    novelty: float = 0.0  # [0, 1]
    prediction_error: float = 0.5  # [0, 1]
    social_loss: float = 0.0  # [0, 1]
    goal_success: float = 0.0  # [-1, 1] recent reward proxy

    def to_dict(self) -> JSONDict:
        return {
            "valence": self.valence,
            "activation": self.activation,
            "social_drive": self.social_drive,
            "uncertainty": self.uncertainty,
            "security": self.security,
            "novelty": self.novelty,
            "prediction_error": self.prediction_error,
            "social_loss": self.social_loss,
            "goal_success": self.goal_success,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> AffectiveState:
        return cls(**{k: float(data.get(k, 0.0)) for k in cls.__dataclass_fields__ if k in data})


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def compute_affect(agent, *, active_ids: set[str] | None = None) -> tuple[AffectiveState, JSONDict]:
    """
    Compute affective dimensions from current agent internals.

    Returns (state, factors) where factors list the measurable inputs used.
    """
    active_ids = active_ids or set()
    factors: JSONDict = {}

    # Motivation / drives
    mot = getattr(agent, "last_motivation", None)
    social_drive = float(getattr(mot, "social_drive", 0.0) or 0.0)
    exploration = float(getattr(mot, "exploration_drive", 0.0) or 0.0)
    energy_drive = float(getattr(mot, "energy_drive", 0.0) or 0.0)
    factors["social_drive_source"] = social_drive
    factors["exploration_drive"] = exploration
    factors["energy_drive"] = energy_drive

    # Prediction error
    pe = 0.5
    if hasattr(agent, "world_model"):
        pe = float(getattr(agent.world_model, "mean_error_ema", 0.5))
    factors["prediction_error_ema"] = pe

    # Relationship uncertainty / trust / loss
    mean_unc = 0.5
    mean_trust = 0.5
    social_loss = 0.0
    if hasattr(agent, "loss_tracker") and agent.loss_tracker.records:
        # Prefer explicit computational-loss aggregate when tracker has historical entities
        social_loss = float(agent.loss_tracker.aggregate_social_loss)
        factors["social_loss_source"] = "loss_tracker"
        factors["loss_dimension_changes_recent"] = len(agent.loss_tracker.dimension_changes)
    if hasattr(agent, "relationships") and agent.relationships.relationships:
        rels = list(agent.relationships.relationships.values())
        mean_unc = sum(r.uncertainty for r in rels) / len(rels)
        mean_trust = sum(r.trust for r in rels) / len(rels)
        # Fallback social loss: strong bonds to absent agents (when tracker empty)
        if social_loss <= 0.0:
            for r in rels:
                if active_ids and r.other_id not in active_ids and r.strength > 0.4:
                    social_loss = max(social_loss, r.strength * (0.5 + 0.5 * r.trust))
            factors["social_loss_source"] = "relationship_strength_fallback"
        factors["mean_relationship_uncertainty"] = mean_unc
        factors["mean_trust"] = mean_trust
        factors["historical_strong_bonds"] = sum(
            1
            for r in rels
            if (not active_ids or r.other_id not in active_ids) and r.strength > 0.4
        )
    factors["social_loss"] = social_loss

    # Novelty from exploration + recent unique visits ratio
    novelty = exploration
    factors["novelty"] = novelty

    # Goal success from recent reward if available
    goal = 0.0
    if hasattr(agent, "total_reward") and getattr(agent, "state", None):
        age = max(1, agent.state.age)
        goal = _clamp(agent.total_reward / age, -1.0, 1.0)
    factors["reward_per_tick"] = goal

    # Security: energy remaining + trust - pe - social_loss
    energy = float(getattr(getattr(agent, "state", None), "energy", 50.0)) / 100.0
    security = _clamp(0.5 * energy + 0.3 * mean_trust - 0.2 * pe - 0.3 * social_loss, 0.0, 1.0)
    factors["energy_frac"] = energy

    # Valence: positive from goal/trust/security; negative from pe/social_loss/energy need
    valence = _clamp(
        0.35 * goal
        + 0.25 * (mean_trust - 0.5)
        + 0.25 * (security - 0.5)
        - 0.35 * pe
        - 0.45 * social_loss
        - 0.15 * energy_drive,
        -1.0,
        1.0,
    )

    # Activation: pe + social_drive + novelty + energy_drive
    activation = _clamp(0.35 * pe + 0.25 * social_drive + 0.2 * novelty + 0.2 * energy_drive, 0.0, 1.0)

    uncertainty = _clamp(0.5 * mean_unc + 0.5 * pe, 0.0, 1.0)

    state = AffectiveState(
        valence=valence,
        activation=activation,
        social_drive=social_drive,
        uncertainty=uncertainty,
        security=security,
        novelty=novelty,
        prediction_error=pe,
        social_loss=social_loss,
        goal_success=goal,
    )
    return state, factors


def affect_label(state: AffectiveState) -> tuple[str, list[str]]:
    """
    Cautious descriptive label + reasons. Not a claim of felt emotion.
    """
    reasons: list[str] = []
    if state.social_loss >= 0.45 and state.valence < -0.15:
        reasons.append(f"elevated social_loss ({state.social_loss:.2f})")
        reasons.append(f"negative valence ({state.valence:.2f})")
        if state.social_drive >= 0.4:
            reasons.append(f"elevated social_drive ({state.social_drive:.2f})")
            return "withdrawal-like / social-loss response", reasons
        return "negative-valence social-loss state", reasons
    if state.prediction_error >= 0.35 and state.uncertainty >= 0.45:
        reasons.append(f"prediction_error ({state.prediction_error:.2f})")
        reasons.append(f"uncertainty ({state.uncertainty:.2f})")
        return "uncertainty / prediction-mismatch", reasons
    if state.activation >= 0.55 and state.novelty >= 0.35 and state.valence >= -0.1:
        reasons.append(f"activation ({state.activation:.2f})")
        reasons.append(f"novelty ({state.novelty:.2f})")
        return "exploratory", reasons
    if state.social_drive >= 0.45 and state.valence >= 0.0:
        reasons.append(f"social_drive ({state.social_drive:.2f})")
        reasons.append(f"valence ({state.valence:.2f})")
        return "social engagement", reasons
    if state.valence >= 0.2 and state.security >= 0.55 and state.activation <= 0.45:
        reasons.append(f"valence ({state.valence:.2f})")
        reasons.append(f"security ({state.security:.2f})")
        return "contentment-like", reasons
    if state.valence < -0.25 and state.activation >= 0.5:
        reasons.append(f"valence ({state.valence:.2f})")
        reasons.append(f"activation ({state.activation:.2f})")
        return "distress-like (computational)", reasons
    if state.valence >= 0.0:
        reasons.append(f"valence ({state.valence:.2f})")
        return "stable / neutral-positive", reasons
    reasons.append(f"valence ({state.valence:.2f})")
    return "stable / neutral-negative", reasons


def significant_affect_delta(before: AffectiveState, after: AffectiveState) -> bool:
    """Whether an affective change is large enough to record."""
    return (
        abs(after.valence - before.valence) >= 0.08
        or abs(after.social_drive - before.social_drive) >= 0.12
        or abs(after.social_loss - before.social_loss) >= 0.08
        or abs(after.prediction_error - before.prediction_error) >= 0.1
        or abs(after.uncertainty - before.uncertainty) >= 0.1
    )
