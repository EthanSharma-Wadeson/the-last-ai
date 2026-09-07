"""Social cognition package."""

from the_last_ai.social.graph import SocialGraph
from the_last_ai.social.propagation import (
    PropagationConfig,
    PropagationLog,
    propagate_after_interaction,
    shared_absent_contacts,
)
from the_last_ai.social.relationship import Relationship, RelationshipStore
from the_last_ai.social.resolve import resolve_pair
from the_last_ai.social.types import InteractionKind, InteractionOutcome

__all__ = [
    "InteractionKind",
    "InteractionOutcome",
    "PropagationConfig",
    "PropagationLog",
    "Relationship",
    "RelationshipStore",
    "SocialGraph",
    "propagate_after_interaction",
    "resolve_pair",
    "shared_absent_contacts",
]
