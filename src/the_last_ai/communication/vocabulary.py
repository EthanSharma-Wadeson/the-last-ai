"""Primitive vocabulary: tokens are symbols; meanings are separate concepts.

Version 1 uses a tiny fixed English-like lexicon. Later phases can replace
tokens with arbitrary symbols without rewriting the exchange pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from the_last_ai.types import JSONDict


class Concept(str, Enum):
    """Internal semantic slots — not claims of linguistic understanding."""

    GREETING = "greeting"
    AFFIRM = "affirm"
    NEGATE = "negate"
    APPROACH = "approach"
    DEPART = "depart"
    RESOURCE = "resource"
    ASSIST = "assist"
    HALT = "halt"
    LOCATION_HERE = "location_here"
    LOCATION_THERE = "location_there"
    ADDRESSEE = "addressee"
    SELF = "self"
    DELAY = "delay"
    UNKNOWN = "unknown"


# Version-1 fixed lexicon. Mapping is replaceable via Vocabulary.
DEFAULT_TOKEN_MEANINGS: dict[str, Concept] = {
    "hi": Concept.GREETING,
    "yes": Concept.AFFIRM,
    "no": Concept.NEGATE,
    "come": Concept.APPROACH,
    "go": Concept.DEPART,
    "food": Concept.RESOURCE,
    "help": Concept.ASSIST,
    "stop": Concept.HALT,
    "here": Concept.LOCATION_HERE,
    "there": Concept.LOCATION_THERE,
    "you": Concept.ADDRESSEE,
    "me": Concept.SELF,
    "wait": Concept.DELAY,
}


@dataclass
class Vocabulary:
    """Bidirectional token↔concept map. Tokens are arbitrary strings."""

    token_to_concept: dict[str, Concept] = field(
        default_factory=lambda: dict(DEFAULT_TOKEN_MEANINGS)
    )

    def meaning(self, token: str) -> Concept:
        return self.token_to_concept.get(token, Concept.UNKNOWN)

    def concepts(self, tokens: list[str]) -> list[Concept]:
        return [self.meaning(t) for t in tokens]

    def tokens_for(self, concept: Concept) -> list[str]:
        return [t for t, c in self.token_to_concept.items() if c == concept]

    def primary_token(self, concept: Concept, fallback: str = "?") -> str:
        tokens = self.tokens_for(concept)
        return tokens[0] if tokens else fallback

    @property
    def tokens(self) -> list[str]:
        return sorted(self.token_to_concept.keys())

    def to_dict(self) -> JSONDict:
        return {token: concept.value for token, concept in self.token_to_concept.items()}

    @classmethod
    def from_dict(cls, data: JSONDict) -> Vocabulary:
        mapping = {
            str(token): Concept(str(concept)) if str(concept) in Concept._value2member_map_ else Concept.UNKNOWN
            for token, concept in data.items()
        }
        return cls(token_to_concept=mapping)


DEFAULT_VOCABULARY = Vocabulary()
