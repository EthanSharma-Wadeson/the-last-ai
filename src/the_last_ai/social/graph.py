"""Population social graph: directed edges weighted by learned relationship strength."""

from __future__ import annotations

from dataclasses import dataclass, field

from the_last_ai.social.relationship import RelationshipStore
from the_last_ai.types import JSONDict


@dataclass
class SocialGraph:
    """
    Directed graph of interaction strength.

    Edge A→B weight is A's learned relationship strength toward B.
    Indirect / multi-hop queries are supported for future socially-propagated memory work.
    """

    edges: dict[tuple[str, str], float] = field(default_factory=dict)
    # Optional annotations for future indirect memory experiments.
    edge_meta: dict[tuple[str, str], JSONDict] = field(default_factory=dict)

    def set_edge(self, source: str, target: str, weight: float, **meta: object) -> None:
        if source == target:
            return
        key = (source, target)
        self.edges[key] = max(0.0, min(1.0, weight))
        if meta:
            self.edge_meta[key] = {**self.edge_meta.get(key, {}), **meta}

    def weight(self, source: str, target: str) -> float:
        return self.edges.get((source, target), 0.0)

    def undirected_weight(self, a: str, b: str) -> float:
        return max(self.weight(a, b), self.weight(b, a))

    def neighbors(self, agent_id: str, *, min_weight: float = 0.05) -> list[tuple[str, float]]:
        found = [
            (target, weight)
            for (source, target), weight in self.edges.items()
            if source == agent_id and weight >= min_weight
        ]
        found.sort(key=lambda item: (-item[1], item[0]))
        return found

    def predecessors(self, agent_id: str, *, min_weight: float = 0.05) -> list[tuple[str, float]]:
        found = [
            (source, weight)
            for (source, target), weight in self.edges.items()
            if target == agent_id and weight >= min_weight
        ]
        found.sort(key=lambda item: (-item[1], item[0]))
        return found

    def sync_from_agent(self, agent_id: str, store: RelationshipStore) -> None:
        for other_id, rel in store.relationships.items():
            self.set_edge(
                agent_id,
                other_id,
                rel.strength,
                trust=rel.trust,
                predicted_utility=rel.predicted_utility,
                interaction_count=rel.interaction_count,
            )

    def sync_population(self, agents: dict[str, object]) -> None:
        for agent_id, agent in agents.items():
            store = getattr(agent, "relationships", None)
            if store is not None:
                self.sync_from_agent(agent_id, store)

    def two_hop_neighbors(self, agent_id: str, *, min_weight: float = 0.1) -> list[tuple[str, float, str]]:
        """
        Return (node, path_strength, via) for nodes reachable in two hops.

        Scaffolding for later: socially propagated memory via mutual contacts.
        """
        results: list[tuple[str, float, str]] = []
        direct = {n for n, _ in self.neighbors(agent_id, min_weight=min_weight)}
        for mid, w1 in self.neighbors(agent_id, min_weight=min_weight):
            for dst, w2 in self.neighbors(mid, min_weight=min_weight):
                if dst == agent_id or dst in direct:
                    continue
                results.append((dst, w1 * w2, mid))
        results.sort(key=lambda item: (-item[1], item[0], item[2]))
        return results

    def summary(self) -> JSONDict:
        return {
            "edge_count": len(self.edges),
            "nodes": sorted({n for edge in self.edges for n in edge}),
            "mean_weight": (sum(self.edges.values()) / len(self.edges)) if self.edges else 0.0,
            "edges": {f"{a}->{b}": w for (a, b), w in sorted(self.edges.items())},
        }

    def to_dict(self) -> JSONDict:
        return {
            "edges": {f"{a}->{b}": w for (a, b), w in self.edges.items()},
            "edge_meta": {f"{a}->{b}": meta for (a, b), meta in self.edge_meta.items()},
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> SocialGraph:
        graph = cls()
        for key, weight in data.get("edges", {}).items():
            a, b = key.split("->", 1)
            graph.edges[(a, b)] = float(weight)
        for key, meta in data.get("edge_meta", {}).items():
            a, b = key.split("->", 1)
            graph.edge_meta[(a, b)] = dict(meta)
        return graph
