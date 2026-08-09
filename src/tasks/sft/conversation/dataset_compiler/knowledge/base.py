"""
Created on Sun Aug  9 14:07:46 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from collections.abc import Iterable

from src.tasks.sft.conversation.dataset_compiler.core.graph import FactGraph
from src.tasks.sft.conversation.dataset_compiler.core.models import (
    Fact,
    Node,
    RelationDefinition,
)
from src.tasks.sft.conversation.dataset_compiler.validation.knowledge import (
    validate_fact_types,
)

###############################################################################
# Knowledge Base
###############################################################################

class KnowledgeBase:
    """
    Authoritative semantic write boundary for the dataset compiler.

    Nodes and relations are registered first. Facts are validated against
    their RelationDefinition before being inserted into the underlying graph.
    """

    def __init__(self) -> None:
        self._graph = FactGraph()
        self._relations: dict[str, RelationDefinition] = {}

    ###########################################################################
    # Nodes
    ###########################################################################

    def add_node(self, node: Node) -> None:
        self._graph.add_node(node)

    def add_nodes(self, nodes: Iterable[Node]) -> None:
        for node in nodes:
            self.add_node(node)

    def get_node(self, node_id: str) -> Node:
        return self._graph.get_node(node_id)

    ###########################################################################
    # Relations
    ###########################################################################

    def add_relation(self, relation: RelationDefinition) -> None:
        existing = self._relations.get(relation.id)

        if existing is None:
            self._relations[relation.id] = relation
            return

        if existing != relation:
            raise ValueError(
                f"Relation ID {relation.id!r} is already registered "
                "with different data."
            )

    def add_relations(
        self,
        relations: Iterable[RelationDefinition],
    ) -> None:
        for relation in relations:
            self.add_relation(relation)

    def get_relation(self, relation_id: str) -> RelationDefinition:
        try:
            return self._relations[relation_id]
        except KeyError as exc:
            raise KeyError(
                f"Relation {relation_id!r} is not registered."
            ) from exc

    def has_relation(self, relation_id: str) -> bool:
        return relation_id in self._relations

    ###########################################################################
    # Facts
    ###########################################################################

    def add_fact(self, fact: Fact) -> None:
        """
        Validate a fact semantically, then insert it into the graph.
        """
        relation = self.get_relation(fact.relation_id)

        validate_fact_types(
            graph=self._graph,
            fact=fact,
            relation=relation,
        )

        self._graph.add_fact(fact)

    def add_facts(self, facts: Iterable[Fact]) -> None:
        for fact in facts:
            self.add_fact(fact)

    def get_fact(
        self,
        subject_id: str,
        relation_id: str,
        object_id: str,
    ) -> Fact:
        return self._graph.get_fact(
            subject_id,
            relation_id,
            object_id,
        )

    ###########################################################################
    # Diagnostics
    ###########################################################################

    @property
    def node_count(self) -> int:
        return self._graph.node_count

    @property
    def relation_count(self) -> int:
        return len(self._relations)

    @property
    def fact_count(self) -> int:
        return self._graph.fact_count

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"nodes={self.node_count}, "
            f"relations={self.relation_count}, "
            f"facts={self.fact_count}"
            ")"
        )