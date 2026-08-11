"""
Created on Sun Aug  9 14:07:46 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from collections.abc import Iterable
from pathlib import Path

from src.tasks.sft.conversation.dataset_compiler.core.graph import FactGraph
from src.tasks.sft.conversation.dataset_compiler.core.models import (
    Fact,
    Node,
    RelationDefinition,
)
from src.tasks.sft.conversation.dataset_compiler.knowledge.io import (
    load_facts_jsonl,
    load_nodes_jsonl,
    load_relations_jsonl,
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
    # Fact iterator
    ###########################################################################
    def iter_facts(
        self,
        *,
        subject_id: str | None = None,
        relation_id: str | None = None,
        object_id: str | None = None,
    ):
        return self._graph.iter_facts(
            subject_id=subject_id,
            relation_id=relation_id,
            object_id=object_id,
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

###############################################################################
# Generic knowledge base builder
###############################################################################

def build_knowledge_base(data_dir: Path) -> KnowledgeBase:
    """
    Build a fully-loaded KnowledgeBase from one domain's relations.jsonl,
    nodes.jsonl, and facts.jsonl -- all three now load by convention from
    the same directory, so a domain's base.py needs nothing beyond its
    own DATA_DIR. No relations.py file or import required per domain
    anymore; relations.jsonl replaces it the same way nodes.jsonl/
    facts.jsonl already replaced hand-written node/fact modules.
    """
    kb = KnowledgeBase()
    kb.add_relations(load_relations_jsonl(data_dir / "relations.jsonl"))
    kb.add_nodes(load_nodes_jsonl(data_dir / "nodes.jsonl"))
    kb.add_facts(load_facts_jsonl(data_dir / "facts.jsonl"))
    return kb