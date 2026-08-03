"""
Created on Tue Jul 28 22:06:21 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from collections.abc import Iterable, Iterator

from src.tasks.sft.conversation.dataset_compiler.core.models import Fact, Node

###############################################################################
# Fact Graph
###############################################################################

class FactGraph:
    """
    In-memory collection of canonical nodes and directed facts.

    Example:
        graph = FactGraph()

        graph.add_node(
            Node(
                id="animal.dog",
                node_type="animal",
                labels={
                    "en": "dog",
                    "pt": "cachorro",
                },
            )
        )

        graph.add_node(
            Node(
                id="animal.puppy",
                node_type="animal_young",
                labels={
                    "en": "puppy",
                    "pt": "filhote de cachorro",
                },
            )
        )

        graph.add_fact(
            Fact(
                subject_id="animal.dog",
                relation_id="animal_baby",
                object_id="animal.puppy",
            )
        )
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._facts: dict[tuple[str, str, str], Fact] = {}

    ###########################################################################
    # Nodes
    ###########################################################################

    def add_node(self, node: Node) -> None:
        """
        Add a canonical node to the graph.

        Re-adding the exact same node is allowed and has no effect.
        Adding different node data under an existing ID is rejected.
        """
        existing = self._nodes.get(node.id)

        if existing is None:
            self._nodes[node.id] = node
            return

        if existing != node:
            raise ValueError(
                f"Node ID {node.id!r} is already registered "
                "with different data."
            )

    def add_nodes(self, nodes: Iterable[Node]) -> None:
        """
        Add multiple nodes.
        """
        for node in nodes:
            self.add_node(node)

    def get_node(self, node_id: str) -> Node:
        """
        Return a node by its canonical ID.
        """
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise KeyError(
                f"Node {node_id!r} is not registered."
            ) from exc

    def has_node(self, node_id: str) -> bool:
        """
        Return whether a node ID exists in the graph.
        """
        return node_id in self._nodes

    def iter_nodes(
        self,
        *,
        node_type: str | None = None,
    ) -> Iterator[Node]:
        """
        Iterate over nodes, optionally filtering by node type.
        """
        for node in self._nodes.values():
            if node_type is not None and node.node_type != node_type:
                continue

            yield node

    ###########################################################################
    # Facts
    ###########################################################################

    def add_fact(self, fact: Fact) -> None:
        """
        Add a fact after checking that both endpoint nodes exist.

        Re-adding the exact same fact is allowed and has no effect.
        """
        if fact.subject_id not in self._nodes:
            raise KeyError(
                f"Cannot add fact {fact.key!r}: subject node "
                f"{fact.subject_id!r} is not registered."
            )

        if fact.object_id not in self._nodes:
            raise KeyError(
                f"Cannot add fact {fact.key!r}: object node "
                f"{fact.object_id!r} is not registered."
            )

        existing = self._facts.get(fact.key)

        if existing is None:
            self._facts[fact.key] = fact
            return

        if existing != fact:
            raise ValueError(
                f"Fact {fact.key!r} is already registered "
                "with different metadata."
            )

    def add_facts(self, facts: Iterable[Fact]) -> None:
        """
        Add multiple facts.
        """
        for fact in facts:
            self.add_fact(fact)

    def has_fact(
        self,
        subject_id: str,
        relation_id: str,
        object_id: str,
    ) -> bool:
        """
        Return whether an exact semantic fact exists.
        """
        key = (subject_id, relation_id, object_id)
        return key in self._facts

    def get_fact(
        self,
        subject_id: str,
        relation_id: str,
        object_id: str,
    ) -> Fact:
        """
        Return an exact semantic fact.
        """
        key = (subject_id, relation_id, object_id)

        try:
            return self._facts[key]
        except KeyError as exc:
            raise KeyError(
                f"Fact {key!r} is not registered."
            ) from exc

    def iter_facts(
        self,
        *,
        subject_id: str | None = None,
        relation_id: str | None = None,
        object_id: str | None = None,
    ) -> Iterator[Fact]:
        """
        Iterate over facts matching the supplied filters.

        Any filter left as None is ignored.

        Examples:
            graph.iter_facts(subject_id="animal.dog")

            graph.iter_facts(relation_id="animal_baby")

            graph.iter_facts(
                relation_id="animal_baby",
                object_id="animal.puppy",
            )
        """
        for fact in self._facts.values():
            if (
                subject_id is not None
                and fact.subject_id != subject_id
            ):
                continue

            if (
                relation_id is not None
                and fact.relation_id != relation_id
            ):
                continue

            if (
                object_id is not None
                and fact.object_id != object_id
            ):
                continue

            yield fact

    ###########################################################################
    # Convenience queries
    ###########################################################################

    def objects(
        self,
        subject_id: str,
        relation_id: str,
    ) -> list[Node]:
        """
        Return object nodes connected to a subject through a relation.
        """
        return [
            self.get_node(fact.object_id)
            for fact in self.iter_facts(
                subject_id=subject_id,
                relation_id=relation_id,
            )
        ]

    def subjects(
        self,
        relation_id: str,
        object_id: str,
    ) -> list[Node]:
        """
        Return subject nodes connected to an object through a relation.
        """
        return [
            self.get_node(fact.subject_id)
            for fact in self.iter_facts(
                relation_id=relation_id,
                object_id=object_id,
            )
        ]

    ###########################################################################
    # Diagnostics
    ###########################################################################

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def fact_count(self) -> int:
        return len(self._facts)

    def relation_ids(self) -> set[str]:
        """
        Return all relation IDs currently used by graph facts.
        """
        return {
            fact.relation_id
            for fact in self._facts.values()
        }

    def __len__(self) -> int:
        """
        Return the number of facts in the graph.
        """
        return self.fact_count

    def __contains__(self, node_id: object) -> bool:
        """
        Support membership checks such as:

            "animal.dog" in graph
        """
        return isinstance(node_id, str) and node_id in self._nodes

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"nodes={self.node_count}, "
            f"facts={self.fact_count}"
            ")"
        )