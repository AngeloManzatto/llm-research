"""
Created on Tue Jul 28 21:26:51 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from dataclasses import dataclass, field
from typing import Any

###############################################################################
# Node
###############################################################################

@dataclass(frozen=True)
class Node:
    """
    A canonical entity or value in the knowledge base.

    The node ID is stable and language-independent. Labels contain
    the natural-language forms used when rendering the node.

    Example:
        Node(
            id="animal.dog",
            labels={
                "en": "dog",
                "pt": "cachorro",
            },
        )
    """

    id: str
    labels: dict[str, str]
    node_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Node ID cannot be empty.")

        if not self.node_type.strip():
            raise ValueError(
                f"Node {self.id!r} must have a node type."
            )

        if not self.labels:
            raise ValueError(
                f"Node {self.id!r} must define at least one label."
            )

        for language, label in self.labels.items():
            if not language.strip():
                raise ValueError(
                    f"Node {self.id!r} contains an empty language code."
                )

            if not label.strip():
                raise ValueError(
                    f"Node {self.id!r} has an empty label for "
                    f"language {language!r}."
                )

    def label(self, language: str) -> str:
        """
        Return this node's label in the requested language.
        """
        try:
            return self.labels[language]
        except KeyError as exc:
            raise KeyError(
                f"Node {self.id!r} has no label for "
                f"language {language!r}."
            ) from exc

###############################################################################
# Fact
###############################################################################

@dataclass(frozen=True)
class Fact:
    """
    A directed semantic relationship between two nodes.

    Example:
        Fact(
            subject_id="animal.dog",
            relation_id="animal_baby",
            object_id="animal.puppy",
        )

    Represents:
        dog --animal_baby--> puppy
    """

    subject_id: str
    relation_id: str
    object_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.subject_id.strip():
            raise ValueError("Fact subject ID cannot be empty.")

        if not self.relation_id.strip():
            raise ValueError("Fact relation ID cannot be empty.")

        if not self.object_id.strip():
            raise ValueError("Fact object ID cannot be empty.")

    @property
    def key(self) -> tuple[str, str, str]:
        """
        Stable semantic identity of the fact.
        """
        return (
            self.subject_id,
            self.relation_id,
            self.object_id,
        )