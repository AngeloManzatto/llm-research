"""
Created on Sun Aug  9 19:16:43 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

"""
Knowledge storage I/O.

This module converts JSONL records into the canonical semantic models
used by the dataset compiler.

It does not perform semantic validation or mutate a KnowledgeBase.
"""

import json
from pathlib import Path

from src.tasks.sft.conversation.dataset_compiler.core.models import (
    Fact,
    Node,
    RelationDefinition,
)

###############################################################################
# Load nodes
###############################################################################

def load_nodes_jsonl(
    path: str | Path,
) -> list[Node]:
    """
    Load canonical nodes from a JSONL file.
    """

    path = Path(path)
    nodes: list[Node] = []

    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_no}: {exc}"
                ) from exc

            try:
                node = Node(
                    id=data["id"],
                    node_type=data["node_type"],
                    labels=data["labels"],
                    metadata=data.get("metadata", {}),
                )
            except KeyError as exc:
                raise ValueError(
                    f"Missing field {exc.args[0]!r} in "
                    f"{path} at line {line_no}."
                ) from exc

            nodes.append(node)

    return nodes

###############################################################################
# Load facts
###############################################################################

def load_facts_jsonl(
    path: str | Path,
) -> list[Fact]:
    """
    Load canonical facts from a JSONL file.
    """

    path = Path(path)
    facts: list[Fact] = []

    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_no}: {exc}"
                ) from exc

            try:
                fact = Fact(
                    subject_id=data["subject_id"],
                    relation_id=data["relation_id"],
                    object_id=data["object_id"],
                    metadata=data.get("metadata", {}),
                )
            except KeyError as exc:
                raise ValueError(
                    f"Missing field {exc.args[0]!r} in "
                    f"{path} at line {line_no}."
                ) from exc

            facts.append(fact)

    return facts

###############################################################################
# Load relations
###############################################################################

def load_relations_jsonl(
    path: str | Path,
) -> tuple[RelationDefinition, ...]:
    """
    Load relation definitions from a JSONL file.

    Expected row schema:

        {
            "id": "animal_baby",
            "subject_type": "animal",
            "object_type": "animal_young"
        }
    """

    path = Path(path)
    relations: list[RelationDefinition] = []

    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_no}: {exc}"
                ) from exc

            try:
                relation = RelationDefinition(
                    id=data["id"],
                    subject_type=data["subject_type"],
                    object_type=data["object_type"],
                )
            except KeyError as exc:
                raise ValueError(
                    f"Missing field {exc.args[0]!r} in "
                    f"{path} at line {line_no}."
                ) from exc

            relations.append(relation)

    return tuple(relations)