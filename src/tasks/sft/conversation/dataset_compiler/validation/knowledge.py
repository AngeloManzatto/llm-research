"""
Created on Sun Aug  9 13:21:35 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.core.graph import FactGraph
from src.tasks.sft.conversation.dataset_compiler.core.models import Fact, RelationDefinition

###############################################################################
# Fact Validation
###############################################################################

def validate_fact_types(
    *,
    graph: FactGraph,
    fact: Fact,
    relation: RelationDefinition,
) -> None:
    """
    Validate that a fact satisfies the node-type contract
    declared by its relation.

    Raises ValueError when the fact is semantically invalid.
    """

    if fact.relation_id != relation.id:
        raise ValueError(
            f"Fact relation {fact.relation_id!r} does not match "
            f"relation definition {relation.id!r}."
        )

    subject = graph.get_node(fact.subject_id)
    object_ = graph.get_node(fact.object_id)

    if subject.node_type != relation.subject_type:
        raise ValueError(
            f"Relation {relation.id!r} requires subject type "
            f"{relation.subject_type!r}, but node "
            f"{subject.id!r} has type {subject.node_type!r}."
        )

    if object_.node_type != relation.object_type:
        raise ValueError(
            f"Relation {relation.id!r} requires object type "
            f"{relation.object_type!r}, but node "
            f"{object_.id!r} has type {object_.node_type!r}."
        )