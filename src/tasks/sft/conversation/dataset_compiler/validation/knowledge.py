"""
Created on Sun Aug  9 13:21:35 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.core.graph import FactGraph
from src.tasks.sft.conversation.dataset_compiler.core.models import Fact, MissingFact, RelationDefinition
from src.tasks.sft.conversation.dataset_compiler.scenarios.models import (
    UncertaintyScenario,
)

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

###############################################################################
# Missing Fact Validation
###############################################################################

def validate_missing_fact(
    *,
    knowledge_base,
    missing_fact: MissingFact,
) -> None:
    """
    Validate that a requested fact is structurally valid and genuinely absent.

    Raises:
        KeyError:
            If the subject node or relation does not exist.

        ValueError:
            If the subject type is incompatible with the relation,
            or if the supposedly missing fact is actually present.
    """

    subject = knowledge_base.get_node(
        missing_fact.subject_id
    )

    relation = knowledge_base.get_relation(
        missing_fact.relation_id
    )

    if subject.node_type != relation.subject_type:
        raise ValueError(
            f"Relation {relation.id!r} requires subject type "
            f"{relation.subject_type!r}, but node "
            f"{subject.id!r} has type {subject.node_type!r}."
        )

    existing_facts = list(
        knowledge_base.iter_facts(
            subject_id=missing_fact.subject_id,
            relation_id=missing_fact.relation_id,
        )
    )

    if existing_facts:
        raise ValueError(
            f"Fact is not missing: subject "
            f"{missing_fact.subject_id!r} already has relation "
            f"{missing_fact.relation_id!r}."
        )

###############################################################################
# Validate uncertainty scenario
###############################################################################

def validate_uncertainty_scenario(
    *,
    knowledge_base,
    scenario: UncertaintyScenario,
) -> None:
    """
    Validate a contextual uncertainty scenario.

    Rules:
    - the context fact must exist in the knowledge base;
    - the target must genuinely be missing;
    - context and target must use the same relation;
    - context and target must refer to different subjects.
    """

    context_fact = scenario.context_fact
    target = scenario.target

    # Context fact must actually exist.
    knowledge_base.get_fact(
        context_fact.subject_id,
        context_fact.relation_id,
        context_fact.object_id,
    )

    # Target must genuinely be absent.
    validate_missing_fact(
        knowledge_base=knowledge_base,
        missing_fact=target,
    )

    # For this first uncertainty scenario, keep both facts
    # within the same semantic relation.
    if context_fact.relation_id != target.relation_id:
        raise ValueError(
            f"Uncertainty scenario relation mismatch: "
            f"context uses {context_fact.relation_id!r}, "
            f"target uses {target.relation_id!r}."
        )

    if context_fact.subject_id == target.subject_id:
        raise ValueError(
            "Uncertainty scenario context and target "
            "must refer to different subjects."
        )