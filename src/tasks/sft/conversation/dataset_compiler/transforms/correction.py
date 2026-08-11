"""
Created on Sun Aug  9 18:51:15 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import random

from src.tasks.sft.conversation.dataset_compiler.core.models import Fact
from src.tasks.sft.conversation.dataset_compiler.knowledge.base import (
    KnowledgeBase,
)

from src.tasks.sft.conversation.dataset_compiler.templates.models import (
    TemplateDefinition,
)

###############################################################################
# Get wrong object
###############################################################################

def get_wrong_object(
    *,
    knowledge_base: KnowledgeBase,
    fact: Fact,
) -> str:
    """
    Return the ID of a RANDOMLY chosen alternative object from the same
    semantic relation.

    The returned object:
    - belongs to the same semantic relation;
    - is different from the fact's correct object;
    - has therefore already passed relation-type validation.

    Raises ValueError when no alternative object exists.
    """

    alternatives = [
        candidate.object_id
        for candidate in knowledge_base.iter_facts(
            relation_id=fact.relation_id,
        )
        if candidate.object_id != fact.object_id
    ]

    if not alternatives:
        raise ValueError(
            f"No alternative object available for relation "
            f"{fact.relation_id!r}."
        )

    return random.choice(alternatives)

###############################################################################
# Correction Values
###############################################################################

def correction_values(
    knowledge_base: KnowledgeBase,
    fact: Fact,
    template: TemplateDefinition,
) -> dict[str, str]:
    """
    Provide additional rendering values required by correction templates.
    """

    wrong_object_id = get_wrong_object(
        knowledge_base=knowledge_base,
        fact=fact,
    )

    wrong_object = knowledge_base.get_node(
        wrong_object_id
    )

    return {
        "wrong_object": wrong_object.label(
            template.language
        ),
    }