"""
Created on Sun Aug  9 18:51:15 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

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
    Return the ID of another object used by the same relation.

    The returned object:
    - belongs to the same semantic relation;
    - is different from the fact's correct object;
    - has therefore already passed relation-type validation.

    Raises ValueError when no alternative object exists.
    """

    for candidate in knowledge_base.iter_facts(
        relation_id=fact.relation_id,
    ):
        if candidate.object_id != fact.object_id:
            return candidate.object_id

    raise ValueError(
        f"No alternative object available for relation "
        f"{fact.relation_id!r}."
    )

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

