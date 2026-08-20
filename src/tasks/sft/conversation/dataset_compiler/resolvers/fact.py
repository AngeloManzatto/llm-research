"""
Created on Sun Aug 16 20:52:55 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.core.models import (
    Fact,
)

from src.tasks.sft.conversation.dataset_compiler.knowledge.base import (
    KnowledgeBase,
)

###############################################################################
# Resolve Fact Values
###############################################################################

def resolve_fact_values(
    *,
    knowledge_base: KnowledgeBase,
    fact: Fact,
    language: str,
) -> dict[str, str]:
    """
    Resolve one fact into its canonical render values.

    This function performs semantic resolution only.

    Example:

        animal.dog --animal_baby--> animal.puppy

    with language="pt"

    becomes:

        {
            "subject": "cachorro",
            "object": "cachorrinho",
        }

    Grammar, correction values, formatting transforms, and template
    rendering are intentionally handled elsewhere.
    """

    subject = knowledge_base.get_node(
        fact.subject_id
    )

    object_ = knowledge_base.get_node(
        fact.object_id
    )

    return {
        "subject": subject.label(language),
        "object": object_.label(language),
    }