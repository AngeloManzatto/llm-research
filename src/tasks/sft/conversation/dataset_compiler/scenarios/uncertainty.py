"""
Created on Sat Aug 15 17:26:44 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.core.models import (
    MissingFact,
)
from src.tasks.sft.conversation.dataset_compiler.knowledge.base import (
    KnowledgeBase,
)
from src.tasks.sft.conversation.dataset_compiler.scenarios.models import (
    UncertaintyScenario,
)

# TODO! Fix this with some generic pipeline
###############################################################################
# Build Pet Name Uncertainty Scenario
###############################################################################

def build_pet_name_uncertainty_scenario(
    knowledge_base: KnowledgeBase,
) -> UncertaintyScenario:
    context_fact = knowledge_base.get_fact(
        "pet.dog",
        "pet_name",
        "name.rex",
    )

    target = MissingFact(
        subject_id="pet.cat",
        relation_id="pet_name",
    )

    return UncertaintyScenario(
        context_fact=context_fact,
        target=target,
    )