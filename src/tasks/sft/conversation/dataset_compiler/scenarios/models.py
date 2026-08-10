"""
Created on Sun Aug  9 21:37:20 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from dataclasses import dataclass

from src.tasks.sft.conversation.dataset_compiler.core.models import (
    Fact,
    MissingFact,
)

###############################################################################
# Uncertainty Scenario
###############################################################################

@dataclass(frozen=True)
class UncertaintyScenario:
    """
    Contextual uncertainty scenario.

    The conversation contains one known fact, while the requested
    target fact is explicitly absent from the current knowledge state.

    Example:

        context_fact:
            pet.dog --pet_name--> name.rex

        target:
            pet.cat --pet_name--> ?
    """

    context_fact: Fact
    target: MissingFact