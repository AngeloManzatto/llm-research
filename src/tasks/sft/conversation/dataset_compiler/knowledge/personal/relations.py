"""
Created on Sun Aug  9 16:34:24 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.core.models import (
    RelationDefinition,
)

###############################################################################
# Relations
###############################################################################

PET_NAME = RelationDefinition(
    id="pet_name",
    subject_type="pet",
    object_type="name",
)


PERSONAL_RELATIONS = (
    PET_NAME,
)