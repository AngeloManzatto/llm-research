"""
Created on Sun Aug  9 16:34:24 2026

@author: root
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

ANIMAL_BABY = RelationDefinition(
    id="animal_baby",
    subject_type="animal",
    object_type="animal_young",
)


ANIMAL_SOUND = RelationDefinition(
    id="animal_sound",
    subject_type="animal",
    object_type="animal_sound",
)


ANIMAL_RELATIONS = (
    ANIMAL_BABY,
    ANIMAL_SOUND,
)