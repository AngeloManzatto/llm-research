"""
Created on Sun Aug  9 11:25:17 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.core.models import Fact

###############################################################################
# Facts
###############################################################################

ANIMAL_FACTS = (
    Fact(
        subject_id="animal.dog",
        relation_id="animal_baby",
        object_id="animal.puppy",
    ),
    Fact(
        subject_id="animal.cat",
        relation_id="animal_baby",
        object_id="animal.kitten",
    ),
    Fact(
        subject_id="animal.dog",
        relation_id="animal_sound",
        object_id="sound.bark",
    ),
    Fact(
        subject_id="animal.cat",
        relation_id="animal_sound",
        object_id="sound.meow",
    ),
)