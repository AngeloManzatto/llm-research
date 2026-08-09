"""
Created on Sun Aug  9 11:18:38 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.core.models  import Node

###############################################################################
# Entities
###############################################################################

ANIMAL_NODES = (
    Node(
        id="animal.dog",
        node_type="animal",
        labels={
            "en": "dog",
            "pt": "cachorro",
        },
    ),
    Node(
        id="animal.cat",
        node_type="animal",
        labels={
            "en": "cat",
            "pt": "gato",
        },
    ),
    Node(
        id="animal.puppy",
        node_type="animal_young",
        labels={
            "en": "puppy",
            "pt": "filhote de cachorro",
        },
    ),
    Node(
        id="animal.kitten",
        node_type="animal_young",
        labels={
            "en": "kitten",
            "pt": "filhote de gato",
        },
    ),
    Node(
        id="sound.bark",
        node_type="animal_sound",
        labels={
            "en": "bark",
            "pt": "latido",
        },
    ),
    Node(
        id="sound.meow",
        node_type="animal_sound",
        labels={
            "en": "meow",
            "pt": "miado",
        },
    ),
)