# -*- coding: utf-8 -*-
from .base import Relation

animal_sound = Relation(
    name="animal_sound",
    category="knowledge_completion",
    facts={
        "en": {
            "dog": "A bark.", "cat": "A meow.", "cow": "A moo.", "rooster": "A crow.",
            "lion": "A roar.", "duck": "A quack.", "owl": "A hoot.", "snake": "A hiss.",
            "horse": "A neigh.", "sheep": "A baa.", "pig": "An oink.", "wolf": "A howl.",
            "bee": "A buzz.", "frog": "A croak.", "elephant": "A trumpet.",
        },
        "pt": {
            "um cachorro": "Um latido.", "um gato": "Um miado.", "uma vaca": "Um mugido.", "um galo": "Um cocoricó.",
            "um leão": "Um rugido.", "um pato": "Um quack.", "uma coruja": "Um pio.", "uma cobra": "Um chiado.",
            "um cavalo": "Um relincho.", "uma ovelha": "Um balido.", "um porco": "Um grunhido.", "um lobo": "Um uivo.",
            "uma abelha": "Um zumbido.", "um sapo": "Um coaxar.", "um elefante": "Um barrido.",
        },
    },
    templates={
        "en": [
            "What sound does a {subject} make?",
            "What noise do we associate with a {subject}?",
            "Can you tell me the sound a {subject} makes?",
        ],
        "pt": [
            "Que som {subject} faz?",
            "Qual barulho associamos a {subject}?",
            "Você sabe dizer o som que {subject} faz?",
        ],
    },
)
