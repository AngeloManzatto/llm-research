# -*- coding: utf-8 -*-
from .base import Relation

animal_baby = Relation(
    name="animal_baby",
    category="knowledge_completion",
    facts={
        "en": {
            "dog": "A puppy.", "cat": "A kitten.", "cow": "A calf.", "sheep": "A lamb.",
            "frog": "A tadpole.", "horse": "A foal.", "kangaroo": "A joey.", "goat": "A kid.",
            "chicken": "A chick.", "duck": "A duckling.", "goose": "A gosling.", "eagle": "An eaglet.",
            "deer": "A fawn.", "pig": "A piglet.", "fox": "A kit.",
        },
        "pt": {
            "um cachorro": "Um filhote.", "um gato": "Um filhote.", "uma vaca": "Um bezerro.", "uma ovelha": "Um cordeiro.",
            "um sapo": "Um girino.", "um cavalo": "Um potro.", "um canguru": "Um filhote.", "uma cabra": "Um cabrito.",
            "uma galinha": "Um pintinho.", "um pato": "Um patinho.", "um ganso": "Um filhote de ganso.", "uma águia": "Uma aguieta.",
            "um veado": "Um filhote de veado.", "um porco": "Um leitão.", "uma raposa": "Um filhote de raposa.",
        },
    },
    templates={
        "en": [
            "What do we call a baby {subject}?",
            "What is a young {subject} called?",
            "Complete this: a baby {subject} is called a ___.",
        ],
        "pt": [
            "Como se chama o filhote de {subject}?",
            "O que é um filhote de {subject} chamado?",
            "Complete: o filhote de {subject} é chamado de ___.",
        ],
    },
)
