# -*- coding: utf-8 -*-
from .base import Relation

animal_group = Relation(
    name="animal_group",
    category="knowledge_completion",
    facts={
        "en": {
            "lions": "A pride.", "fish": "A school.", "wolves": "A pack.",
            "crows": "A murder.", "geese": "A gaggle.", "owls": "A parliament.",
            "elephants": "A herd.", "ants": "A colony.", "monkeys": "A troop.",
        },
        "pt": {
            "leões": "Um bando.", "peixes": "Um cardume.", "lobos": "Uma alcateia.",
            "corvos": "Um bando.", "gansos": "Um bando.", "corujas": "Um bando.",
            "elefantes": "Uma manada.", "formigas": "Uma colônia.", "macacos": "Um bando.",
        },
    },
    templates={
        "en": [
            "What do we call a group of {subject}?",
            "What is a group of {subject} called?",
        ],
        "pt": [
            "Como chamamos um grupo de {subject}?",
            "O que é um grupo de {subject} chamado?",
        ],
    },
)
