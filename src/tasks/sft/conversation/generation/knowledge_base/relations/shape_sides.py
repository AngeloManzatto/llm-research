# -*- coding: utf-8 -*-
from .base import Relation

shape_sides = Relation(
    name="shape_sides",
    category="knowledge_completion",
    facts={
        "en": {
            "a triangle": "Three.", "a square": "Four.", "a pentagon": "Five.",
            "a hexagon": "Six.", "an octagon": "Eight.", "a decagon": "Ten.",
        },
        "pt": {
            "um triângulo": "Três.", "um quadrado": "Quatro.", "um pentágono": "Cinco.",
            "um hexágono": "Seis.", "um octógono": "Oito.", "um decágono": "Dez.",
        },
    },
    templates={
        "en": [
            "How many sides does {subject} have?",
            "Can you tell me how many sides {subject} has?",
            "{subject} has how many sides?",
        ],
        "pt": [
            "Quantos lados {subject} tem?",
            "Você sabe quantos lados {subject} tem?",
            "{subject} tem quantos lados?",
        ],
    },
)
