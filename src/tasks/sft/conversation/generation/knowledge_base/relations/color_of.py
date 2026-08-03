# -*- coding: utf-8 -*-
from .base import Relation

color_of = Relation(
    name="color_of",
    category="knowledge_completion",
    facts={
        "en": {
            "grass": "Green.", "coal": "Black.", "a lemon": "Yellow.", "a carrot": "Orange.",
            "an eggplant": "Purple.", "chocolate": "Brown.", "snow": "White.", "a strawberry": "Red.",
            "a banana": "Yellow.", "a blueberry": "Blue.",
        },
        "pt": {
            "a grama": "Verde.", "o carvão": "Preto.", "um limão siciliano": "Amarelo.", "uma cenoura": "Laranja.",
            "uma berinjela": "Roxa.", "o chocolate": "Marrom.", "a neve": "Branca.", "um morango": "Vermelho.",
            "uma banana": "Amarela.", "um mirtilo": "Azul.",
        },
    },
    templates={
        "en": [
            "What color is {subject}?",
            "What color would you say {subject} is?",
            "Can you tell me the color of {subject}?",
        ],
        "pt": [
            "De que cor é {subject}?",
            "Qual cor você diria que {subject} tem?",
            "Você sabe dizer a cor de {subject}?",
        ],
    },
)
