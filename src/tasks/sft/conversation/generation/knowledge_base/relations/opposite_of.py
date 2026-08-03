# -*- coding: utf-8 -*-
"""
This relation exists specifically to fix the root cause of the
"opposite of hot" duplication problem (73 confirmed entries in the
real corpus): free-text generation converges hard on a tiny handful of
"obvious" antonym pairs. A deliberately large table (25 pairs) crossed
with several templates produces genuine breadth instead — at 25 facts
x 3 templates x 2 languages, this relation alone yields 150 distinct
rows without a single one repeating.
"""

from .base import Relation

opposite_of = Relation(
    name="opposite_of",
    category="knowledge_completion",
    facts={
        "en": {
            "hot": "Cold.", "big": "Small.", "up": "Down.", "fast": "Slow.",
            "day": "Night.", "happy": "Sad.", "light": "Dark.", "full": "Empty.",
            "open": "Closed.", "wet": "Dry.", "young": "Old.", "heavy": "Light.",
            "near": "Far.", "hard": "Soft.", "loud": "Quiet.", "clean": "Dirty.",
            "strong": "Weak.", "early": "Late.", "thick": "Thin.", "wide": "Narrow.",
            "deep": "Shallow.", "smooth": "Rough.", "sharp": "Dull.", "true": "False.",
            "rich": "Poor.",
        },
        "pt": {
            "quente": "Frio.", "grande": "Pequeno.", "cima": "Baixo.", "rápido": "Lento.",
            "dia": "Noite.", "feliz": "Triste.", "claro": "Escuro.", "cheio": "Vazio.",
            "aberto": "Fechado.", "molhado": "Seco.", "jovem": "Velho.", "pesado": "Leve.",
            "perto": "Longe.", "duro": "Macio.", "alto": "Baixo.", "limpo": "Sujo.",
            "forte": "Fraco.", "cedo": "Tarde.", "grosso": "Fino.", "largo": "Estreito.",
            "profundo": "Raso.", "liso": "Áspero.", "afiado": "Cego.", "verdadeiro": "Falso.",
            "rico": "Pobre.",
        },
    },
    templates={
        "en": [
            "What is the opposite of {subject}?",
            "What word means the opposite of {subject}?",
            "Can you tell me the opposite of {subject}?",
        ],
        "pt": [
            "Qual é o oposto de {subject}?",
            "Que palavra significa o oposto de {subject}?",
            "Você sabe dizer o oposto de {subject}?",
        ],
    },
)
