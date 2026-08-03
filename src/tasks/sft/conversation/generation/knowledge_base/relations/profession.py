# -*- coding: utf-8 -*-
from .base import Relation

profession = Relation(
    name="profession",
    category="knowledge_completion",
    facts={
        "en": {
            "teaches students": "A teacher.", "flies an airplane": "A pilot.",
            "fights fires": "A firefighter.", "treats sick people": "A doctor.",
            "cooks food in a restaurant": "A chef.", "designs buildings": "An architect.",
            "defends people in court": "A lawyer.", "repairs cars": "A mechanic.",
            "cuts hair": "A barber.", "fixes electrical wiring": "An electrician.",
            "studies rocks and minerals": "A geologist.", "arranges and sells flowers": "A florist.",
        },
        "pt": {
            "ensina os alunos": "Um professor.", "pilota um avião": "Um piloto.",
            "combate incêndios": "Um bombeiro.", "trata pessoas doentes": "Um médico.",
            "cozinha em um restaurante": "Um chef.", "projeta prédios": "Um arquiteto.",
            "defende pessoas no tribunal": "Um advogado.", "conserta carros": "Um mecânico.",
            "corta cabelo": "Um cabeleireiro.", "conserta a fiação elétrica": "Um eletricista.",
            "estuda rochas e minerais": "Um geólogo.", "arruma e vende flores": "Um florista.",
        },
    },
    templates={
        "en": [
            "What do we call a person who {subject}?",
            "What is the name for someone who {subject}?",
            "If someone {subject}, what is their job called?",
        ],
        "pt": [
            "Como chamamos uma pessoa que {subject}?",
            "Qual é o nome de alguém que {subject}?",
            "Se alguém {subject}, como se chama essa profissão?",
        ],
    },
)
