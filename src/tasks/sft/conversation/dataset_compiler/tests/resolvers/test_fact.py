"""
Created on Sun Aug 16 20:55:03 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.knowledge.animals.base import (
    build_animal_knowledge_base,
)

from src.tasks.sft.conversation.dataset_compiler.resolvers.fact import (
    resolve_fact_values,
)

###############################################################################
# Knowledge Base
###############################################################################

kb = build_animal_knowledge_base()

###############################################################################
# Fact
###############################################################################

fact = kb.get_fact(
    "animal.dog",
    "animal_baby",
    "animal.puppy",
)

###############################################################################
# Resolve English
###############################################################################

values_en = resolve_fact_values(
    knowledge_base=kb,
    fact=fact,
    language="en",
)

print(values_en)

assert values_en == {
    "subject": "dog",
    "object": "puppy",
}

###############################################################################
# Resolve Portuguese
###############################################################################

values_pt = resolve_fact_values(
    knowledge_base=kb,
    fact=fact,
    language="pt",
)

print(values_pt)

assert values_pt == {
    "subject": "cachorro",
    "object": "cachorrinho",
}

print("Fact resolver OK.")