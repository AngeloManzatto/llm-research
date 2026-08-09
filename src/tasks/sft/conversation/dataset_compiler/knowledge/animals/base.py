"""
Created on Sun Aug  9 16:35:44 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

"""
All facts are inserted through KnowledgeBase, so relation lookup,
semantic type validation, and structural graph validation are applied
before knowledge reaches the underlying FactGraph.
"""

from src.tasks.sft.conversation.dataset_compiler.knowledge.base import (
    KnowledgeBase,
)

from src.tasks.sft.conversation.dataset_compiler.knowledge.animals.entities  import ANIMAL_NODES
from src.tasks.sft.conversation.dataset_compiler.knowledge.animals.facts     import ANIMAL_FACTS
from src.tasks.sft.conversation.dataset_compiler.knowledge.animals.relations import ANIMAL_RELATIONS

###############################################################################
# Build Knowledge Base
###############################################################################

def build_animal_knowledge_base() -> KnowledgeBase:
    kb = KnowledgeBase()

    kb.add_nodes(ANIMAL_NODES)
    kb.add_relations(ANIMAL_RELATIONS)
    kb.add_facts(ANIMAL_FACTS)

    return kb