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

from pathlib import Path

from src.tasks.sft.conversation.dataset_compiler.knowledge.base import (
    KnowledgeBase,
)


from src.tasks.sft.conversation.dataset_compiler.knowledge.animals.relations import ANIMAL_RELATIONS

from src.tasks.sft.conversation.dataset_compiler.knowledge.io import (
    load_facts_jsonl,
    load_nodes_jsonl,
)

###############################################################################
# Files and Folders
###############################################################################

DATA_DIR = Path(__file__).resolve().parent

###############################################################################
# Build Knowledge Base
###############################################################################

def build_animal_knowledge_base() -> KnowledgeBase:
    kb = KnowledgeBase()

    kb.add_relations(ANIMAL_RELATIONS)

    kb.add_nodes(
        load_nodes_jsonl(
            DATA_DIR / "nodes.jsonl"
        )
    )

    kb.add_facts(
        load_facts_jsonl(
            DATA_DIR / "facts.jsonl"
        )
    )

    return kb