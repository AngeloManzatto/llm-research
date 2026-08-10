"""
Created on Sun Aug  9 16:35:44 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

"""
Personal/local knowledge-base assembly.

This domain is intended for conversational facts that are not general
world knowledge, such as a user's pet name.
"""

from pathlib import Path

from src.tasks.sft.conversation.dataset_compiler.knowledge.base import (
    KnowledgeBase,
)
from src.tasks.sft.conversation.dataset_compiler.knowledge.io import (
    load_facts_jsonl,
    load_nodes_jsonl,
)

from src.tasks.sft.conversation.dataset_compiler.knowledge.personal.relations import PERSONAL_RELATIONS

###############################################################################
# Files and Folders
###############################################################################

DATA_DIR = Path(__file__).resolve().parent

###############################################################################
# Build Knowledge Base
###############################################################################

def build_personal_knowledge_base() -> KnowledgeBase:
    kb = KnowledgeBase()

    kb.add_relations(PERSONAL_RELATIONS)

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