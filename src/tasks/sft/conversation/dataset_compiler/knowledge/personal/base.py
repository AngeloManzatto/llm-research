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
from src.tasks.sft.conversation.dataset_compiler.knowledge.base import KnowledgeBase, build_knowledge_base
 
###############################################################################
# Files and Folders
###############################################################################

DATA_DIR = Path(__file__).resolve().parent

###############################################################################
# Build Knowledge Base
###############################################################################

def build_personal_knowledge_base() -> KnowledgeBase:
    return build_knowledge_base(DATA_DIR)

