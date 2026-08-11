"""
Created on Sun Aug  9 16:35:44 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from pathlib import Path
from src.tasks.sft.conversation.dataset_compiler.knowledge.base import KnowledgeBase, build_knowledge_base

###############################################################################
# Files and Folders
###############################################################################

DATA_DIR = Path(__file__).resolve().parent

###############################################################################
# Build Knowledge Base
###############################################################################

def build_animal_knowledge_base() -> KnowledgeBase:
    return build_knowledge_base(DATA_DIR)