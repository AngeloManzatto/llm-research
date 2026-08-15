"""
Created on Fri Aug 14 07:38:17 2026

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

def build_colors_knowledge_base() -> KnowledgeBase:
    return build_knowledge_base(DATA_DIR)

