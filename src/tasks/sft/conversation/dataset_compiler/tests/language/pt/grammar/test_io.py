"""
Created on Sun Aug 16 21:40:27 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from pathlib import Path

from src.tasks.sft.conversation.dataset_compiler.language.pt.grammar.io import (
    load_grammar_forms,
)

###############################################################################
# Path
###############################################################################

path = Path(
    "src/tasks/sft/conversation/dataset_compiler/"
    "language/pt/grammar/forms.jsonl"
)


###############################################################################
# Load
###############################################################################

forms = load_grammar_forms(
    path
)


###############################################################################
# Inspect
###############################################################################

print(f"Loaded: {len(forms)} forms")

for form in forms:
    print(form)


###############################################################################
# Basic assertions
###############################################################################

assert len(forms) > 0

assert all(
    "form" in item
    and "features" in item
    and "value" in item
    for item in forms
)

print("Grammar forms loader OK.")