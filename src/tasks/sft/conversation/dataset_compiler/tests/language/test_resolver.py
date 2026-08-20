#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 07:10:19 2026

@author: root
"""

###############################################################################
# Libraries
###############################################################################

from pathlib import Path

from src.tasks.sft.conversation.dataset_compiler.language.io import (
    load_jsonl,
)

from src.tasks.sft.conversation.dataset_compiler.language.pt.grammar.resolver import (
    build_grammar_index,
    resolve_form,
)

###############################################################################
# Files
###############################################################################

FORMS_PATH = Path(
    "src/tasks/sft/conversation/dataset_compiler/"
    "language/pt/grammar/forms.jsonl"
)

###############################################################################
# Load Grammar Forms
###############################################################################

forms = load_jsonl(
    FORMS_PATH
)

print(f"Loaded: {len(forms)} grammar forms")

###############################################################################
# Build Grammar Index
###############################################################################

grammar_index = build_grammar_index(
    forms
)

print(f"Indexed: {len(grammar_index)} grammar forms")

###############################################################################
# Resolve Feminine Singular Article
###############################################################################

value = resolve_form(
    grammar_index,
    form="article.de_definite",
    features={
        "Gender": "Fem",
        "Number": "Sing",
    },
)

print(value)

assert value == "da"

###############################################################################
# Resolve Masculine Singular Possessive
###############################################################################

value = resolve_form(
    grammar_index,
    form="possessive.first_person",
    features={
        "Gender": "Masc",
        "Number": "Sing",
    },
)

print(value)

assert value == "meu"


###############################################################################
# Resolve Feminine Plural Article
###############################################################################

value = resolve_form(
    grammar_index,
    form="article.definite",
    features={
        "Gender": "Fem",
        "Number": "Plur",
    },
)

print(value)

assert value == "as"

###############################################################################
# Result
###############################################################################

print("Grammar resolver end-to-end OK.")