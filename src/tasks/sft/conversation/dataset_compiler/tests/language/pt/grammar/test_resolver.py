"""
Created on Sun Aug 16 22:12:43 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from pathlib import Path

from src.tasks.sft.conversation.dataset_compiler.language.pt.grammar.io import (
    load_grammar_forms,
)

from src.tasks.sft.conversation.dataset_compiler.language.pt.grammar.resolver import (
    build_grammar_index,
    resolve_form,
)


path = Path(
    "src/tasks/sft/conversation/dataset_compiler/"
    "language/pt/grammar/forms.jsonl"
)

forms = load_grammar_forms(path)

grammar_index = build_grammar_index(
    forms
)

print(f"Indexed: {len(grammar_index)} forms")


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

print("Generic grammar resolver OK.")