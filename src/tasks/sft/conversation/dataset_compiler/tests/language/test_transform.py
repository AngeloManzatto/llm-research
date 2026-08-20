"""
Created on Wed Aug 19 08:11:51 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from pathlib import Path

from src.tasks.sft.conversation.dataset_compiler.knowledge.colors.base import (
    build_colors_knowledge_base,
)

from src.tasks.sft.conversation.dataset_compiler.language.transform import (
    create_grammar_transform,
)

from src.tasks.sft.conversation.dataset_compiler.templates.io import (
    load_templates_jsonl,
)


###############################################################################
# Grammar
###############################################################################

grammar_transform = create_grammar_transform(
    language="pt",
    node_role="subject",
    grammar_dir=Path(
        "src/tasks/sft/conversation/dataset_compiler/"
        "language/pt/grammar"
    ),
)


###############################################################################
# Knowledge Base
###############################################################################

kb = build_colors_knowledge_base()


###############################################################################
# Fact
###############################################################################

fact = kb.get_fact(
    "object.banana",
    "object_color",
    "color.yellow",
)


###############################################################################
# Template
###############################################################################

templates = load_templates_jsonl(
    Path(
        "src/tasks/sft/conversation/dataset_compiler/"
        "templates/knowledge_completion/object_color.jsonl"
    )
)

template = next(
    template
    for template in templates
    if template.id
    == "knowledge_completion.object_color.pt.direct_01"
)


###############################################################################
# Grammar Transform
###############################################################################

values = grammar_transform(
    kb,
    fact,
    template,
)

print(values)


###############################################################################
# Assertions
###############################################################################

assert values["subject_def_article"] == "a"
assert values["subject_indef_article"] == "uma"
assert values["subject_de_def_article"] == "da"
assert values["subject_em_def_article"] == "na"
assert values["subject_a_def_article"] == "à"

print("Portuguese grammar transform OK.")