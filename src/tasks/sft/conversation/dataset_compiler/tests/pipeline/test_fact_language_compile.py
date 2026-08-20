"""
Created on Mon Aug 17 21:49:02 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from pathlib import Path

from src.tasks.sft.conversation.dataset_compiler.compiler.compile import (
    compile_row,
)

from src.tasks.sft.conversation.dataset_compiler.knowledge.colors.base import (
    build_colors_knowledge_base,
)

from src.tasks.sft.conversation.dataset_compiler.language.features import (
    resolve_node_features,
)

from src.tasks.sft.conversation.dataset_compiler.language.io import (
    load_json,
    load_jsonl,
    load_value_specs,
)

from src.tasks.sft.conversation.dataset_compiler.language.pt.grammar.resolver import (
    build_grammar_index,
)

from src.tasks.sft.conversation.dataset_compiler.language.values import (
    grammar_values,
)

from src.tasks.sft.conversation.dataset_compiler.resolvers.fact import (
    resolve_fact_values,
)

from src.tasks.sft.conversation.dataset_compiler.templates.io import (
    load_templates_jsonl,
)

###############################################################################
# Files
###############################################################################

GRAMMAR_DIR = Path(
    "src/tasks/sft/conversation/dataset_compiler/"
    "language/pt/grammar"
)

FORMS_PATH = (
    GRAMMAR_DIR
    / "forms.jsonl"
)

FEATURE_SPEC_PATH = (
    GRAMMAR_DIR
    / "feature_spec.json"
)

VALUE_SPECS_PATH = (
    GRAMMAR_DIR
    / "value_specs.jsonl"
)

TEMPLATE_PATH = Path(
    "src/tasks/sft/conversation/dataset_compiler/"
    "templates/knowledge_completion/object_color.jsonl"
)


###############################################################################
# Knowledge Base
###############################################################################

kb = build_colors_knowledge_base()

print(kb)

###############################################################################
# Fact
###############################################################################

fact = kb.get_fact(
    "object.banana",
    "object_color",
    "color.yellow",
)

print(fact)

###############################################################################
# Resolve Canonical Fact Values
###############################################################################

semantic_values = resolve_fact_values(
    knowledge_base=kb,
    fact=fact,
    language="pt",
)

print(semantic_values)

assert semantic_values == {
    "subject": "banana",
    "object": "amarelo",
}

###############################################################################
# Resolve Subject Node
###############################################################################

subject = kb.get_node(
    fact.subject_id
)

print(subject)

###############################################################################
# Load Feature Specification
###############################################################################

feature_spec = load_json(
    FEATURE_SPEC_PATH
)

print(feature_spec)

###############################################################################
# Resolve Node Features
###############################################################################

features = resolve_node_features(
    node=subject,
    feature_spec=feature_spec,
)

print(features)

assert features == {
    "Gender": "Fem",
    "Number": "Sing",
}


###############################################################################
# Load Grammar Forms
###############################################################################

forms = load_jsonl(
    FORMS_PATH
)

grammar_index = build_grammar_index(
    forms
)


###############################################################################
# Load Grammar Value Specifications
###############################################################################

value_specs = load_value_specs(
    VALUE_SPECS_PATH
)

print(value_specs)


###############################################################################
# Resolve Grammar Values
###############################################################################

language_values = grammar_values(
    prefix="subject",
    grammar_index=grammar_index,
    value_specs=value_specs,
    features=features,
)

print(language_values)

assert language_values["subject_de_def_article"] == "da"

###############################################################################
# Compose Final Render Values
###############################################################################

values = {
    **semantic_values,
    **language_values,
}

print(values)

###############################################################################
# Load Template
###############################################################################

templates = load_templates_jsonl(
    TEMPLATE_PATH
)

template = next(
    template
    for template in templates
    if (
        template.language == "pt"
        and template.id
        == "knowledge_completion.object_color.pt.direct_01"
    )
)

print(template)

###############################################################################
# Compile Row
###############################################################################

row = compile_row(
    row_id="knowledge_completion_pt_object_color_test_00001",
    template=template,
    values=values,
)

print(row)

###############################################################################
# Assertions
###############################################################################

assert row["category"] == "knowledge_completion"
assert row["language"] == "pt"
assert row["stage"] == "stage0"

assert row["messages"] == [
    {
        "role": "user",
        "content": "Qual é a cor da banana?",
    },
    {
        "role": "assistant",
        "content": "amarelo.",
    },
]

###############################################################################
# Result
###############################################################################

print("Generic fact -> language -> compiler pipeline OK.")