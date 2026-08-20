"""
Created on Mon Aug 17 06:41:52 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from pathlib import Path

from src.tasks.sft.conversation.dataset_compiler.core.models import (
    Node,
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


###############################################################################
# Node
###############################################################################

node = Node(
    id="object.banana",
    node_type="object",
    labels={
        "en": "banana",
        "pt": "banana",
    },
    metadata={
        "pt_gender": "Fem",
    },
)

print(node)


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
    node=node,
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


###############################################################################
# Build Grammar Index
###############################################################################

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
# Resolve Template-ready Grammar Values
###############################################################################

values = grammar_values(
    prefix="subject",
    grammar_index=grammar_index,
    value_specs=value_specs,
    features=features,
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

assert values["subject_possessive"] == "minha"

assert values["subject_de_possessive"] == "da minha"

assert values["subject_em_possessive"] == "na minha"


###############################################################################
# Result
###############################################################################

print("Grammar values end-to-end OK.")