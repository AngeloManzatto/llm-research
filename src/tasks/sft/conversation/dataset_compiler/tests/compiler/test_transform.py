"""
Created on Wed Aug 19 07:11:04 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.compiler.transform import (
    compose_transforms,
)


###############################################################################
# Sample Enrichers
###############################################################################

def grammar_enricher(
    knowledge_base,
    fact,
    template,
):
    return {
        "subject_de_def_article": "da",
    }


def correction_enricher(
    knowledge_base,
    fact,
    template,
):
    return {
        "wrong_object": "verde",
    }


###############################################################################
# Compose
###############################################################################

transform = compose_transforms(
    grammar_enricher,
    correction_enricher,
)


###############################################################################
# Resolve
###############################################################################

values = transform(
    None,
    None,
    None,
)

print(values)


###############################################################################
# Assertions
###############################################################################

assert values == {
    "subject_de_def_article": "da",
    "wrong_object": "verde",
}


###############################################################################
# Result
###############################################################################

print("Transoforms composition OK.")