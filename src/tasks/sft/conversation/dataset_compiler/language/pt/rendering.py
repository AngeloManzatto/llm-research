
"""
Created on Sat Aug 15 07:45:14 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.core.models import Node

from src.tasks.sft.conversation.dataset_compiler.language.pt.grammar.articles import (
    contraction_de,
    contraction_em,
    contraction_a,
    definite_article,
    indefinite_article,
)

from src.tasks.sft.conversation.dataset_compiler.language.pt.grammar.possessives import (
    contraction_de_possessive,
    contraction_em_possessive,
    possessive,
)

from src.tasks.sft.conversation.dataset_compiler.core.models import Fact
from src.tasks.sft.conversation.dataset_compiler.knowledge.base import KnowledgeBase
from src.tasks.sft.conversation.dataset_compiler.templates.models import TemplateDefinition
from src.tasks.sft.conversation.dataset_compiler.scenarios.models import (
    UncertaintyScenario,
)
###############################################################################
# Portuguese Article Values
###############################################################################

def portuguese_article_values(
    *,
    node: Node,
    prefix: str,
) -> dict[str, str]:
    """
    Build Portuguese article/contraction render values for one node.

    Example:
        node label: "cenoura"
        pt_gender: "Fem"
        prefix: "subject"

    Returns:
        {
            "subject_def_article": "a",
            "subject_indef_article": "uma",
            "subject_de_def_article": "da",
            "subject_em_def_article": "na",
            "subject_a_def_article": "a",
        }
    """

    if not prefix.strip():
        raise ValueError("Render-value prefix cannot be empty.")

    try:
        gender = node.metadata["pt_gender"]
    except KeyError as exc:
        raise ValueError(
            f"Node {node.id!r} does not define 'pt_gender'."
        ) from exc

    return {
        f"{prefix}_def_article"   : definite_article(gender),
        f"{prefix}_indef_article" : indefinite_article(gender),
        f"{prefix}_de_def_article": contraction_de(gender),
        f"{prefix}_em_def_article": contraction_em(gender),
        f"{prefix}_a_def_article" : contraction_a(gender),
    }

###############################################################################
# Portuguese Subject Article Values
###############################################################################

def portuguese_subject_article_values(
    knowledge_base: KnowledgeBase,
    fact: Fact,
    template: TemplateDefinition,
) -> dict[str, str]:
    """
    Provide Portuguese grammatical article values for the fact subject.

    Non-Portuguese templates receive no additional values.
    """

    if template.language != "pt":
        return {}

    subject = knowledge_base.get_node(
        fact.subject_id
    )

    return portuguese_article_values(
        node=subject,
        prefix="subject",
    )

###############################################################################
# Portuguese Possessive Values
###############################################################################

def portuguese_possessive_values(
    *,
    node: Node,
    prefix: str,
) -> dict[str, str]:
    """
    Build Portuguese possessive render values for one node.

    Example:
        node label: "cenoura"
        pt_gender: "Fem"
        prefix: "subject"

    Returns:
        {
            "subject_possessive": "meu",
            "subject_de_possessive": "do meu",
            "subject_em_possessive": "no meu"
        }
    """

    if not prefix.strip():
        raise ValueError("Render-value prefix cannot be empty.")

    try:
        gender = node.metadata["pt_gender"]
    except KeyError as exc:
        raise ValueError(
            f"Node {node.id!r} does not define 'pt_gender'."
        ) from exc

    return {
        f"{prefix}_possessive"    : possessive(gender),
        f"{prefix}_de_possessive" : contraction_de_possessive(gender),
        f"{prefix}_em_possessive" : contraction_em_possessive(gender)
    }

###############################################################################
# Portuguese Possessive Article Values
###############################################################################

def portuguese_subject_possessive_values(
    knowledge_base: KnowledgeBase,
    fact: Fact,
    template: TemplateDefinition,
) -> dict[str, str]:
    """
    Provide Portuguese possessive render values for the fact subject.

    Non-Portuguese templates receive no additional values.
    """

    if template.language != "pt":
        return {}

    subject = knowledge_base.get_node(
        fact.subject_id
    )

    return portuguese_possessive_values(
        node=subject,
        prefix="subject",
    )

###############################################################################
# Portuguese Uncertainty Possessive Values
###############################################################################

def portuguese_uncertainty_possessive_values(
    knowledge_base: KnowledgeBase,
    scenario: UncertaintyScenario,
    template: TemplateDefinition,
) -> dict[str, str]:
    """
    Provide Portuguese possessive values for uncertainty scenario
    context and target subjects.
    """

    if template.language != "pt":
        return {}

    context_subject = knowledge_base.get_node(
        scenario.context_fact.subject_id
    )

    target_subject = knowledge_base.get_node(
        scenario.target.subject_id
    )

    values = {}

    values.update(
        portuguese_possessive_values(
            node=context_subject,
            prefix="context",
        )
    )

    values.update(
        portuguese_possessive_values(
            node=target_subject,
            prefix="target",
        )
    )

    return values