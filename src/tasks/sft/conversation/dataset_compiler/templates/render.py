"""
Created on Sun Aug  9 17:12:20 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.core.models import Fact
from src.tasks.sft.conversation.dataset_compiler.knowledge.base import KnowledgeBase
from src.tasks.sft.conversation.dataset_compiler.templates.models import (
    TemplateDefinition,
)

###############################################################################
# Select compatible templates
###############################################################################

def templates_for(
    templates: tuple[TemplateDefinition, ...],
    *,
    relation_id: str,
    language: str | None = None,
) -> list[TemplateDefinition]:
    """
    Filter a template collection down to those matching a relation
    (and optionally a language).

    """
    return [
        template
        for template in templates
        if template.relation_id == relation_id
        and (
            language is None
            or template.language == language
        )
    ]

###############################################################################
# Render Template
###############################################################################

def render_template(
    *,
    knowledge_base: KnowledgeBase,
    fact: Fact,
    template: TemplateDefinition,
    render_values: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """
    Render one validated fact through one conversational template.

    Additional semantic values can be supplied through ``values``.
    """

    if fact.relation_id != template.relation_id:
        raise ValueError(
            f"Fact relation {fact.relation_id!r} does not match "
            f"template relation {template.relation_id!r}."
        )

    subject = knowledge_base.get_node(fact.subject_id)
    object_ = knowledge_base.get_node(fact.object_id)

    resolved_values = {
        "subject": subject.label(template.language),
        "object": object_.label(template.language),
    }

    if render_values:
        resolved_values.update(render_values)

    return [
        {
            "role": message.role,
            "content": message.content.format(**resolved_values),
        }
        for message in template.messages
    ]

###############################################################################
# Render Relation Template
###############################################################################

def render_relation_templates(
    *,
    knowledge_base: KnowledgeBase,
    relation_id: str,
    templates: tuple[TemplateDefinition, ...],
    language: str | None = None,
) -> list[list[dict[str, str]]]:
    """
    Render every fact of one relation through every compatible template.
    """

    conversations = []

    for fact in knowledge_base.iter_facts(
        relation_id=relation_id,
    ):
        conversations.extend(
            render_fact_templates(
                knowledge_base=knowledge_base,
                fact=fact,
                templates=templates,
                language=language,
            )
        )

    return conversations

###############################################################################
# Render Facts
###############################################################################

def render_fact_templates(
    *,
    knowledge_base: KnowledgeBase,
    fact: Fact,
    templates: tuple[TemplateDefinition, ...],
    language: str | None = None,
) -> list[list[dict[str, str]]]:

    compatible_templates = templates_for(
        templates, relation_id=fact.relation_id, language=language,
    )

    return [
        render_template(
            knowledge_base=knowledge_base,
            fact=fact,
            template=template,
        )
        for template in compatible_templates
    ]

###############################################################################
# Render static template
###############################################################################

def render_static_template(
    template: TemplateDefinition,
) -> list[dict[str, str]]:
    """
    Render a template that contains no semantic placeholders.
    """

    if template.relation_id is not None:
        raise ValueError(
            f"Static template {template.id!r} must not define "
            "a relation ID."
        )

    return [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in template.messages
    ]