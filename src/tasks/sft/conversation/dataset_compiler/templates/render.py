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
# Render Template
###############################################################################

def render_template(
    *,
    knowledge_base: KnowledgeBase,
    fact: Fact,
    template: TemplateDefinition,
) -> list[dict[str, str]]:
    """
    Render one validated fact through one language template.

    Returns Stage-0-style user/assistant messages only.
    """

    if fact.relation_id != template.relation_id:
        raise ValueError(
            f"Fact relation {fact.relation_id!r} does not match "
            f"template relation {template.relation_id!r}."
        )

    subject = knowledge_base.get_node(fact.subject_id)
    object_ = knowledge_base.get_node(fact.object_id)

    subject_label = subject.label(template.language)
    object_label = object_.label(template.language)

    user_content = template.user_template.format(
        subject=subject_label,
        object=object_label,
    )

    assistant_content = template.assistant_template.format(
        subject=subject_label,
        object=object_label,
    )

    return [
        {
            "role": "user",
            "content": user_content,
        },
        {
            "role": "assistant",
            "content": assistant_content,
        },
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

    compatible_templates = [
        template
        for template in templates
        if template.relation_id == fact.relation_id
        and (
            language is None
            or template.language == language
        )
    ]

    return [
        render_template(
            knowledge_base=knowledge_base,
            fact=fact,
            template=template,
        )
        for template in compatible_templates
    ]