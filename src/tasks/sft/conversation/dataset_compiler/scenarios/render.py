"""
Created on Sun Aug  9 21:40:26 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.knowledge.base import (
    KnowledgeBase,
)
from src.tasks.sft.conversation.dataset_compiler.scenarios.models import (
    UncertaintyScenario,
)
from src.tasks.sft.conversation.dataset_compiler.templates.models import (
    TemplateDefinition,
)
from src.tasks.sft.conversation.dataset_compiler.validation.knowledge import (
    validate_uncertainty_scenario,
)

###############################################################################
# Render uncertainty scenario
###############################################################################

def render_uncertainty_scenario(
    *,
    knowledge_base: KnowledgeBase,
    scenario: UncertaintyScenario,
    template: TemplateDefinition,
) -> list[dict[str, str]]:
    """
    Render one validated uncertainty scenario.

    Expected semantic values:

        context_subject
        context_object
        target_subject
    """

    validate_uncertainty_scenario(
        knowledge_base=knowledge_base,
        scenario=scenario,
    )

    context_fact = scenario.context_fact
    target = scenario.target

    if template.relation_id != context_fact.relation_id:
        raise ValueError(
            f"Template relation {template.relation_id!r} does not match "
            f"scenario relation {context_fact.relation_id!r}."
        )

    context_subject = knowledge_base.get_node(
        context_fact.subject_id
    ).label(template.language)

    context_object = knowledge_base.get_node(
        context_fact.object_id
    ).label(template.language)

    target_subject = knowledge_base.get_node(
        target.subject_id
    ).label(template.language)

    render_values = {
        "context_subject": context_subject,
        "context_object": context_object,
        "target_subject": target_subject,
    }

    return [
        {
            "role": message.role,
            "content": message.content.format(**render_values),
        }
        for message in template.messages
    ]