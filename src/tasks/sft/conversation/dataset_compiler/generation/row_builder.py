"""
Created on Sun Aug  9 17:44:11 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.templates.models import (
    TemplateDefinition,
)
from src.tasks.sft.conversation.dataset_compiler.core.models import Fact
from src.tasks.sft.conversation.dataset_compiler.knowledge.base import (
    KnowledgeBase,
)

from src.tasks.sft.conversation.dataset_compiler.templates.render import (
    render_template,
)

###############################################################################
# Build training row
###############################################################################

def build_training_row(
    *,
    row_id: str,
    template: TemplateDefinition,
    messages: list[dict[str, str]],
) -> dict:
    """
    Build one Stage 0 training row from rendered messages.
    """

    return {
        "id": row_id,
        "category": template.category,
        "language": template.language,
        "stage": "stage0",
        "messages": messages,
    }

###############################################################################
# Build fact row
###############################################################################

def build_fact_row(
    *,
    row_id: str,
    knowledge_base: KnowledgeBase,
    fact: Fact,
    template: TemplateDefinition,
    values: dict[str, str] | None = None,
) -> dict:
    """
    Render one fact through one template and build a Stage 0 row.
    """

    messages = render_template(
        knowledge_base=knowledge_base,
        fact=fact,
        template=template,
        render_values=values,
    )

    return build_training_row(
        row_id=row_id,
        template=template,
        messages=messages,
    )

###############################################################################
# Build scenario row
###############################################################################

def build_scenario_row(
    *,
    row_id: str,
    template: TemplateDefinition,
    messages: list[dict[str, str]],
) -> dict:
    """
    Build one Stage 0 row from already-rendered scenario messages.
    """

    return build_training_row(
        row_id=row_id,
        template=template,
        messages=messages,
    )