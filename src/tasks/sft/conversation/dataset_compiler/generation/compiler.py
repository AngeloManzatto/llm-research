"""
Created on Sun Aug  9 17:49:15 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from collections.abc import Callable

from src.tasks.sft.conversation.dataset_compiler.core.models import Fact
from src.tasks.sft.conversation.dataset_compiler.generation.row_builder import (
    build_fact_row,
)
from src.tasks.sft.conversation.dataset_compiler.knowledge.base import (
    KnowledgeBase,
)
from src.tasks.sft.conversation.dataset_compiler.templates.models import (
    TemplateDefinition,
)

###############################################################################
# Type
###############################################################################

RenderValuesProvider = Callable[
    [KnowledgeBase, Fact, TemplateDefinition],
    dict[str, str],
]
################
# ###############################################################
# Compile Relation Rows
###############################################################################

def compile_relation_rows(
    *,
    knowledge_base: KnowledgeBase,
    relation_id: str,
    templates: tuple[TemplateDefinition, ...],
    language: str | None = None,
    render_values_provider: RenderValuesProvider | None = None,
) -> list[dict]:
    """
    Compile all facts of one relation through all compatible templates.

    ``value_provider`` may supply additional semantic rendering values,
    such as ``wrong_object`` for correction templates.
    """

    rows = []
    index = 0

    compatible_templates = [
        template
        for template in templates
        if template.relation_id == relation_id
        and (
            language is None
            or template.language == language
        )
    ]

    for fact in knowledge_base.iter_facts(
        relation_id=relation_id,
    ):
        for template in compatible_templates:
            index += 1

            render_values = {}

            if render_values_provider is not None:
                render_values = render_values_provider(
                    knowledge_base,
                    fact,
                    template,
                )

            row_id = (
                f"{template.category}_"
                f"{template.language}_"
                f"{relation_id}_"
                f"{index:05d}"
            )

            rows.append(
                build_fact_row(
                    row_id=row_id,
                    knowledge_base=knowledge_base,
                    fact=fact,
                    template=template,
                    values=render_values,
                )
            )

    return rows