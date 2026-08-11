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
    build_fact_row, build_training_row
)
from src.tasks.sft.conversation.dataset_compiler.knowledge.base import (
    KnowledgeBase,
)
from src.tasks.sft.conversation.dataset_compiler.templates.models import (
    TemplateDefinition,
)

from src.tasks.sft.conversation.dataset_compiler.templates.render import (
    render_static_template, templates_for,
)

###############################################################################
# Type
###############################################################################

RenderValuesProvider = Callable[
    [KnowledgeBase, Fact, TemplateDefinition],
    dict[str, str],
]

###############################################################################
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

    compatible_templates = templates_for(
        templates, relation_id=relation_id, language=language,
    )

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

###############################################################################
# Compile static templates
###############################################################################

def compile_static_templates(
    *,
    templates: tuple[TemplateDefinition, ...],
) -> list[dict]:
    """
    Compile relation-free static templates into Stage 0 rows.
    """

    rows = []

    for index, template in enumerate(templates, start=1):
        if template.relation_id is not None:
            raise ValueError(
                f"Static template {template.id!r} must not define "
                "a relation ID."
            )

        messages = render_static_template(template)

        row_id = (
            f"{template.category}_"
            f"{template.language}_"
            f"{index:05d}"
        )

        rows.append(
            build_training_row(
                row_id=row_id,
                template=template,
                messages=messages,
            )
        )

    return rows