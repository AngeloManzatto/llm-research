"""
Created on Sun Aug 16 16:47:03 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.compiler.render import (
    render_messages,
)

from src.tasks.sft.conversation.dataset_compiler.compiler.transform import (
    Transforms,
)

from src.tasks.sft.conversation.dataset_compiler.knowledge.base import (
    KnowledgeBase,
)

from src.tasks.sft.conversation.dataset_compiler.resolvers.fact import (
    resolve_fact_values,
)

from src.tasks.sft.conversation.dataset_compiler.templates.models import (
    TemplateDefinition,
)

from src.tasks.sft.conversation.dataset_compiler.generation.render_values import (
    RenderValuesProvider,
)

###############################################################################
# Compile Row
###############################################################################

def compile_row(
    *,
    row_id: str,
    template: TemplateDefinition,
    values: dict[str, str] | None = None,
) -> dict:
    """
    Compile one template and its already-resolved values into
    one Stage 0 training row.
    """

    messages = render_messages(
        template=template,
        values=values,
    )

    return {
        "id": row_id,
        "category": template.category,
        "language": template.language,
        "stage": "stage0",
        "messages": messages,
    }

###############################################################################
# Compile Rows
###############################################################################

def compile_rows(
    *,
    items: list[
        tuple[
            str,
            TemplateDefinition,
            dict[str, str] | None,
        ]
    ],
) -> list[dict]:
    """
    Compile multiple prepared template/value combinations.
    """

    return [
        compile_row(
            row_id=row_id,
            template=template,
            values=values,
        )
        for row_id, template, values in items
    ]

###############################################################################
# Compile Fact Rows
###############################################################################

def compile_fact_rows(
    *,
    knowledge_base: KnowledgeBase,
    relation_id: str,
    templates: tuple[TemplateDefinition, ...],
    language: str | None = None,
    render_values_provider: RenderValuesProvider | None = None,
    transform: Transforms | None = None
) -> list[dict]:
    """
    Compile all facts of one relation through compatible templates.

    Pipeline:

        Fact
          ↓
        resolve_fact_values()
          ↓
        optional render-value provider
          ↓
        compile_row()
          ↓
        Stage 0 rows
    """

    rows: list[dict] = []

    compatible_templates = [
        template
        for template in templates
        if template.relation_id == relation_id
        and (
            language is None
            or template.language == language
        )
    ]

    index = 0

    for fact in knowledge_base.iter_facts(
        relation_id=relation_id,
    ):
        for template in compatible_templates:

            index += 1

            # Canonical semantic values
            values = resolve_fact_values(
                knowledge_base=knowledge_base,
                fact=fact,
                language=template.language,
            )
            
            if transform is not None:
                extra_values = transform(
                    knowledge_base,
                    fact,
                    template,
                )
            
                overlapping_keys = (
                    values.keys()
                    & extra_values.keys()
                )
            
                if overlapping_keys:
                    raise ValueError(
                        "Transform collision for keys: "
                        f"{sorted(overlapping_keys)!r}."
                    )
            
                values.update(extra_values)


            if render_values_provider is not None:

                extra_values = render_values_provider(
                    knowledge_base,
                    fact,
                    template,
                )

                overlapping_keys = (
                    values.keys()
                    & extra_values.keys()
                )

                if overlapping_keys:
                    raise ValueError(
                        "Render-value collision for keys: "
                        f"{sorted(overlapping_keys)!r}."
                    )

                values.update(
                    extra_values
                )

            row_id = (
                f"{template.category}_"
                f"{template.language}_"
                f"{relation_id}_"
                f"{index:05d}"
            )

            rows.append(
                compile_row(
                    row_id=row_id,
                    template=template,
                    values=values,
                )
            )

    return rows