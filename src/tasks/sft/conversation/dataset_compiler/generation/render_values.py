"""
Created on Sat Aug 15 07:46:08 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from collections.abc import Callable

from src.tasks.sft.conversation.dataset_compiler.core.models import Fact
from src.tasks.sft.conversation.dataset_compiler.knowledge.base import (
    KnowledgeBase,
)
from src.tasks.sft.conversation.dataset_compiler.templates.models import (
    TemplateDefinition,
)


RenderValuesProvider = Callable[
    [KnowledgeBase, Fact, TemplateDefinition],
    dict[str, str],
]


###############################################################################
# Compose Render Values
###############################################################################

def compose_render_values(
    *providers: RenderValuesProvider,
) -> RenderValuesProvider:
    """
    Compose multiple render-value providers into one provider.

    Providers are evaluated from left to right.

    Each provider must return a dictionary containing additional
    rendering values.

    Duplicate keys are rejected so that one provider cannot silently
    overwrite values produced by another provider.
    """

    def composed(
        knowledge_base: KnowledgeBase,
        fact: Fact,
        template: TemplateDefinition,
    ) -> dict[str, str]:

        values: dict[str, str] = {}

        for provider in providers:
            new_values = provider(
                knowledge_base,
                fact,
                template,
            )

            overlapping_keys = (
                values.keys()
                & new_values.keys()
            )

            if overlapping_keys:
                raise ValueError(
                    "Render-value provider collision for keys: "
                    f"{sorted(overlapping_keys)!r}."
                )

            values.update(new_values)

        return values

    return composed