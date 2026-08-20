"""
Created on Wed Aug 19 07:09:54 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from collections.abc import Callable

from src.tasks.sft.conversation.dataset_compiler.core.models import (
    Fact,
)

from src.tasks.sft.conversation.dataset_compiler.knowledge.base import (
    KnowledgeBase,
)

from src.tasks.sft.conversation.dataset_compiler.templates.models import (
    TemplateDefinition,
)


###############################################################################
# Types
###############################################################################

Transforms = Callable[
    [KnowledgeBase, Fact, TemplateDefinition],
    dict[str, str],
]


###############################################################################
# Compose Transformations
###############################################################################

def compose_transforms(
    *transforms: Transforms,
) -> Transforms:
    """
    Compose multiple value transforms into one.
    
    Each transform contributes independent values.
    
    Duplicate keys are rejected.
    """

    def composed(
        knowledge_base: KnowledgeBase,
        fact: Fact,
        template: TemplateDefinition,
    ) -> dict[str, str]:

        values: dict[str, str] = {}

        for transform in transforms:

            new_values = transform(
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
                    "Transform collision for keys: "
                    f"{sorted(overlapping_keys)!r}."
                )

            values.update(new_values)

        return values

    return composed