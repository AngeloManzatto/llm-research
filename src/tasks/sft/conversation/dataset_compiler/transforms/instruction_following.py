"""
Created on Sun Aug  9 21:09:17 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.core.models import Fact
from src.tasks.sft.conversation.dataset_compiler.knowledge.base import (
    KnowledgeBase,
)
from src.tasks.sft.conversation.dataset_compiler.templates.models import (
    TemplateDefinition,
)

###############################################################################
# Upper case
###############################################################################

def uppercase_values(
    knowledge_base: KnowledgeBase,
    fact: Fact,
    template: TemplateDefinition,
) -> dict[str, str]:
    """
    Provide an uppercase version of the fact object for templates that
    require uppercase output.
    """

    object_ = knowledge_base.get_node(
        fact.object_id
    )

    object_label = object_.label(
        template.language
    )

    return {
        "object_upper": object_label.upper(),
    }