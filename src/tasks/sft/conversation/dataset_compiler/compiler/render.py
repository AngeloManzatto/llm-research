"""
Created on Sun Aug 16 09:13:30 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.templates.models import (
    TemplateDefinition,
)

###############################################################################
# Render Messages
###############################################################################

def render_messages(
    *,
    template: TemplateDefinition,
    values: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """
    Render one template using already-resolved values.

    This function is intentionally source-agnostic.

    It does not know whether values came from:
        - a Fact
        - a Scenario
        - grammar resolution
        - a transform
        - a static template

    It only performs template substitution.
    """

    resolved_values = values or {}

    return [
        {
            "role": message.role,
            "content": message.content.format(
                **resolved_values
            ),
        }
        for message in template.messages
    ]