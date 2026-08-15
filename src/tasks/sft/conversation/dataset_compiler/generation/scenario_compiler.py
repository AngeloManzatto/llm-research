"""
Created on Sun Aug  9 21:43:11 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.generation.row_builder import (
    build_scenario_row,
)
from src.tasks.sft.conversation.dataset_compiler.knowledge.base import (
    KnowledgeBase,
)
from src.tasks.sft.conversation.dataset_compiler.scenarios.models import (
    UncertaintyScenario,
)
from src.tasks.sft.conversation.dataset_compiler.scenarios.render import (
    render_uncertainty_scenario,
)
from src.tasks.sft.conversation.dataset_compiler.templates.models import (
    TemplateDefinition,
)
from src.tasks.sft.conversation.dataset_compiler.build.models import (
    ScenarioRenderValuesProvider,
)

###############################################################################
# Compile uncertainty scenario
###############################################################################

def compile_uncertainty_scenario(
    *,
    knowledge_base: KnowledgeBase,
    scenario: UncertaintyScenario,
    templates: tuple[TemplateDefinition, ...],
    render_values_provider: ScenarioRenderValuesProvider | None = None,
) -> list[dict]:
    """
    Compile one uncertainty scenario through all compatible templates.
    """

    rows = []

    compatible_templates = [
        template
        for template in templates
        if template.category == "uncertainty"
        and template.relation_id == scenario.target.relation_id
    ]

    context_key = scenario.context_fact.subject_id.replace(".", "_")
    target_key = scenario.target.subject_id.replace(".", "_")

    for index, template in enumerate(
        compatible_templates,
        start=1,
    ):
        render_values = {}

        if render_values_provider is not None:
            render_values = render_values_provider(
                knowledge_base,
                scenario,
                template,
            )

        messages = render_uncertainty_scenario(
            knowledge_base=knowledge_base,
            scenario=scenario,
            template=template,
            render_values=render_values,
        )

        row_id = (
            f"{template.category}_"
            f"{template.language}_"
            f"{template.relation_id}_"
            f"{context_key}_"
            f"{target_key}_"
            f"{index:03d}"
        )

        rows.append(
            build_scenario_row(
                row_id=row_id,
                template=template,
                messages=messages,
            )
        )

    return rows