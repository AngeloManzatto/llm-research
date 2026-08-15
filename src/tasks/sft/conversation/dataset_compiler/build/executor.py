"""
Created on Sat Aug 15 12:32:22 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.generation.compiler import (
    compile_relation_rows,
)

from src.tasks.sft.conversation.dataset_compiler.templates.io import (
    load_templates_jsonl,
)

from src.tasks.sft.conversation.dataset_compiler.build.models import (
    RelationBuildSpec,
    ScenarioBuildSpec,
)

from src.tasks.sft.conversation.dataset_compiler.generation.scenario_compiler import (
    compile_uncertainty_scenario,
)

###############################################################################
# Execute Relation Build
###############################################################################

def execute_relation_build(
    spec: RelationBuildSpec,
) -> list[dict]:
    """
    Execute one relation-based build specification.

    The executor coordinates existing compiler components:

        1. Build the knowledge base.
        2. Load the templates.
        3. Compile the relation.
        4. Return the generated rows.

    It contains no domain-specific generation logic.
    """

    knowledge_base = spec.knowledge_base_builder()

    templates = load_templates_jsonl(
        spec.template_path
    )

    rows = compile_relation_rows(
        knowledge_base=knowledge_base,
        relation_id=spec.relation_id,
        templates=templates,
        render_values_provider=spec.render_values_provider,
    )

    return rows

###############################################################################
# Execute Scenario Build
###############################################################################

def execute_scenario_build(
    spec: ScenarioBuildSpec,
) -> list[dict]:
    """
    Execute one scenario-based build specification.

    Steps:
        1. Build the knowledge base.
        2. Build the scenario.
        3. Load the templates.
        4. Compile the scenario.
        5. Return the generated rows.
    """

    knowledge_base = spec.knowledge_base_builder()

    scenario = spec.scenario_builder(
        knowledge_base
    )

    templates = load_templates_jsonl(
        spec.template_path
    )

    rows = compile_uncertainty_scenario(
        knowledge_base=knowledge_base,
        scenario=scenario,
        templates=templates,
        render_values_provider=spec.render_values_provider,
    )

    return rows