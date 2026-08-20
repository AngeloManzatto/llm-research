"""
Created on Sat Aug 15 17:43:49 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from pathlib import Path

from src.tasks.sft.conversation.dataset_compiler.knowledge.personal.base import (
    build_personal_knowledge_base,
)

from src.tasks.sft.conversation.dataset_compiler.scenarios.uncertainty import (
    build_pet_name_uncertainty_scenario,
)

from src.tasks.sft.conversation.dataset_compiler.templates.io import (
    load_templates_jsonl,
)

from src.tasks.sft.conversation.dataset_compiler.language.pt.rendering import (
    portuguese_possessive_values, portuguese_uncertainty_possessive_values
)

from src.tasks.sft.conversation.dataset_compiler.scenarios.render import (
    render_uncertainty_scenario,
)

from src.tasks.sft.conversation.dataset_compiler.build.models import (
    ScenarioBuildSpec,
)

from src.tasks.sft.conversation.dataset_compiler.build.executor import (
    execute_scenario_build
)

from src.tasks.sft.conversation.dataset_compiler.build.stage0 import (
    build_stage0_with_report
)

###############################################################################
# Load personal knowledge base
###############################################################################

kb = build_personal_knowledge_base()

print(kb)

###############################################################################
# Build Scenario
###############################################################################

scenario = build_pet_name_uncertainty_scenario(
    kb
)

print(scenario)

###############################################################################
# Resolve the scenario nodes manually
###############################################################################

context_subject_node = kb.get_node(
    scenario.context_fact.subject_id
)

context_object_node = kb.get_node(
    scenario.context_fact.object_id
)

target_subject_node = kb.get_node(
    scenario.target.subject_id
)

print("CONTEXT SUBJECT:")
print(context_subject_node)

print("\nCONTEXT OBJECT:")
print(context_object_node)

print("\nTARGET SUBJECT:")
print(target_subject_node)

###############################################################################
# Gender metadata
###############################################################################

print("\nContext subject PT:")
print(context_subject_node.label("pt"))
print(context_subject_node.metadata)

print("\nContext object PT:")
print(context_object_node.label("pt"))
print(context_object_node.metadata)

print("\nTarget subject PT:")
print(target_subject_node.label("pt"))
print(target_subject_node.metadata)

###############################################################################
# Load the uncertainty templates
###############################################################################

template_path = Path(
    "src/tasks/sft/conversation/dataset_compiler/"
    "templates/uncertainty/pet_name.jsonl"
)

templates = load_templates_jsonl(
    template_path
)

print(f"Loaded: {len(templates)} templates")

for t in templates:
    print(t.id, t.language)
    
###############################################################################
# Select one template
###############################################################################    
    
template = next(
    t
    for t in templates
    if t.language == "pt"
)

print(template)

###############################################################################
# Generate possessive values for the context subject
###############################################################################    

context_values = portuguese_possessive_values(
    node=context_subject_node,
    prefix="context",
)

print(context_values)

###############################################################################
# Generate possessive values for the target subject
###############################################################################

target_values = portuguese_possessive_values(
    node=target_subject_node,
    prefix="target",
)

print(target_values)

###############################################################################
# Merge only the grammar values
###############################################################################

grammar_values = {
    **context_values,
    **target_values,
}

print(grammar_values)

###############################################################################
# Render the scenario with the extra grammar values
###############################################################################

messages = render_uncertainty_scenario(
    knowledge_base=kb,
    scenario=scenario,
    template=template,
    render_values=grammar_values,
)

for message in messages:
    print(message)
    
###############################################################################
# Scenario Specification
###############################################################################

scenario_spec = ScenarioBuildSpec(
    id="uncertainty.pet_name",
    knowledge_base_builder=build_personal_knowledge_base,
    scenario_builder=build_pet_name_uncertainty_scenario,
    template_path=Path(
        "src/tasks/sft/conversation/dataset_compiler/"
        "templates/uncertainty/pet_name.jsonl"
    ),
    render_values_provider=portuguese_uncertainty_possessive_values,
)

###############################################################################
# Build Scenario
###############################################################################

rows = execute_scenario_build(
    scenario_spec
)

print(f"Generated: {len(rows)} rows")

for row in rows:
    print(row)
    
    
rows, report = build_stage0_with_report()

for build_id, count in report.items():
    print(f"{build_id}: {count}")

print(f"TOTAL: {len(rows)}")