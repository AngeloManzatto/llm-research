"""
Created on Sat Aug 15 12:40:43 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from pathlib import Path

from src.tasks.sft.conversation.dataset_compiler.build.models import (
    RelationBuildSpec,
)

from src.tasks.sft.conversation.dataset_compiler.generation.render_values import (
    compose_render_values,
)

from src.tasks.sft.conversation.dataset_compiler.knowledge.animals.base import (
    build_animal_knowledge_base,
)

from src.tasks.sft.conversation.dataset_compiler.language.pt.rendering import (
    portuguese_subject_article_values,
)

from src.tasks.sft.conversation.dataset_compiler.transforms.correction import (
    correction_values,
)

from src.tasks.sft.conversation.dataset_compiler.build.models import (
    RelationBuildSpec,
    ScenarioBuildSpec,
)

from src.tasks.sft.conversation.dataset_compiler.knowledge.personal.base import (
    build_personal_knowledge_base,
)

from src.tasks.sft.conversation.dataset_compiler.scenarios.uncertainty import (
    build_pet_name_uncertainty_scenario,
)

from src.tasks.sft.conversation.dataset_compiler.language.pt.rendering import (
    portuguese_subject_article_values,
    portuguese_uncertainty_possessive_values,
)

###############################################################################
# Files and Folders
###############################################################################

TEMPLATE_DIR = (
    Path(__file__).resolve().parent.parent
    / "templates"
)


###############################################################################
# Relation Build Specs
###############################################################################

KNOWLEDGE_COMPLETION_ANIMAL_BABY = RelationBuildSpec(
    id="knowledge_completion.animal_baby",
    knowledge_base_builder=build_animal_knowledge_base,
    relation_id="animal_baby",
    template_path=(
        TEMPLATE_DIR
        / "knowledge_completion"
        / "animal_baby.jsonl"
    ),
    render_values_provider=portuguese_subject_article_values,
)


CORRECTION_ANIMAL_BABY = RelationBuildSpec(
    id="correction.animal_baby",
    knowledge_base_builder=build_animal_knowledge_base,
    relation_id="animal_baby",
    template_path=(
        TEMPLATE_DIR
        / "correction"
        / "animal_baby.jsonl"
    ),
    render_values_provider=compose_render_values(
        portuguese_subject_article_values,
        correction_values,
    ),
)

UNCERTAINTY_PET_NAME = ScenarioBuildSpec(
    id="uncertainty.pet_name",
    knowledge_base_builder=build_personal_knowledge_base,
    scenario_builder=build_pet_name_uncertainty_scenario,
    template_path=(
        TEMPLATE_DIR
        / "uncertainty"
        / "pet_name.jsonl"
    ),
    render_values_provider=portuguese_uncertainty_possessive_values,
)

###############################################################################
# Registry
###############################################################################

RELATION_BUILD_SPECS = (
    KNOWLEDGE_COMPLETION_ANIMAL_BABY,
    CORRECTION_ANIMAL_BABY,
)

SCENARIO_BUILD_SPECS = (
    UNCERTAINTY_PET_NAME,
)