"""
Created on Sat Aug 15 12:22:05 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

from src.tasks.sft.conversation.dataset_compiler.generation.render_values import (
    RenderValuesProvider,
)
from src.tasks.sft.conversation.dataset_compiler.knowledge.base import (
    KnowledgeBase,
)

from src.tasks.sft.conversation.dataset_compiler.scenarios.models import (
    UncertaintyScenario,
)
from src.tasks.sft.conversation.dataset_compiler.templates.models import (
    TemplateDefinition,
)

###############################################################################
# Types
###############################################################################

KnowledgeBaseBuilder = Callable[[], KnowledgeBase]

ScenarioRenderValuesProvider = Callable[
    [KnowledgeBase, UncertaintyScenario, TemplateDefinition],
    dict[str, str],
]

###############################################################################
# Relation Build Specification
###############################################################################

@dataclass(frozen=True)
class RelationBuildSpec:
    """
    Declarative specification for one relation-based dataset build.

    Describes WHAT should be compiled, but does not perform compilation.

    Example:
        correction.animal_baby
            -> animal knowledge base
            -> relation "animal_baby"
            -> correction templates
            -> optional render-value provider
    """

    id: str
    knowledge_base_builder: KnowledgeBaseBuilder
    relation_id: str
    template_path: Path
    render_values_provider: RenderValuesProvider | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError(
                "Relation build specification ID cannot be empty."
            )

        if not self.relation_id.strip():
            raise ValueError(
                f"Build specification {self.id!r} must define "
                "a relation ID."
            )
            
            
###############################################################################
# Scenario Build Specification
###############################################################################

@dataclass(frozen=True)
class ScenarioBuildSpec:
    """
    Declarative specification for one scenario-based dataset build.
    """

    id: str
    knowledge_base_builder: KnowledgeBaseBuilder
    scenario_builder: Callable[
        [KnowledgeBase],
        UncertaintyScenario,
    ]
    template_path: Path
    render_values_provider: ScenarioRenderValuesProvider | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError(
                "Scenario build specification ID cannot be empty."
            )