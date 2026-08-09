"""
Created on Sun Aug  9 17:06:39 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from dataclasses import dataclass

###############################################################################
# Template Definition
###############################################################################

@dataclass(frozen=True)
class TemplateDefinition:
    """
    Natural-language rendering template for one dataset capability.

    The template does not contain facts. It only defines how semantic
    values are expressed in a given language.

    Example:
        TemplateDefinition(
            id="knowledge_completion.animal_baby.en.direct_01",
            category="knowledge_completion",
            language="en",
            relation_id="animal_baby",
            user_template="What is a baby {subject} called?",
            assistant_template="{object}.",
        )
    """

    id: str
    category: str
    language: str
    relation_id: str
    user_template: str
    assistant_template: str

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Template ID cannot be empty.")

        if not self.category.strip():
            raise ValueError(
                f"Template {self.id!r} must define a category."
            )

        if not self.language.strip():
            raise ValueError(
                f"Template {self.id!r} must define a language."
            )

        if not self.relation_id.strip():
            raise ValueError(
                f"Template {self.id!r} must define a relation ID."
            )

        if not self.user_template.strip():
            raise ValueError(
                f"Template {self.id!r} must define a user template."
            )

        if not self.assistant_template.strip():
            raise ValueError(
                f"Template {self.id!r} must define an assistant template."
            )

        if "{subject}" not in self.user_template:
            raise ValueError(
                f"Template {self.id!r} user template must contain "
                "{subject}."
            )

        if "{object}" not in self.assistant_template:
            raise ValueError(
                f"Template {self.id!r} assistant template must contain "
                "{object}."
            )