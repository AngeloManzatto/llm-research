"""
Created on Sun Aug  9 21:37:20 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from dataclasses import dataclass

###############################################################################
# Messagem Template
###############################################################################

@dataclass(frozen=True)
class MessageTemplate:
    """
    One message inside a conversational template.

    Example:
        MessageTemplate(
            role="user",
            content="What is a baby {subject} called?",
        )
    """

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant"}:
            raise ValueError(
                f"Invalid message role: {self.role!r}."
            )

        if not self.content.strip():
            raise ValueError(
                "Message template content cannot be empty."
            )

###############################################################################
# Template Definition
###############################################################################

@dataclass(frozen=True)
class TemplateDefinition:
    """
    Natural-language conversational template for one dataset capability.

    Templates contain language and conversational structure, but no
    semantic facts.

    Example:
        TemplateDefinition(
            id="knowledge_completion.animal_baby.en.direct_01",
            category="knowledge_completion",
            language="en",
            relation_id="animal_baby",
            messages=(
                MessageTemplate(
                    role="user",
                    content="What is a baby {subject} called?",
                ),
                MessageTemplate(
                    role="assistant",
                    content="{object}.",
                ),
            ),
        )
    """

    id: str
    category: str
    language: str
    messages: tuple[MessageTemplate, ...]
    relation_id: str | None = None  # optional -- None means not knowledge/relation based

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError(
                "Template ID cannot be empty."
            )

        if not self.category.strip():
            raise ValueError(
                f"Template {self.id!r} must define a category."
            )

        if not self.language.strip():
            raise ValueError(
                f"Template {self.id!r} must define a language."
            )

        # relation_id is optional.
        # None means this template is not knowledge/relation based.
        if (
            self.relation_id is not None
            and not self.relation_id.strip()
        ):
            raise ValueError(
                f"Template {self.id!r} relation ID "
                "cannot be empty."
            )

        if not self.messages:
            raise ValueError(
                f"Template {self.id!r} must contain "
                "at least one message."
            )

        if self.messages[0].role != "user":
            raise ValueError(
                f"Template {self.id!r} must start "
                "with a user message."
            )

        if self.messages[-1].role != "assistant":
            raise ValueError(
                f"Template {self.id!r} must end "
                "with an assistant message."
            )

        for index in range(1, len(self.messages)):
            previous = self.messages[index - 1]
            current = self.messages[index]

            if previous.role == current.role:
                raise ValueError(
                    f"Template {self.id!r} messages must "
                    f"alternate roles; messages {index - 1} "
                    f"and {index} are both {current.role!r}."
                )