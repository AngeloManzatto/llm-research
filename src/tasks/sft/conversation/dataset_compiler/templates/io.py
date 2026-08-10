"""
Created on Sun Aug  9 19:16:43 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import json
from pathlib import Path

from src.tasks.sft.conversation.dataset_compiler.templates.models import (
    MessageTemplate,
    TemplateDefinition,
)

###############################################################################
# Load Templates
###############################################################################

def load_templates_jsonl(
    path: str | Path,
) -> tuple[TemplateDefinition, ...]:
    """
    Load template definitions from a JSONL file.
    """

    path = Path(path)
    templates = []

    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_no}: {exc}"
                ) from exc

            try:
                messages = tuple(
                    MessageTemplate(
                        role=message["role"],
                        content=message["content"],
                    )
                    for message in data["messages"]
                )

                template = TemplateDefinition(
                    id=data["id"],
                    category=data["category"],
                    language=data["language"],
                    relation_id=data["relation_id"],
                    messages=messages,
                )

            except KeyError as exc:
                raise ValueError(
                    f"Missing field {exc.args[0]!r} in "
                    f"{path} at line {line_no}."
                ) from exc

            templates.append(template)

    return tuple(templates)