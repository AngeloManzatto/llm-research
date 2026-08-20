"""
Created on Sun Aug 16 17:00:48 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.compiler.compile import (
    compile_row,
)

from src.tasks.sft.conversation.dataset_compiler.templates.models import (
    MessageTemplate,
    TemplateDefinition,
)

###############################################################################
# Template
###############################################################################

template = TemplateDefinition(
    id="knowledge_completion.object_color.pt.direct_01",
    category="knowledge_completion",
    language="pt",
    relation_id="object_color",
    messages=(
        MessageTemplate(
            role="user",
            content="Qual é a cor {subject_de_def_article} {subject}?",
        ),
        MessageTemplate(
            role="assistant",
            content="{object}.",
        ),
    ),
)

###############################################################################
# Render Values
###############################################################################

values = {
    "subject": "banana",
    "object": "amarelo",
    "subject_de_def_article": "da",
}

###############################################################################
# Compile Row
###############################################################################

row = compile_row(
    row_id="knowledge_completion_pt_object_color_00001",
    template=template,
    values=values,
)

###############################################################################
# Output
###############################################################################

print(row)

###############################################################################
# Assertions
###############################################################################

assert row["id"] == "knowledge_completion_pt_object_color_00001"
assert row["category"] == "knowledge_completion"
assert row["language"] == "pt"
assert row["stage"] == "stage0"

assert row["messages"] == [
    {
        "role": "user",
        "content": "Qual é a cor da banana?",
    },
    {
        "role": "assistant",
        "content": "amarelo.",
    },
]

print("compile_row OK.")