"""
Created on Sun Aug 16 09:32:31 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.compiler.render import (
    render_messages,
)

from src.tasks.sft.conversation.dataset_compiler.templates.models import (
    TemplateDefinition, MessageTemplate
)

###############################################################################
# Create Basic Template
###############################################################################

template = TemplateDefinition(
    id="test.static",
    category="turn_taking",
    language="en",
    relation_id=None,
    messages=(
        MessageTemplate(
            role="user",
            content="Good morning!",
        ),
        MessageTemplate(
            role="assistant",
            content="Good morning!",
        ),
    ),
)

messages = render_messages(
    template=template,
)

for message in messages:
    print(message)
    
###############################################################################
# Create Advanced Template Values
###############################################################################

template = TemplateDefinition(
    id="test.values",
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

messages = render_messages(
    template=template,
    values={
        "subject_de_def_article": "da",
        "subject": "banana",
        "object": "amarelo",
    },
)

for message in messages:
    print(message)