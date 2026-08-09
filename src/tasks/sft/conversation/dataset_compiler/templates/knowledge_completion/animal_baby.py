"""
Created on Sun Aug  9 17:08:52 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Template Definition
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.templates.models import (
    TemplateDefinition,
)

###############################################################################
# Templatates
###############################################################################

ANIMAL_BABY_TEMPLATES = (
    TemplateDefinition(
        id="knowledge_completion.animal_baby.en.direct_01",
        category="knowledge_completion",
        language="en",
        relation_id="animal_baby",
        user_template="What is a baby {subject} called?",
        assistant_template="{object}.",
    ),
    TemplateDefinition(
        id="knowledge_completion.animal_baby.en.direct_02",
        category="knowledge_completion",
        language="en",
        relation_id="animal_baby",
        user_template="What do we call the young of a {subject}?",
        assistant_template="{object}.",
    ),
    TemplateDefinition(
        id="knowledge_completion.animal_baby.pt.direct_01",
        category="knowledge_completion",
        language="pt",
        relation_id="animal_baby",
        user_template="Como se chama o filhote de um {subject}?",
        assistant_template="{object}.",
    ),
    TemplateDefinition(
        id="knowledge_completion.animal_baby.pt.direct_02",
        category="knowledge_completion",
        language="pt",
        relation_id="animal_baby",
        user_template="Qual é o nome dado ao filhote de um {subject}?",
        assistant_template="{object}.",
    ),
)