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

ANIMAL_SOUND_TEMPLATES = (
    TemplateDefinition(
        id="knowledge_completion.animal_sound.en.direct_01",
        category="knowledge_completion",
        language="en",
        relation_id="animal_sound",
        user_template="What sound does a {subject} make?",
        assistant_template="{object}.",
    ),
    TemplateDefinition(
        id="knowledge_completion.animal_sound.en.direct_02",
        category="knowledge_completion",
        language="en",
        relation_id="animal_sound",
        user_template="What noise do we associate with a {subject}?",
        assistant_template="{object}.",
    ),
    TemplateDefinition(
        id="knowledge_completion.animal_sound.pt.direct_01",
        category="knowledge_completion",
        language="pt",
        relation_id="animal_sound",
        user_template="Que som um {subject} faz?",
        assistant_template="{object}.",
    ),
    TemplateDefinition(
        id="knowledge_completion.animal_sound.pt.direct_02",
        category="knowledge_completion",
        language="pt",
        relation_id="animal_sound",
        user_template="Qual som associamos a um {subject}?",
        assistant_template="{object}.",
    ),
)