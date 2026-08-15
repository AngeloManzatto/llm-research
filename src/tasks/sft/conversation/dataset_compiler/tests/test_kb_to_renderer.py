"""
Created on Sat Aug 15 09:17:45 2026

@author: Angelo Antonio Manzatto
"""

"""

Pipeline Execution

1. build_animal_knowledge_base()
        ↓
2. kb.get_fact()
        ↓
3. kb.get_node()
        ↓
4. portuguese_article_values()
        ↓
5. load_templates_jsonl()
        ↓
6. correction_values()
        ↓
7. compose_render_values()
        ↓
8. render_template()
        ↓
9. build_training_row()
        ↓
10. write_jsonl()


SEMANTIC DATA
    Node / Relation / Fact
          ↓
LANGUAGE REALIZATION
    Portuguese grammar
          ↓
CATEGORY TRANSFORM
    correction
          ↓
      TEMPLATE
          ↓
     RENDERING
          ↓
    TRAINING ROW
          ↓
        JSONL

"""


###############################################################################
# Libraries
###############################################################################

from pathlib import Path

from src.tasks.sft.conversation.dataset_compiler.knowledge.animals.base import (
    build_animal_knowledge_base,
)

from src.tasks.sft.conversation.dataset_compiler.language.pt.rendering import (
    portuguese_article_values,
)

from src.tasks.sft.conversation.dataset_compiler.templates.io import (
    load_templates_jsonl,
)

from src.tasks.sft.conversation.dataset_compiler.transforms.correction import (
    correction_values,
)

from src.tasks.sft.conversation.dataset_compiler.generation.render_values import (
    compose_render_values,
)

from src.tasks.sft.conversation.dataset_compiler.language.pt.rendering import (
    portuguese_subject_article_values,
)

from src.tasks.sft.conversation.dataset_compiler.templates.render import (
    render_template,
)

from src.tasks.sft.conversation.dataset_compiler.generation.row_builder import (
    build_training_row,
)

from src.tasks.sft.conversation.dataset_compiler.generation.serialization import (
    write_jsonl,
)

###############################################################################
# Output file 
###############################################################################

output_path = Path(
    "data/sft/conversation/level0/compiler_test/"
    "correction_pt_animal_baby_single.jsonl"
)

###############################################################################
# Load Knowledge Base
###############################################################################

kb = build_animal_knowledge_base()

print(kb)

###############################################################################
# Select a Fact (knows WHICH nodes)
###############################################################################

fact = kb.get_fact(
    "animal.dog",
    "animal_baby",
    "animal.puppy",
)

print(fact)

###############################################################################
# Resolve the nodes referenced by the fact (knows labels + lexical metadata)
###############################################################################

subject = kb.get_node(
    fact.subject_id
)

object_ = kb.get_node(
    fact.object_id
)

print(subject)
print(object_)

###############################################################################
# Inspect the Portuguese lexical
###############################################################################

print("Subject PT:", subject.label("pt"))
print("Subject metadata:", subject.metadata)

print("Object PT:", object_.label("pt"))
print("Object metadata:", object_.metadata)


grammar_values = portuguese_article_values(
    node=subject,
    prefix="subject",
)

print(grammar_values)

###############################################################################
# Load templates
###############################################################################

template_path = Path(
    "src/tasks/sft/conversation/dataset_compiler/"
    "templates/correction/animal_baby.jsonl"
)

templates = load_templates_jsonl(
    template_path
)

print(f"Loaded: {len(templates)} templates")

###############################################################################
# Select Portuguese template
###############################################################################

template = next(
    template
    for template in templates
    if template.language == "pt"
)

print(template)

###############################################################################
# Get the category-specific values
###############################################################################

correction_render_values = correction_values(
    kb,
    fact,
    template,
)

print(correction_render_values)

###############################################################################
# Lexical PT values
###############################################################################

provider = compose_render_values(
    portuguese_subject_article_values,
    correction_values,
)

render_values = provider(
    kb,
    fact,
    template,
)

print(render_values)

###############################################################################
# Template messages
###############################################################################

messages = render_template(
    knowledge_base=kb,
    fact=fact,
    template=template,
    render_values=render_values,
)

for message in messages:
    print(message)
    
###############################################################################
# Build training row
###############################################################################   

row = build_training_row(
    row_id="correction_pt_animal_baby_00001",
    template=template,
    messages=messages,
)

print(row)

###############################################################################
# Write row on jsonl file
############################################################################### 

written = write_jsonl(
    [row],
    output_path,
)

print(f"Written: {written}")
print(f"Path: {output_path}")