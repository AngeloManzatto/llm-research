"""
Created on Wed Aug 19 08:31:41 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from pathlib import Path

from src.tasks.sft.conversation.dataset_compiler.compiler.compile import (
    compile_fact_rows,
)

from src.tasks.sft.conversation.dataset_compiler.knowledge.animals.base import (
    build_animal_knowledge_base,
)

from src.tasks.sft.conversation.dataset_compiler.language.transform import (
    create_grammar_transform,
)

from src.tasks.sft.conversation.dataset_compiler.templates.io import (
    load_templates_jsonl,
)


###############################################################################
# Files
###############################################################################

GRAMMAR_DIR = Path(
    "src/tasks/sft/conversation/dataset_compiler/"
    "language/pt/grammar"
)

TEMPLATE_PATH = Path(
    "src/tasks/sft/conversation/dataset_compiler/"
    "templates/knowledge_completion/animal_baby.jsonl"
)


###############################################################################
# Knowledge Base
###############################################################################

kb = build_animal_knowledge_base()


###############################################################################
# Templates
###############################################################################

templates = load_templates_jsonl(
    TEMPLATE_PATH
)


###############################################################################
# Portuguese Grammar Transform
###############################################################################

grammar_transform = create_grammar_transform(
    language="pt",
    node_role="subject",
    grammar_dir=GRAMMAR_DIR,
)


###############################################################################
# Compile
###############################################################################

rows = compile_fact_rows(
    knowledge_base=kb,
    relation_id="animal_baby",
    templates=templates,
    transform=grammar_transform,
)


###############################################################################
# Output
###############################################################################

print(f"Generated: {len(rows)} rows")

for row in rows:
    print(row)