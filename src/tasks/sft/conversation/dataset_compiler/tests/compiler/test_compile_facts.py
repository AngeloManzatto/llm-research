"""
Created on Tue Aug 18 21:24:49 2026

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

from src.tasks.sft.conversation.dataset_compiler.templates.io import (
    load_templates_jsonl,
)

###############################################################################
# Test compile facts
###############################################################################

kb = build_animal_knowledge_base()

templates = load_templates_jsonl(
    Path(
        "src/tasks/sft/conversation/dataset_compiler/"
        "templates/knowledge_completion/animal_baby.jsonl"
    )
)

rows = compile_fact_rows(
    knowledge_base=kb,
    relation_id="animal_baby",
    templates=templates,
    language="en",
)

print(f"Generated: {len(rows)} rows")

for row in rows:
    print(row)