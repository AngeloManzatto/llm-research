#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 08:31:19 2026

@author: root
"""

###############################################################################
# Libraries
###############################################################################

from pathlib import Path

from src.tasks.sft.conversation.dataset_compiler.build.models import (
    RelationBuildSpec,
)

from src.tasks.sft.conversation.dataset_compiler.build.executor import (
    execute_relation_build,
)

from src.tasks.sft.conversation.dataset_compiler.compiler.serialization import (
    write_jsonl,
)

from src.tasks.sft.conversation.dataset_compiler.knowledge.animals.base import (
    build_animal_knowledge_base,
)

from src.tasks.sft.conversation.dataset_compiler.language.transform import (
    create_grammar_transform,
)


###############################################################################
# Files
###############################################################################

ROOT = Path(
    "src/tasks/sft/conversation/dataset_compiler"
)

GRAMMAR_DIR = (
    ROOT
    / "language"
    / "pt"
    / "grammar"
)

TEMPLATE_PATH = (
    ROOT
    / "templates"
    / "knowledge_completion"
    / "animal_baby.jsonl"
)

OUTPUT_PATH = (
    ROOT
    / "tests"
    / "pipeline"
    / "knowledge_completion_animal_baby.jsonl"
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
# Build Specification
###############################################################################

spec = RelationBuildSpec(
    id="knowledge_completion.animal_baby",
    knowledge_base_builder=build_animal_knowledge_base,
    relation_id="animal_baby",
    template_path=TEMPLATE_PATH,
    transform=grammar_transform,
)


###############################################################################
# Execute Build
###############################################################################

rows = execute_relation_build(
    spec
)

print(f"Generated: {len(rows)} rows")


###############################################################################
# Basic Dataset Assertions
###############################################################################

assert len(rows) == 8

assert all(
    row["category"] == "knowledge_completion"
    for row in rows
)

assert all(
    row["stage"] == "stage0"
    for row in rows
)

assert {
    row["language"]
    for row in rows
} == {
    "en",
    "pt",
}


###############################################################################
# Inspect Rows
###############################################################################

for row in rows:
    print(row)


###############################################################################
# Serialize
###############################################################################

written = write_jsonl(
    rows,
    OUTPUT_PATH,
)

print(f"Written: {written}")
print(f"Output: {OUTPUT_PATH}")

assert written == len(rows)


###############################################################################
# Result
###############################################################################

print(
    "Knowledge completion end-to-end build OK."
)