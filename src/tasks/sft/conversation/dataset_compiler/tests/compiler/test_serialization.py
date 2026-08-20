"""
Created on Sun Aug 16 16:39:50 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from pathlib import Path

from src.tasks.sft.conversation.dataset_compiler.compiler.serialization import (
    write_jsonl,
)

###############################################################################
# Rows samples
###############################################################################

rows = [
    {
        "id": "test_001",
        "category": "turn_taking",
        "language": "pt",
        "stage": "stage0",
        "messages": [
            {
                "role": "user",
                "content": "Bom dia!",
            },
            {
                "role": "assistant",
                "content": "Bom dia!",
            },
        ],
    },
]


output_path = Path(
    "src/tasks/sft/conversation/dataset_compiler/tests/compiler/test_serialization.jsonl"
)

written = write_jsonl(
    rows,
    output_path,
)

print(f"Written: {written}")
print(output_path.read_text(encoding="utf-8"))