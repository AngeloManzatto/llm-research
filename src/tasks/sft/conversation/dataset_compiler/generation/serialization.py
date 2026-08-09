"""
Created on Sun Aug  9 17:56:33 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################
"""
Serialization helpers for compiled dataset rows.

This module is intentionally unaware of knowledge, templates,
relations, and rendering. It only writes already-compiled rows.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

###############################################################################
# Write jsonl
###############################################################################

def write_jsonl(
    rows: Iterable[dict[str, Any]],
    output_path: str | Path,
) -> int:
    """
    Write rows to a UTF-8 JSONL file.

    Returns:
        Number of rows written.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
            )
            file.write("\n")
            count += 1

    return count