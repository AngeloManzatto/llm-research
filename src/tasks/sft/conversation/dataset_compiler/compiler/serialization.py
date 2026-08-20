"""
Created on Sun Aug 16 16:27:36 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import json

from collections.abc import Iterable
from pathlib import Path

###############################################################################
# Write JSONL
###############################################################################

def write_jsonl(
    rows: Iterable[dict],
    path: str | Path,
) -> int:
    """
    Write rows to a JSONL file.

    Returns the number of rows written.
    """

    output_path = Path(path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

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
                + "\n"
            )

            count += 1

    return count