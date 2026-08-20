"""
Created on Mon Aug 17 06:43:01 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import json

from pathlib import Path

###############################################################################
# Load JSON
###############################################################################

def load_json(
    path: str | Path,
) -> dict:
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


###############################################################################
# Load JSONL
###############################################################################

def load_jsonl(
    path: str | Path,
) -> list[dict]:
    """
    Load one JSON object per line.
    """

    path = Path(path)

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(line)
            )

    return rows
    
###############################################################################
# Load Grammar Forms
###############################################################################

def load_value_specs(
    path: str | Path,
) -> dict[str, str]:
    """
    Load grammar render-value specifications from JSONL.
    """

    path = Path(path)

    specs: dict[str, str] = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            key = item["key"]
            form = item["form"]

            if key in specs:
                raise ValueError(
                    f"Duplicate grammar value key {key!r}."
                )

            specs[key] = form

    return specs