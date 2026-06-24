"""
Created on Wed Jan 21 08:07:53 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

from typing import Dict, Any

###############################################################################
# Schema
###############################################################################

SUMMARY_SCHEMA_JSON: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["type", "memory", "keep", "drop_before", "blackboard_updates", "why"],
    "properties": {
        "type": {"const": "summary"},
        # Short working memory blob; your whole goal is to keep this small
        "memory": {"type": "string", "minLength": 1, "maxLength": 1200},
        # Names of blackboard fields that must be kept verbatim (do not compress)
        "keep": {
            "type": "array",
            "minItems": 0,
            "maxItems": 20,
            "items": {"type": "string", "minLength": 1, "maxLength": 128},
        },
        # Suggest dropping tool history entries before this index (inclusive/exclusive is your choice)
        "drop_before": {"type": ["integer", "null"], "minimum": 0},
        "blackboard_updates": {"type": "object"},
        "why": {"type": "string", "maxLength": 200},
    },
}