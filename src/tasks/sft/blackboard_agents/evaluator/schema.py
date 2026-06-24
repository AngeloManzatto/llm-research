"""
Created on Wed Jan 21 07:53:01 2026

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

EVAL_SCHEMA_JSON: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["type", "decision", "blackboard_updates", "why"],
    "properties": {
        "type": {"const": "eval"},
        "decision": {"type": "string", "enum": ["advance", "retry", "needs_input", "complete", "fail"]},
        "blackboard_updates": {"type": "object"},
        "why": {"type": "string", "maxLength": 200},
    },
}