"""
Created on Wed Jan 21 08:03:15 2026

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

ACTION_SCHEMA_JSON: Dict[str, Any] = {
    "type": "object",
    "required": ["type", "tool", "args", "blackboard_updates", "why"],
    "properties": {
        "type": {"enum": ["tool_call", "final"]},
        "tool": {"type": ["string", "null"]},
        "args": {"type": ["object", "null"]},
        "blackboard_updates": {"type": "object"},
        "why": {"type": "string", "maxLength": 200},
    },
    "additionalProperties": False,
}
