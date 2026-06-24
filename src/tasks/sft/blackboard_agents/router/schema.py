"""
Created on Wed Jan 21 08:05:54 2026

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

ROUTE_SCHEMA_JSON: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["type", "next_agent", "blackboard_updates", "why"],
    "properties": {
        "type": {"const": "route"},
        "next_agent": {
            "type": "string",
            "enum": ["retriever", "planner", "operator", "evaluator", "final"],
        },
        "blackboard_updates": {"type": "object"},
        "why": {"type": "string", "maxLength": 200},
    },
}