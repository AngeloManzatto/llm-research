"""
Created on Wed Jan 21 08:04:12 2026

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

PLAN_SCHEMA_JSON: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["type", "steps", "blackboard_updates", "why"],
    "properties": {
        "type": {"const": "plan"},
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "intent", "done_when"],
                "properties": {
                    "id": {"type": "string", "minLength": 1, "maxLength": 32},
                    "intent": {"type": "string", "minLength": 1, "maxLength": 120},
                    "done_when": {"type": "string", "minLength": 1, "maxLength": 160},
                    "tool_hint": {
                        "type": ["string", "null"],
                        "maxLength": 64,
                        "description": "Optional hint for operator; should match a tool name if present.",
                    },
                },
            },
        },
        "blackboard_updates": {"type": "object"},
        "why": {"type": "string", "maxLength": 200},
    },
}