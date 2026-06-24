"""
Created on Wed Jan 21 08:04:39 2026

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

RETRIEVAL_SCHEMA_JSON: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["type", "query", "top_k", "chunks", "blackboard_updates", "why"],
    "properties": {
        "type": {"const": "retrieval"},
        "query": {"type": "string", "minLength": 1, "maxLength": 200},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
        "chunks": {
            "type": "array",
            "minItems": 0,
            "maxItems": 10,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "source", "text"],
                "properties": {
                    "id": {"type": "string", "minLength": 1, "maxLength": 64},
                    "source": {"type": "string", "minLength": 1, "maxLength": 64},
                    "text": {"type": "string", "minLength": 1, "maxLength": 800},
                    "score": {"type": ["number", "null"]},
                    "meta": {"type": ["object", "null"]},
                },
            },
        },
        "blackboard_updates": {"type": "object"},
        "why": {"type": "string", "maxLength": 200},
    },
}