"""
Created on Sun Jan 18 10:11:29 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

###############################################################################
# Tools
###############################################################################

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_schema: Dict[str, Any]
    returns_schema: Dict[str, Any]
    
FS_READ = ToolSpec(
    name="fs.read",
    description="Read a UTF-8 text file from the workspace.",
    args_schema={
        "type": "object",
        "required": ["path"],
        "properties": {"path": {"type": "string", "minLength": 1}},
        "additionalProperties": False,
    },
    returns_schema={
        "type": "object",
        "required": ["text"],
        "properties": {"text": {"type": "string"}},
        "additionalProperties": False,
    },
)

FS_WRITE = ToolSpec(
    name="fs.write",
    description="Write UTF-8 text to a file (creates parent dirs if needed).",
    args_schema={
        "type": "object",
        "required": ["path", "text"],
        "properties": {"path": {"type": "string", "minLength": 1}, "text": {"type": "string"}},
        "additionalProperties": False,
    },
    returns_schema={
        "type": "object",
        "required": ["ok"],
        "properties": {"ok": {"type": "boolean"}},
        "additionalProperties": False,
    },
)

PY_EXEC = ToolSpec(
    name="python.exec",
    description="Execute Python code in a sandbox. Return stdout and optional JSON result.",
    args_schema={
        "type": "object",
        "required": ["code"],
        "properties": {"code": {"type": "string", "minLength": 1}},
        "additionalProperties": False,
    },
    returns_schema={
        "type": "object",
        "required": ["stdout"],
        "properties": {"stdout": {"type": "string"}, "result_json": {"type": ["object", "null"]}},
        "additionalProperties": False,
    },
)

HTTP_REQUEST = ToolSpec(
    name="http.request",
    description="Make an HTTP request and return status + JSON/text response.",
    args_schema={
        "type": "object",
        "required": ["method", "url"],
        "properties": {
            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"]},
            "url": {"type": "string", "minLength": 1},
            "headers": {"type": "object"},
            "json": {"type": ["object", "null"]},
            "timeout_s": {"type": "number", "minimum": 0.1, "maximum": 120.0},
        },
        "additionalProperties": False,
    },
    returns_schema={
        "type": "object",
        "required": ["status"],
        "properties": {"status": {"type": "integer"}, "json": {"type": ["object", "null"]}, "text": {"type": ["string", "null"]}},
        "additionalProperties": False,
    },
)

###############################################################################
# Default List
###############################################################################

DEFAULT_TOOLS: List[ToolSpec] = [FS_READ, FS_WRITE, PY_EXEC, HTTP_REQUEST]
TOOLS_BY_NAME: Dict[str, ToolSpec] = {t.name: t for t in DEFAULT_TOOLS}

###############################################################################
# Tools
###############################################################################
def render_tools_block(tools: List["ToolSpec"]) -> str:
    """
    Compact tool list.
    The model already learned argument structure during training.
    """
    lines: List[str] = ["AVAILABLE TOOLS:"]
    for t in tools:
        if t.args_schema and "properties" in t.args_schema:
            args = ", ".join(t.args_schema["properties"].keys())
            lines.append(f"- {t.name}({args})")
        else:
            lines.append(f"- {t.name}")
    return "\n".join(lines)


