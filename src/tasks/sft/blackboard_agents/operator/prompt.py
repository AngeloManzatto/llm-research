"""
Created on Tue Jan 20 08:27:01 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

from typing import Any, Dict

from src.tasks.sft.blackboard_agents.common.utils import prompt_json_formatter
from src.tasks.sft.blackboard_agents.common.tools import render_tools_block, DEFAULT_TOOLS

###############################################################################
# System Prompt
###############################################################################

SYSTEM_PROMPT = """You are a Tool Operator agent.

Decide the next executable action.
Output exactly one Action JSON object. Do not output anything else.
"""

###############################################################################
# User Prompt
###############################################################################

def render_user_prompt(blackboard: Dict[str, Any]) -> str:
    return (
        f"{render_tools_block(DEFAULT_TOOLS)}\n\n"
        "BLACKBOARD:\n"
        f"{prompt_json_formatter(blackboard)}\n\n"
        "Decide the next action.\n"
    )

###############################################################################
# To chatML
###############################################################################

def make_chatml(user_prompt: str, assistant_action: Dict[str, Any],) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "assistant": {
            "role": "assistant",
            "content": prompt_json_formatter(assistant_action),
        },
    }