"""
Created on Sun Jan 18 09:11:36 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

from src.tasks.sft.blackboard_agents.common.utils import prompt_json_formatter
from src.tasks.sft.blackboard_agents.common.tools import render_tools_block, DEFAULT_TOOLS

from typing import Any, Dict, List

###############################################################################
# System Prompt
###############################################################################

SYSTEM_PROMPT = (
    "You are a Planner agent.\n\n"
    "Your job is to write a short step-by-step plan.\n"
    "Output exactly one Plan JSON object. Do not output anything else.\n"
)

###############################################################################
# User Prompt
###############################################################################

def render_user_prompt(tools: List[Any], blackboard: Dict[str, Any]) -> str:
    return (
        f"{render_tools_block(DEFAULT_TOOLS)}\n\n"
        "BLACKBOARD:\n"
        f"{prompt_json_formatter(blackboard)}\n\n"
        "Write a plan with 1–8 steps that achieves the goal.\n"
    )

###############################################################################
# To chatML
###############################################################################

def make_chatml(user_prompt: str, planner_plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "assistant": {"role": "assistant", "content": prompt_json_formatter(planner_plan)},
    }