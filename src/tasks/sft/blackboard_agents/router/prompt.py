"""
Created on Tue Jan 20 06:40:12 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

from typing import Any, Dict

from src.tasks.sft.blackboard_agents.common.utils import prompt_json_formatter

###############################################################################
# System Prompt
###############################################################################

SYSTEM_PROMPT = (
    "You are a Router agent.\n\n"
    "Choose which agent should run next to progress toward the goal.\n"
    "Output exactly one Route JSON object. Do not output anything else.\n"
)

###############################################################################
# User Prompt
###############################################################################

def render_user_prompt(blackboard: Dict[str, Any]) -> str:
    return (
        "BLACKBOARD:\n"
        f"{prompt_json_formatter(blackboard)}\n\n"
        "Choose next_agent from: retriever | planner | operator | evaluator | final.\n"
        "Use these heuristics:\n"
        "- retriever: missing context or references needed\n"
        "- planner: no plan.steps or plan.status != running\n"
        "- operator: plan.current_step exists and needs action\n"
        "- evaluator: tools.history has a fresh result to judge\n"
        "- final: goal is satisfied or needs user input\n"
    )

###############################################################################
# To chatML
###############################################################################

def make_chatml(user_prompt: str, route_obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "assistant": {"role": "assistant", "content": prompt_json_formatter(route_obj)},
    }