"""
Created on Tue Jan 20 15:37:47 2026

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
    "You are a Summarizer agent.\n\n"
    "Compress the blackboard into a short working memory while keeping critical fields.\n"
    "Output exactly one Summary JSON object. Do not output anything else.\n"
)

###############################################################################
# User Prompt
###############################################################################

def render_user_prompt(blackboard: Dict[str, Any]) -> str:
    return (
        "BLACKBOARD:\n"
        f"{prompt_json_formatter(blackboard)}\n\n"
        "Create a short memory that preserves:\n"
        "- goal\n"
        "- plan.status / plan.current_step\n"
        "- key state variables needed to continue\n"
        "- last tool outcomes (compact)\n"
        "Keep memory concise and actionable.\n"
    )

###############################################################################
# To chatML
###############################################################################

def make_chatml(user_prompt: str, summary_obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "assistant": {"role": "assistant", "content": prompt_json_formatter(summary_obj)},
    }