"""
Created on Tue Jan 20 13:08:08 2026

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

SYSTEM_PROMPT = (
    "You are an Evaluator agent.\n\n"
    "Decide if the current plan step is done and what should happen next.\n"
    "Output exactly one Eval JSON object. Do not output anything else.\n"
)

###############################################################################
# User Prompt
###############################################################################

def render_user_prompt(blackboard: Dict[str, Any]) -> str:
    return (
        f"{render_tools_block(DEFAULT_TOOLS)}\n\n"
        "BLACKBOARD:\n"
        f"{prompt_json_formatter(blackboard)}\n\n"
        "Given plan.current_step and the latest tools.history entry, decide:\n"
        "advance / retry / needs_input / complete / fail.\n"
    )

###############################################################################
# To chatML
###############################################################################

def make_chatml(user_prompt: str, 
             evaluator_decision: Dict[str, Any]
         ) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "assistant": {"role": "assistant", "content": prompt_json_formatter(evaluator_decision)},
    }