"""
Created on Mon Jan 19 10:23:58 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations
import json
from typing import Any, Dict

def append_tool_history(blackboard: Dict[str, Any], entry: Dict[str, Any]) -> None:
    blackboard.setdefault("tools.history", [])
    blackboard["tools.history"].append(entry)
    
def should_summarize(blackboard: Dict[str, Any]) -> bool:
    hist = blackboard.get("tools.history") or []
    ctx = ((blackboard.get("state") or {}).get("context") or "")
    return (len(hist) >= 6) or (len(ctx) >= 1200)