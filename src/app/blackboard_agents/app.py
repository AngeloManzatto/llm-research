"""
Created on Mon Jan 19 15:30:27 2026

@author: Angelo Antonio Manzatto
"""
###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

import json
from typing import Any, Dict

import streamlit as st

###############################################################################
# Utils
###############################################################################

def pretty(x: Any) -> str:
    return json.dumps(x, indent=2, ensure_ascii=False, sort_keys=True)

###############################################################################
# Streamlit UI
###############################################################################

st.set_page_config(page_title="Blackboard Agents", layout="wide")
st.title("🧠 Blackboard Agents")

# Session state
if "blackboard" not in st.session_state:
    st.session_state.blackboard = {
        "constraints": {"workspace_root": "/workspace"},
        "goal": "Read file content for data/input.csv",
        "task": "Load file content",
        "task.status": "running",
        "state": {},
        "plan.status": "none",
        "plan.steps": [],
        "plan.current_step": None,
        "tools.history": [],
        "tools.history_compact": [],
        "state.memory": "",
    }
if "trace" not in st.session_state:
    st.session_state.trace = []
    
    
# Sidebar: Client Entry
with st.sidebar:
    
    st.header("Client entry (seed blackboard)")
    st.session_state.blackboard["goal"] = st.text_input("Goal", st.session_state.blackboard.get("goal", ""))
    st.session_state.blackboard["task"] = st.text_input("Task", st.session_state.blackboard.get("task", ""))
    
    workspace_root = st.text_input(
        "constraints.workspace_root",
        (st.session_state.blackboard.get("constraints") or {}).get("workspace_root", "/workspace"),
    )
    st.session_state.blackboard.setdefault("constraints", {})
    st.session_state.blackboard["constraints"]["workspace_root"] = workspace_root
    
    st.subheader("Seed state (JSON)")
    state_text = st.text_area("state", value=pretty(st.session_state.blackboard.get("state") or {}), height=160)
    if st.button("Apply state JSON"):
        try:
            st.session_state.blackboard["state"] = json.loads(state_text)
            st.success("state updated.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

    st.divider()
    st.header("Run controls")
    run_n = st.number_input("Run N steps", min_value=1, max_value=200, value=5, step=1)
    enable_summarizer = st.checkbox("Enable summarizer", value=True)
    debug_prompts = st.checkbox("Store prompts/raw outputs in trace", value=True)
    
    c1, c2, c3 = st.columns(3)
    if c1.button("Reset trace"):
        st.session_state.trace = []
    if c2.button("Reset Blackboard"):
        goal = st.session_state.blackboard["goal"]
        task = st.session_state.blackboard["task"]
        st.session_state.blackboard = {
            "constraints": {"workspace_root": workspace_root},
            "goal": goal,
            "task": task,
            "task.status": "running",
            "state": {},
            "plan.status": "none",
            "plan.steps": [],
            "plan.current_step": None,
            "tools.history": [],
            "tools.history_compact": [],
            "state.memory": "",
        }
        st.session_state.trace = []
    if c3.button("Mark running"):
        st.session_state.blackboard["task.status"] = "running"
        
# Main columns
left, right = st.columns([1, 1.6], gap="large")

# Left: Blackboard panels
with left:
    st.subheader("Blackboard (live)")
    st.code(pretty(st.session_state.blackboard), language="json")

    st.subheader("Edit full blackboard JSON")
    bb_text = st.text_area("blackboard_json", value=pretty(st.session_state.blackboard), height=260)
    if st.button("Apply full Blackboard JSON"):
        try:
            st.session_state.blackboard = json.loads(bb_text)
            st.success("Blackboard updated.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")