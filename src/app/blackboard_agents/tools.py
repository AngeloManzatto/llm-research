"""
Created on Mon Jan 19 07:53:42 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations
import os, json, subprocess, requests
from typing import Any, Dict

###############################################################################
# Exception Handlers
###############################################################################

class ToolError(RuntimeError):
    pass

###############################################################################
# Tools
###############################################################################

def fs_read(args: Dict[str, Any], *, workspace_root: str) -> Dict[str, Any]:
    path = args["path"]
    full = os.path.join(workspace_root, path)
    with open(full, "r", encoding="utf-8") as f:
        return {"text": f.read()}
    
def fs_write(args: Dict[str, Any], *, workspace_root: str) -> Dict[str, Any]:
    path = args["path"]
    text = args["text"]
    full = os.path.join(workspace_root, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(text)
    return {"ok": True}

def http_request(args: Dict[str, Any]) -> Dict[str, Any]:
    method = args["method"]
    url = args["url"]
    headers = args.get("headers") or {}
    payload = args.get("json", None)
    timeout = float(args.get("timeout_s", 30.0))
    resp = requests.request(method, url, headers=headers, json=payload, timeout=timeout)
    out: Dict[str, Any] = {"status": resp.status_code, "json": None, "text": None}
    # try json else text
    try:
        out["json"] = resp.json()
    except Exception:
        out["text"] = resp.text
    return out

def python_exec(args: Dict[str, Any]) -> Dict[str, Any]:
    code = args["code"]
    # minimal: run in a subprocess python -c
    p = subprocess.run(["python", "-c", code], capture_output=True, text=True)
    stdout = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        return {"stdout": stdout, "result_json": None}
    return {"stdout": stdout, "result_json": None}

TOOLS = {
    "fs.read": fs_read,
    "fs.write": fs_write,
    "http.request": http_request,
    "python.exec": python_exec,
}