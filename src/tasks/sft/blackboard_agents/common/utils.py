"""
Created on Tue Jan 20 08:22:29 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

import json
import jsonschema
import random
from typing import Any, Dict, Optional, Callable, List

GeneratorFn = Callable[..., List[dict]]
_REGISTRY: Dict[str, GeneratorFn] = {}

###############################################################################
# Formatter
###############################################################################

def prompt_json_formatter(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True)

###############################################################################
# Validator
###############################################################################

def validate_schema(instance: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    try:
        jsonschema.validate(instance=instance, schema=schema)
    except jsonschema.ValidationError as e:
        return f"Schema failed: {e.message}"
    return None

###############################################################################
# Register
###############################################################################

def register_generator(name: str):
    def _wrap(fn: GeneratorFn) -> GeneratorFn:
        if name in _REGISTRY:
            raise ValueError(f"Generator already registered: {name}")
        _REGISTRY[name] = fn
        return fn
    return _wrap

def choose_weighted(rng: random.Random, weights: Dict[str, float]) -> str:
    items = list(weights.items())
    names = [k for k, _ in items]
    w = [max(0.0, float(v)) for _, v in items]
    total = sum(w)
    if total <= 0:
        raise ValueError("All tier weights are zero.")
    r = rng.random() * total
    acc = 0.0
    for name, wt in zip(names, w):
        acc += wt
        if r <= acc:
            return name
    return names[-1]