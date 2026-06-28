"""
Created on Sun Jun 28 16:32:28 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.model.generation import greedy_decode
from src.benchmarks.core.special_tokens import TOKEN_BY_NAME

###############################################################################
# Text Generator
###############################################################################

@dataclass
class TextGenerator:
    model: Any
    tokenizer: Any
    decode_config: dict[str, Any]

    def generate(self, prompt: str) -> str:

        # Next token prediction strategy
        strategy = self.decode_config.get("strategy", "greedy")

        # Stop Tolens
        stop_token_names = self.decode_config.get("stop_tokens" ,[])

        stop_tokens = [
            TOKEN_BY_NAME[name].token
            for name in stop_token_names
        ]

        # Max tokens per prediction
        max_length = int(self.decode_config.get("max_length", 64))

        # Print decode status
        verbose = bool(self.decode_config.get("verbose", False))
     
        if strategy == "greedy":
            return greedy_decode(
                model=self.model,
                prompt=prompt,
                tokenizer=self.tokenizer,
                max_length=max_length,
                stop_tokens=stop_tokens,
                verbose=verbose,
            )

        raise ValueError(f"Unsupported decoding strategy: {strategy}")