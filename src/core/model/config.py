"""
Created on Tue Dec 23 16:06:35 2025

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

###############################################################################
# Config
###############################################################################

@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: Optional[int] = None
    ffn_dim_multiplier: Optional[float] = None
    multiple_of: int = 256
    seq_len: int = 1024
    norm_eps: float = 1e-5
    use_cache: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TransformerConfig":
        return TransformerConfig(**d)
