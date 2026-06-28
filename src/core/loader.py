"""
Created on Thu Jan 29 22:29:06 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import tensorflow as tf

from src.core.model.transformer import Transformer
from src.core.model.transformer import TransformerConfig
from src.core.tokenizer.tokenizer import BBPETokenizer

###############################################################################
# JSON Config Reader
###############################################################################

def load_json(path: str | Path) -> Dict[str, Any]:
    """
    Read a UTF-8 JSON file and return a Python dict.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() != ".json":
        raise ValueError(f"Expected a .json file, got: {p.name}")
    return json.loads(p.read_text(encoding="utf-8"))

###############################################################################
# Model and Tokenizer Loader
###############################################################################

@dataclass(frozen=True)
class ModelTokenizerArtifacts:
    """
    Result of loading a coupled (model + tokenizer) artifact pair.

    Attributes:
        model: Built Transformer model (variables created if build_dummy_forward=True).
        tokenizer: Loaded BBPE tokenizer instance.
        transformer_cfg: Parsed TransformerConfig used to build the model.
        tokenizer_checkpoint: Resolved path to the tokenizer checkpoint file.
        meta: Extra info useful for logging/debugging (e.g., config path).
    """
    model: tf.keras.Model
    tokenizer: BBPETokenizer
    transformer_cfg: TransformerConfig
    tokenizer_checkpoint: Path
    meta: Dict[str, Any]

def load_model_and_tokenizer(
    config_path: str | Path,
    strategy: Optional[tf.distribute.Strategy] = None,
    build_dummy_forward: bool = True,
) -> ModelTokenizerArtifacts:
    """
    Load a coupled Transformer + Tokenizer from a single JSON artifact config.

    This function is intentionally focused on the *immutable* pairing of
    (model architecture config + tokenizer checkpoint). Optimizer, datasets,
    run directories, and training hyperparameters should be handled by the
    task/run script (e.g., NTP train.py).

    Expected JSON schema (minimum):
    {
      "transformer": {...},
      "tokenizer": {"checkpoint": "..."}
    }

    Notes:

    - If transformer.vocab_size is missing/null, it is filled from
      `len(tokenizer.vocab)`.
    - If transformer.vocab_size is provided, it is validated against the
      tokenizer vocab size to prevent silent mismatches.
    - If build_dummy_forward=True, a small dummy forward pass is executed to
      ensure model variables exist (recommended before checkpoint restore).

    Returns:
        ModelTokenizerArtifacts: bundle containing model, tokenizer, config,
        checkpoint path, and small metadata.
    """
    cfg_path = Path(config_path)
    cfg = load_json(cfg_path)
    cfg_dir = cfg_path.parent

    if strategy is None:
        strategy = tf.distribute.get_strategy()

    # --- validate required keys early (better errors) ---
    if "tokenizer" not in cfg or "checkpoint" not in cfg["tokenizer"]:
        raise KeyError("Config must contain tokenizer.checkpoint")
    if "transformer" not in cfg:
        raise KeyError("Config must contain transformer")

    tokenizer_checkpoint = cfg["tokenizer"]["checkpoint"]

    with strategy.scope():
        # Tokenizer
        tokenizer = BBPETokenizer.load(tokenizer_checkpoint)

        # Transformer config
        transformer_dict = dict(cfg["transformer"])

        tok_vocab_size = len(tokenizer.vocab)

        if transformer_dict.get("vocab_size") is None:
            transformer_dict["vocab_size"] = tok_vocab_size
        else:
            if int(transformer_dict["vocab_size"]) != int(tok_vocab_size):
                raise ValueError(
                    f"vocab_size mismatch: transformer={transformer_dict['vocab_size']} "
                    f"vs tokenizer={tok_vocab_size} (checkpoint={tokenizer_checkpoint})"
                )

        transformer_cfg = TransformerConfig.from_dict(transformer_dict)

        # Model
        model = Transformer(transformer_cfg)

        if build_dummy_forward:
            _ = model(
                tf.zeros((1, transformer_cfg.seq_len), tf.int32),
                start_pos=0,
                training=False,
            )

    meta = {
        "config_path": str(cfg_path),
        "config_dir": str(cfg_dir),
        "tokenizer_vocab_size": tok_vocab_size,
    }

    return ModelTokenizerArtifacts(
        model=model,
        tokenizer=tokenizer,
        transformer_cfg=transformer_cfg,
        tokenizer_checkpoint=tokenizer_checkpoint,
        meta=meta,
    )


