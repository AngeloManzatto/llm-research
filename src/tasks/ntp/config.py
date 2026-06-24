"""
Created on Wed Dec 24 08:03:44 2025

@author: recruta42
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from src.core.model.config import TransformerConfig 
from src.tasks.ntp.dataloader import NTPDatasetConfig

###############################################################################
# Data config (what to read)
###############################################################################
@dataclass(frozen=True)
class NTPDataPaths:
    # Can be:
    #   - glob string: "data/ntp/packed/**/packed_*.tfrecord"
    #   - list of globs: [...]
    #   - explicit file list (rare)
    sources: Union[str, Sequence[str]]

    # Tokenizer artifact used to create the packed shards (must match!)
    tokenizer_checkpoint: Path
    tokenizer_id: str

    # Optional: purely informational (useful for manifests)
    dataset_id: str = "mixture"  # e.g. "cc100+brwac+wiki"


@dataclass(frozen=True)
class NTPTrainPaths:
    # Root folders
    runs_root: Path = Path("runs") / "ntp"
    checkpoints_root: Path = Path("checkpoints") / "ntp"

    # Run naming
    run_name: Optional[str] = None

    # A "stable model id" (gold base) separate from runs:
    base_model_id: Optional[str] = None  # e.g. "base_8x768_cc100+brwac_v1"


###############################################################################
# Train hyperparameters (how to train)
###############################################################################
@dataclass(frozen=True)
class NTPTrainHyperparams:
    batch_size: int
    seq_len: int

    steps_per_epoch: int
    epochs: int

    learning_rate: float = 4.2e-4
    warmup_frac: float = 0.1

    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0

    # Dataloader behavior
    shuffle: bool = True
    repeat: bool = True
    assert_in_vocab: bool = False  # keep True for smoke tests

    # Performance
    num_parallel_reads: Any = None  # set in train.py to tf.data.AUTOTUNE
    num_parallel_calls: Any = None  # set in train.py to tf.data.AUTOTUNE

###############################################################################
# Train Strategy
###############################################################################
@dataclass(frozen=True)
class NTPRuntimeConfig:
    seed: int = 42
    use_mirrored_strategy: bool = True
    deterministic: bool = False   # forward to tf.data + other toggles

###############################################################################
# Full NTP job config
###############################################################################
@dataclass(frozen=True)
class NTPJobConfig:
    data: NTPDataPaths
    train: NTPTrainHyperparams
    model: TransformerConfig
    paths: NTPTrainPaths
    runtime: NTPRuntimeConfig

    def resolve_run_name(self) -> str:
        """
        Deterministic run name if not provided.
        Keep it short but informative.
        """
        if self.paths.run_name:
            return self.paths.run_name

        # Example: base_d768_l8_h12_kv4_seq1024_bs4_cc100+brwac_v1
        m = self.model
        t = self.train
        ds = self.data.dataset_id
        tok = self.data.tokenizer_id
        return (
            f"ntp_{ds}_{tok}"
            f"_d{m.d_model}_l{m.n_layers}_h{m.n_heads}_kv{m.n_kv_heads}"
            f"_seq{t.seq_len}_bs{t.batch_size}"
            f"_lr{t.learning_rate:g}"
        )

    def run_dir(self) -> Path:
        return self.paths.runs_root / self.resolve_run_name()

    def checkpoints_dir(self) -> Path:
        return self.run_dir() / "checkpoints"

    def dataset_cfg(self) -> NTPDatasetConfig:
        """
        Convert to NTPDatasetConfig used by dataloader.py
        Note: train.py can override AUTOTUNE here if desired.
        """
        return NTPDatasetConfig(
            seq_len=self.train.seq_len,
            batch_size=self.train.batch_size,
            vocab_size=self.model.vocab_size,
            shuffle=self.train.shuffle,
            repeat=self.train.repeat,
            assert_in_vocab=self.train.assert_in_vocab,
        )

    def to_dict(self) -> Dict[str, Any]:
        # Convert Paths to strings for JSON/YAML
        d = asdict(self)
        d["data"]["tokenizer_checkpoint"] = str(self.data.tokenizer_checkpoint)
        d["paths"]["runs_root"] = str(self.paths.runs_root)
        d["paths"]["checkpoints_root"] = str(self.paths.checkpoints_root)
        return d