"""
Created on Wed Dec 17 11:32:26 2025

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

from src.core.model.config import TransformerConfig

import tensorflow as tf

###############################################################################
# Utils
###############################################################################

def model_all_finite(model: tf.keras.Model):
    ok = True
    for v in model.trainable_variables:
        if not tf.reduce_all(tf.math.is_finite(v)):
            tf.print("❌ non-finite in var:", v.path)
            ok = False
    return ok

def optimizer_all_finite(optimizer: tf.keras.optimizers.Optimizer) -> bool:
    ok = True

    # iterations is a variable
    try:
        it = optimizer.iterations
        if it is not None and not tf.reduce_all(tf.math.is_finite(tf.cast(it, tf.float32))):
            tf.print("❌ non-finite optimizer.iterations")
            ok = False
    except Exception:
        pass

    # slot + optimizer vars (m, v, etc.)
    try:
        vars_ = optimizer.variables
    except Exception:
        vars_ = getattr(optimizer, "variables", None)
        vars_ = [v for v in vars_ if tf.as_dtype(v.dtype).is_floating]
        
    for v in vars_:
        try:
            vv = v.value if hasattr(v, "value") else v
            if not tf.reduce_all(tf.math.is_finite(tf.cast(vv, tf.float32))):
                tf.print("❌ non-finite in optimizer var:", getattr(v, "name", "<?>"))
                ok = False
        except Exception:
            # if something weird happens, fail closed
            tf.print("❌ could not validate optimizer var:", getattr(v, "name", "<?>"))
            ok = False

    return ok

def current_lr_value(optimizer: tf.keras.optimizers.Optimizer) -> tf.Tensor:
    """
    Returns a scalar tensor LR for both constant and schedule LR.
    """
    lr = optimizer.learning_rate
    if callable(lr):  # schedule
        return tf.convert_to_tensor(lr(optimizer.iterations))
    return tf.convert_to_tensor(lr)


def lr_is_finite(optimizer: tf.keras.optimizers.Optimizer) -> bool:
    lr_t = current_lr_value(optimizer)
    return bool(tf.reduce_all(tf.math.is_finite(tf.cast(lr_t, tf.float32))).numpy())

###############################################################################
# Checkpoint manager wrapper
###############################################################################

class TransformerCheckpointManager:
    """
    Thin wrapper around tf.train.CheckpointManager with:
      - finite-variable guard
      - manifest writing
    """

    def __init__(
        self,
        *,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        step_var: tf.Variable,
        checkpoint_dir: Path,
        base_model_id: Optional[str] = None,
        max_to_keep: int = 3,
    ):
        self.checkpoint_dir = checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.step_var = step_var

        self.ckpt = tf.train.Checkpoint(
            step=step_var,
            model=model,
            optimizer=optimizer,
        )

        self.manager = tf.train.CheckpointManager(
            self.ckpt,
            directory=str(checkpoint_dir),
            max_to_keep=max_to_keep,
        )

        # ---- immutable identity info ----
        self.model_config_dict = model.get_config()
        self.base_model_id = base_model_id

        self.manifest_path = checkpoint_dir / "checkpoint_manifest.json"
        
    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def latest_checkpoint(self) -> str | None:
        return self.manager.latest_checkpoint

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------
    def restore_latest(self) -> Optional[str]:
        latest = self.manager.latest_checkpoint
        if not latest:
            print(f"No checkpoint found in {self.checkpoint_dir}. Starting fresh.")
            return None
    
        self.ckpt.restore(latest).expect_partial()
        print(f"🔄 Restored checkpoint: {latest} | step={int(self.step_var.numpy())}")
        return latest
    
    # ------------------------------------------------------------------
    # Restore from path
    # ------------------------------------------------------------------
    def restore(self, path: str) -> None:
        self.ckpt.restore(path).expect_partial()
        print(f"🔄 Restored checkpoint: {path}")
    
    # ------------------------------------------------------------------
    # Save (SAFE)
    # ------------------------------------------------------------------
    def save(self) -> Optional[str]:
        """
        Save checkpoint only if model variables are finite.
        """
        model = self.ckpt.model
        optimizer = self.ckpt.optimizer
        
        if not model_all_finite(model):
            print("❌ Checkpoint NOT saved: non-finite weights detected.")
            return None
        
        if not optimizer_all_finite(optimizer):
            print("❌ Checkpoint NOT saved: non-finite optimizer state detected.")
            return None

        if not lr_is_finite(optimizer):
            print("❌ Checkpoint NOT saved: learning rate is non-finite.")
            return None

        path = self.manager.save(checkpoint_number=int(self.step_var.numpy()))
        self._write_manifest(path)

        print(f"✅ Checkpoint saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    def _write_manifest(self, checkpoint_path: str) -> None:
        manifest = {
            "base_model_id": self.base_model_id,
            "checkpoint_path": checkpoint_path,
            "step": int(self.step_var.numpy()),
            "model_config": self.model_config_dict
        }

        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        
###############################################################################
# Restore model from checkpoint
###############################################################################

def restore_model_from_checkpoint(model, checkpoint_dir):
    latest = tf.train.latest_checkpoint(str(checkpoint_dir))

    if latest is None:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir}"
        )

    tf.train.Checkpoint(model=model)\
        .restore(latest)\
        .expect_partial()

    print(f"✓ Restored model from: {latest}")

    return latest