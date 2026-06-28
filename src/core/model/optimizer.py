"""
Created on Mon Dec 29 07:36:44 2025

@author: Angelo Antonio Manzatto
"""
###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Literal

import tensorflow as tf

from src.core.model.schedules import WarmupCosine 

###############################################################################
# Optimizer Config
###############################################################################

OptimizerName = Literal["adam", "adamw", "sgd", "rmsprop"]

@dataclass(frozen=True)
class OptimizerConfig:
    # ---- Which optimizer ----
    name: OptimizerName = "adamw"

    # ---- LR schedule (WarmupCosine) ----
    base_lr: float = 4.2e-4
    min_lr: float = 1.0e-5
    warmup_steps: int = 1_000
    total_steps: int = 100_000

    # ---- Common-ish optimizer params ----
    epsilon: float = 1e-5  # Adam/AdamW, also acceptable for RMSprop

    # ---- Adam / AdamW ----
    beta_1: float = 0.9
    beta_2: float = 0.95
    weight_decay: float = 0.10  # AdamW only

    # ---- SGD ----
    momentum: float = 0.9
    nesterov: bool = False

    # ---- RMSProp ----
    rho: float = 0.9
    rmsprop_momentum: float = 0.0
    centered: bool = False


###############################################################################
# Helpers (same ideas as your current file)
###############################################################################

def default_no_decay_vars(model: tf.keras.Model) -> List[tf.Variable]:
    """
    Safer exclusion rules:
      - rank == 1 => bias or scale vectors
      - embeddings
      - normalization layers (rmsnorm/layernorm)
    """
    no_decay: List[tf.Variable] = []
    for v in model.trainable_variables:
        name = (getattr(v, "path", "") or v.name).lower()
        if len(v.shape) == 1:
            no_decay.append(v)
            continue
        if "embedding" in name:
            no_decay.append(v)
            continue
        if ("rmsnorm" in name) or ("rms_norm" in name) or ("layernorm" in name) or ("layer_norm" in name):
            no_decay.append(v)
            continue
    return no_decay


def build_lr_schedule(cfg: OptimizerConfig) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    # Same WarmupCosine usage as before, just reading from unified config.
    return WarmupCosine(
        base_lr=cfg.base_lr,
        min_lr=cfg.min_lr,
        warmup_steps=cfg.warmup_steps,
        total_steps=cfg.total_steps,
    )


def _maybe_build_slots(opt: tf.keras.optimizers.Optimizer, model: tf.keras.Model) -> None:
    # Ensure optimizer slots exist (important for restore + fair memory tests)
    if hasattr(opt, "build"):
        opt.build(model.trainable_variables)


def _maybe_exclude_from_wd(
    opt: tf.keras.optimizers.Optimizer,
    model: tf.keras.Model,
    *,
    no_decay_vars: Optional[Iterable[tf.Variable]] = None,
) -> None:
    # AdamW-only behavior (best-effort for TF compatibility)
    var_list = list(no_decay_vars) if no_decay_vars is not None else default_no_decay_vars(model)
    if hasattr(opt, "exclude_from_weight_decay"):
        opt.exclude_from_weight_decay(var_list=var_list)


###############################################################################
# Builder (single cfg)
###############################################################################

def build_optimizer(
    model: tf.keras.Model,
    cfg: OptimizerConfig,
    *,
    no_decay_vars: Optional[Iterable[tf.Variable]] = None,
) -> tf.keras.optimizers.Optimizer:
    """
    Build an optimizer (adam/adamw/sgd/rmsprop) using a single unified config.

    Notes:
      - Uses WarmupCosine LR schedule.
      - Weight decay exclusions are applied only for AdamW (if supported).
      - Slots are built immediately via opt.build(model.trainable_variables) for restore consistency.
    """
    lr_schedule = build_lr_schedule(cfg)

    name = cfg.name.lower()

    if name == "adam":
        opt = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=cfg.beta_1,
            beta_2=cfg.beta_2,
            epsilon=cfg.epsilon,
        )

    elif name == "adamw":
        opt = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=cfg.weight_decay,
            beta_1=cfg.beta_1,
            beta_2=cfg.beta_2,
            epsilon=cfg.epsilon,
        )
        _maybe_exclude_from_wd(opt, model, no_decay_vars=no_decay_vars)

    elif name == "sgd":
        opt = tf.keras.optimizers.SGD(
            learning_rate=lr_schedule,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
        )

    elif name == "rmsprop":
        opt = tf.keras.optimizers.RMSprop(
            learning_rate=lr_schedule,
            rho=cfg.rho,
            momentum=cfg.rmsprop_momentum,  # avoid collision with SGD momentum
            epsilon=cfg.epsilon,
            centered=cfg.centered,
        )

    else:
        raise ValueError(f"Unknown optimizer name: {cfg.name!r}")

    _maybe_build_slots(opt, model)
    return opt


