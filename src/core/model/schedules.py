"""
Created on Wed Dec 24 07:54:50 2025

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import math
import tensorflow as tf

'''
###############################################################################
# Learning Rate Scheduler
###############################################################################
'''   

@tf.keras.utils.register_keras_serializable(package="Custom")
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warmup to `base_lr`, then cosine decay to `min_lr` over `total_steps`.
    No restarts. Simple and stable.

    Args:
      base_lr: peak LR reached right after warmup.
      min_lr: floor LR at the end of the schedule.
      warmup_steps: number of warmup steps (can be 0).
      total_steps: total number of steps for the full schedule (>= warmup_steps).
    """
    def __init__(self, base_lr, min_lr, warmup_steps, total_steps, name=None):
        super().__init__()
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        if warmup_steps < 0 or warmup_steps > total_steps:
            raise ValueError("0 <= warmup_steps <= total_steps")
        if base_lr <= 0 or min_lr < 0:
            raise ValueError("LRs must be non-negative; base_lr > 0")
        if min_lr > base_lr:
            raise ValueError("min_lr must be <= base_lr")

        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.name = name or "WarmupCosine"

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # --- Warmup ---
        wu = tf.cast(tf.maximum(self.warmup_steps, 1), tf.float32)  # avoid /0
        warmup_frac = tf.clip_by_value(step / wu, 0.0, 1.0)
        lr_warm = self.base_lr * warmup_frac

        # --- Cosine decay ---
        tail_len = tf.cast(tf.maximum(self.total_steps - self.warmup_steps, 1), tf.float32)
        tail_step = tf.clip_by_value(step - tf.cast(self.warmup_steps, tf.float32), 0.0, tail_len)
        progress = tail_step / tail_len  # 0..1
        cosine = 0.5 * (1.0 + tf.cos(tf.constant(math.pi, tf.float32) * progress))
        lr_cos = (self.base_lr - self.min_lr) * cosine + self.min_lr

        # Pick warmup or cosine branch
        lr = tf.where(step < self.warmup_steps, lr_warm, lr_cos)

        # Extra safety: clamp to [min_lr, base_lr]
        lr = tf.clip_by_value(lr, self.min_lr, self.base_lr)
        return tf.cast(lr, tf.float32)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "name": self.name,
        }