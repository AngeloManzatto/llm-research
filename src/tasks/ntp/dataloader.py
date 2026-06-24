"""
Created on Tue Dec 23 16:24:23 2025

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

import glob
import random

from dataclasses import dataclass
from pathlib import Path
from typing import List,  Sequence, Union


import tensorflow as tf

###############################################################################
# Config
###############################################################################
@dataclass(frozen=True)
class NTPDatasetConfig:
    seq_len: int
    global_batch_size: int
    vocab_size: int

    shuffle: bool = True
    shuffle_buffer: int = 10_000
    repeat: bool = True

    # TFRecord reading
    num_parallel_reads: int = tf.data.AUTOTUNE
    num_parallel_calls: int = tf.data.AUTOTUNE

    # Determinism
    deterministic: bool = False

    # Safety: strict token range checking (can be expensive)
    assert_in_vocab: bool = True

###############################################################################
# TFRecord sources resolution (patterns or explicit paths)
###############################################################################
def resolve_tfrecord_files(
    sources: Union[str, Sequence[str], Sequence[Path]],
    *,
    recursive: bool = True,
    shuffle_files: bool = False,
    seed: int = 42,
) -> List[str]:
    """
    Resolve TFRecord sources into a list of file paths.

    `sources` can be:
      - a single glob pattern string: "data/**/packed_*.tfrecord"
      - a list of glob patterns: ["data/a/**/*.tfrecord", "data/b/**/*.tfrecord"]
      - a list of Path objects (already resolved)

    Returns a sorted (or shuffled) list[str].
    """
    # Case: already Paths
    if isinstance(sources, (list, tuple)) and sources and isinstance(sources[0], Path):
        files = [str(p) for p in sources]  # type: ignore
    else:
        patterns = [sources] if isinstance(sources, str) else [str(s) for s in sources]  # type: ignore
        files: List[str] = []
        for pat in patterns:
            files.extend(glob.glob(pat, recursive=recursive))

    # unique + stable order
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No .tfrecord files found for sources={sources}")

    if shuffle_files:
        rng = random.Random(seed)
        rng.shuffle(files)

    return files

###############################################################################
# TFRecord parsing
###############################################################################
def parse_tfrecord_tokens(example_proto: tf.Tensor) -> tf.Tensor:
    """
    Parse a TFRecord Example containing:
      - tokens: VarLenFeature(int64)

    Returns:
      tokens: dense int32 tensor of shape [N]
    """
    feature_description = {"tokens": tf.io.VarLenFeature(tf.int64)}
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    tokens = tf.sparse.to_dense(parsed["tokens"])
    tokens = tf.cast(tokens, tf.int32)
    return tokens


def _assert_in_vocab(tokens: tf.Tensor, vocab_size: int) -> tf.Tensor:
    """
    Ensures 0 <= tokens < vocab_size. Returns tokens unchanged.
    Use for debugging / corruption detection.
    """
    tokens = tf.convert_to_tensor(tokens)
    vs = tf.cast(vocab_size, tokens.dtype)
    with tf.control_dependencies(
        [
            tf.debugging.assert_greater_equal(tokens, 0, message="tokens < 0"),
            tf.debugging.assert_less(tokens, vs, message="tokens >= vocab_size"),
        ]
    ):
        return tf.identity(tokens)


def split_into_windows(tokens: tf.Tensor, *, seq_len: int) -> tf.data.Dataset:
    """
    Convert a 1D token stream into windows of length (seq_len + 1),
    stepped by seq_len (non-overlapping chunks).

    Output dataset elements have shape [seq_len + 1].
    """
    n = tf.shape(tokens)[0]

    def no_data() -> tf.data.Dataset:
        empty = tf.zeros([0, seq_len + 1], dtype=tf.int32)
        return tf.data.Dataset.from_tensor_slices(empty)

    def valid_data() -> tf.data.Dataset:
        windows = tf.signal.frame(
            tokens,
            frame_length=seq_len + 1,
            frame_step=seq_len,
            axis=0,
        )
        return tf.data.Dataset.from_tensor_slices(windows)

    return tf.cond(n < (seq_len + 1), no_data, valid_data)


def to_xy(window: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    window: [seq_len + 1]
    returns:
      x: [seq_len]  (window[:-1])
      y: [seq_len]  (window[1:])
    """
    return window[:-1], window[1:]

###############################################################################
# Public API
###############################################################################
def build_dataset(
    tfrecord_files: Sequence[str],
    cfg: NTPDatasetConfig,
) -> tf.data.Dataset:
    """
    Build NTP dataset from TFRecord shards.

    Each record contains a long token stream; we frame it into [seq_len+1] windows,
    then map into (x,y) next-token pairs.
    """
    raw = tf.data.TFRecordDataset(
        tfrecord_files,
        num_parallel_reads=cfg.num_parallel_reads,
    )

    ds = raw.map(parse_tfrecord_tokens, num_parallel_calls=cfg.num_parallel_calls)
    ds = ds.flat_map(lambda tok: split_into_windows(tok, seq_len=cfg.seq_len))

    if cfg.shuffle:
        ds = ds.shuffle(cfg.shuffle_buffer)

    if cfg.assert_in_vocab:
        ds = ds.map(
            lambda window: _assert_in_vocab(window, cfg.vocab_size),
            num_parallel_calls=cfg.num_parallel_calls,
        )

    ds = ds.map(to_xy, num_parallel_calls=cfg.num_parallel_calls)

    if cfg.repeat:
        ds = ds.repeat()

    ds = ds.batch(cfg.global_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    opts = tf.data.Options()
    opts.experimental_deterministic = bool(cfg.deterministic)
    ds = ds.with_options(opts)

    return ds

def build_dataset_from_sources(
    sources: Union[str, Sequence[str], Sequence[Path]],
    cfg: NTPDatasetConfig,
    *,
    recursive: bool = True,
    shuffle_files: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Convenience wrapper: resolve TFRecords from patterns/paths, then build dataset.

    Example (mix datasets):
      sources = [
        "data/ntp/packed/cc100/**/packed_*.tfrecord",
        "data/ntp/packed/brwac/**/packed_*.tfrecord",
      ]
    """
    files = resolve_tfrecord_files(
        sources,
        recursive=recursive,
        shuffle_files=shuffle_files,
        seed=seed,
    )
    return build_dataset(files, cfg)
