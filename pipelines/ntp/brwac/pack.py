"""
Created on Sat Dec 27 07:06:34 2025

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from pathlib import Path
from pipelines.ntp.common.runner import PackJob, pack_pipeline, read_tfrecord_sample
import random

###############################################################################
# Globals
###############################################################################

dataset_id   = "brwac"
tokenizer_id = "tokenizer_bbpe32k_v1" 

split = "train"

###############################################################################
# Files and Folders
###############################################################################

tokenized_dir = Path("data") / "ntp" / "tokenized" / dataset_id / split / tokenizer_id
packed_dir    = Path("data") / "ntp" / "packed"   / dataset_id / split / tokenizer_id

tokenizer_ckpt = Path("src") / "core" / "tokenizer" / tokenizer_id / "bbpe_tokenizer_32000.pkl"

###############################################################################
# Execute Pack Pipeline
###############################################################################

job = PackJob(
    dataset_id=dataset_id,
    tokenizer_id=tokenizer_id,
    input_dir=tokenized_dir,
    output_dir=packed_dir,
    tokenizer_checkpoint=tokenizer_ckpt,
    max_tokens_per_shard=1_024_000,
    append_eos=True,
)

pack_pipeline(job)

# quick test read
sample = random.choice(list(packed_dir.glob("*.tfrecord")))
read_tfrecord_sample(sample, tokenizer_ckpt, num_examples=2)
