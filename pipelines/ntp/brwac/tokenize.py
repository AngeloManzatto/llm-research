"""
Created on Fri Dec 26 21:37:32 2025

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Files and Folders
###############################################################################

from pathlib import Path
from pipelines.ntp.common.runner import TokenizeJob, tokenization_pipeline

###############################################################################
# Globals
###############################################################################

dataset_id = "brwac"
tokenizer_id = "tokenizer_bbpe32k_v1"

shard_size = 10_000
workers = 6 
split = "train"

###############################################################################
# Files and Folders
###############################################################################

input_dir  = Path("data") / "ntp" / "extracted" /  dataset_id / split
output_dir = Path("data") / "ntp" / "tokenized" /  dataset_id / split / tokenizer_id
tokenizer_checkpoint = Path("src") / "core" / "tokenizer" / tokenizer_id / "bbpe_tokenizer_32000.pkl"
                           
###############################################################################
# Execute Tokenization Pipeline
###############################################################################

job = TokenizeJob(
    dataset_id=dataset_id,
    tokenizer_id=tokenizer_id,
    input_dir=input_dir,
    output_dir=output_dir,
    tokenizer_checkpoint=tokenizer_checkpoint,
    shard_size=shard_size,
    workers=workers
)

tokenization_pipeline(job)

