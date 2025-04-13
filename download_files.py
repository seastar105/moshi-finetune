import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from moshi.models import loaders
from datasets import load_dataset

checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
    hf_repo="seastar105/moshi-eeve-extended",
    moshi_weights=None,
    mimi_weights=None,
    tokenizer=None,
    config_path=None,
)
load_dataset("seastar105/k-moshi-ft-2ch")
