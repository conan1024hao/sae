"""Checkpoint the streaming dataset's position alongside the model checkpoint.

Without this, resuming a streaming run makes the Trainer replay the stream from the
top and discard everything before the checkpoint step (`skip_first_batches`). On
FineVisionMax (24M rows / 10k shards) that measured at ~9 minutes per shard, i.e.
more than half a day to reach step 6000 -- longer than the training it was resuming.

Saving `datasets.IterableDataset.state_dict()` (shard index + row offset) lets a
resume seek straight to the right position, so the skip costs nothing and no sample
is replayed.
"""

import json
import os

import torch.distributed as dist
from transformers import TrainerCallback

STATE_FILENAME = "dataset_state_rank{rank}.json"


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def state_path(checkpoint_dir: str, rank: int | None = None) -> str:
    return os.path.join(checkpoint_dir, STATE_FILENAME.format(rank=_rank() if rank is None else rank))


class DatasetStateCallback(TrainerCallback):
    """Writes the stream position into each checkpoint directory as it is saved.

    Every rank writes its own file: with `dispatch_batches: false` each rank drives an
    independent iterator, so their positions are not interchangeable.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(state_path(checkpoint_dir), "w") as f:
            json.dump(self.dataset.state_dict(), f)


def load_into(dataset, checkpoint_dir: str) -> bool:
    """Seek `dataset` to the position recorded in `checkpoint_dir`.

    Returns True if a position was restored. False means the checkpoint predates this
    mechanism, and the caller has to decide between replaying the stream and accepting
    that the samples before the checkpoint are seen a second time.
    """
    path = state_path(checkpoint_dir)
    if not os.path.exists(path):
        return False
    with open(path) as f:
        dataset.load_state_dict(json.load(f))
    return True
