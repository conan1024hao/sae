"""Raise the NCCL collective timeout that DeepSpeed's secondary process groups inherit.

The gradient all-reduce died at step ~12951 with:

    [Rank 2] Watchdog caught collective operation timeout:
    WorkNCCL(SeqNum=55747, OpType=ALLREDUCE, NumelIn=100664832, Timeout(ms)=600000)
    ran for 600095 milliseconds before timing out

`NumelIn=100664832` is the SAE gradient, so one rank simply failed to reach the
all-reduce within ten minutes -- it was still in huggingface_hub's retry backoff
fetching a shard. Once the httpx patch let those retries actually happen instead of
crashing, a slow shard fetch became a stall the other seven ranks wait on.

Neither `--ddp_timeout` nor `DEEPSPEED_TIMEOUT` fixes this. Both configure the default
process group, but the failure was on `PG ID 1`: DeepSpeed builds extra groups via
`dist.new_group(ranks)` with no timeout argument, so torch falls back to
`default_pg_nccl_timeout`, which is exactly the ten minutes observed. There is no
environment variable for it, so the module global has to be replaced before any group
is created.
"""

import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

DEFAULT_MINUTES = 60


def apply(minutes: int = DEFAULT_MINUTES) -> bool:
    """Raise the default NCCL timeout. Must run before any process group is created."""
    from torch.distributed import distributed_c10d as c10d

    if not hasattr(c10d, "default_pg_nccl_timeout"):
        logger.warning("torch.distributed layout changed; skipping NCCL timeout patch")
        return False

    c10d.default_pg_nccl_timeout = timedelta(minutes=minutes)
    return True
