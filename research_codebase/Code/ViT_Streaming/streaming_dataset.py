"""
streaming_dataset.py
--------------------
Queue-free streaming DataLoader.
Drives xrd_generator.worker() directly via multiprocessing.Pool.

Worker task tuple : (task_id, eg, sg_main, sg_imp, seed)
Worker return     : (task_id, sg_out, eg, intensity) | (task_id, None)

Labels = extinction_group - 1  (0-indexed, 0-98)
Sampling controlled by EG_SAMPLING_MODE env var ('rruff' or 'uniform').
"""

from __future__ import annotations

import os
import torch
from torch.utils.data import IterableDataset, DataLoader
import multiprocessing as mp

from xrd_generator import (
    worker,
    build_task,
    NUM_POINTS,
)

NUM_EXTINCTION_GROUPS = 99


# -----------------------------------------------------------------------
# Task stream
# -----------------------------------------------------------------------

def _task_stream(n_tasks: int, seed: int):
    """Yields build_task() tuples up to n_tasks."""
    for task_id in range(n_tasks):
        yield build_task(task_id, seed)


# -----------------------------------------------------------------------
# IterableDataset
# -----------------------------------------------------------------------

class StreamingXRDDataset(IterableDataset):
    """
    Streams freshly generated XRD patterns into PyTorch.
    Each __iter__ call (epoch) creates a new Pool + new random seed
    so patterns are never repeated across epochs.

    Args:
        num_samples:    Total patterns to yield per epoch.
        n_workers:      Pool size. Default: max(1, cpu_count - 4).
        pool_chunksize: imap_unordered chunksize (default 16).
    """

    def __init__(
        self,
        num_samples:    int,
        n_workers:      int | None = None,
        pool_chunksize: int        = 16,
    ):
        self.num_samples    = num_samples
        self.n_workers      = n_workers or max(1, (os.cpu_count() or 8) - 4)
        self.pool_chunksize = pool_chunksize

    def __iter__(self):
        seed       = int.from_bytes(os.urandom(8), "little")
        delivered  = 0
        errors     = 0
        # Budget extra tasks to absorb worker failures
        max_tasks  = self.num_samples * 20

        with mp.Pool(processes=self.n_workers) as pool:
            for result in pool.imap_unordered(
                worker,
                _task_stream(max_tasks, seed),
                chunksize=self.pool_chunksize,
            ):
                task_id, *rest = result

                # Worker failure: rest == [None]
                if rest[0] is None:
                    errors += 1
                    continue

                sg_out, eg, intensity = rest

                if intensity.shape[0] != NUM_POINTS:
                    errors += 1
                    continue

                yield (
                    torch.from_numpy(intensity),
                    torch.tensor(eg - 1, dtype=torch.long),  # 0-indexed label
                )
                delivered += 1

                if delivered >= self.num_samples:
                    break

        if errors > 0:
            print(f"[StreamingXRDDataset] epoch done — "
                  f"delivered={delivered:,}  errors={errors:,}")


# -----------------------------------------------------------------------
# Public loader factory
# -----------------------------------------------------------------------

def get_streaming_dataloader(
    num_samples:    int,
    batch_size:     int        = 256,
    n_workers:      int | None = None,
    pool_chunksize: int        = 16,
    num_classes:    int        = NUM_EXTINCTION_GROUPS,
) -> tuple:
    """
    Returns (loader, num_classes, num_points).
    Drop-in for get_dataloaders() in streaming mode.
    """
    dataset = StreamingXRDDataset(
        num_samples    = num_samples,
        n_workers      = n_workers,
        pool_chunksize = pool_chunksize,
    )
    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        num_workers = 0,       # Pool lives inside __iter__ — must be 0
        pin_memory  = True,    # async DDR5 → H100 DMA
    )
    return loader, num_classes, NUM_POINTS
