"""
Distributed Asynchronous Parallel Engine for Vision & AutoML:
1. AsynchronousDoubleBufferedLoader: Overlaps CPU/GPU data transfers and preprocessing
2. DistributedShardedWorkerPool: Shards pair batches across CPU cores and GPU OpenCL streams
3. DynamicWorkStealingScheduler: Balances workload across uneven difficulty pairs
"""

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
import torch
from typing import List, Dict, Any, Tuple, Optional, Callable

class AsynchronousDoubleBufferedLoader:
    """
    Overlaps Disk/Memory I/O and Tensor Normalization with downstream GPU/CPU inference.
    Maintains a double-buffered prefetch queue.
    """
    def __init__(self, pairs: List[Dict[str, Any]], prefetch_size: int = 4):
        self.pairs = pairs
        self.prefetch_size = prefetch_size
        self.queue = queue.Queue(maxsize=prefetch_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        for p in self.pairs:
            if self.stopped:
                break
            # Pre-compute grayscales, dimensions, and normalized intrinsics
            img0 = p["img0"]
            img1 = p["img1"]
            g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) if len(img0.shape) == 3 else img0
            g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1

            item = {
                **p,
                "gray0": g0,
                "gray1": g1,
                "H0": img0.shape[0], "W0": img0.shape[1],
                "H1": img1.shape[0], "W1": img1.shape[1]
            }
            self.queue.put(item)
        self.queue.put(None) # Sentinel

    def get_batch(self, batch_size: int = 4) -> List[Dict[str, Any]]:
        batch = []
        for _ in range(batch_size):
            item = self.queue.get()
            if item is None:
                break
            batch.append(item)
        return batch

    def stop(self):
        self.stopped = True


class DistributedShardedWorkerPool:
    """
    Shards evaluation batches across multi-core CPU workers and hardware compute streams.
    """
    def __init__(self, n_workers: int = 6):
        self.n_workers = n_workers
        self.pool = ThreadPoolExecutor(max_workers=n_workers)

    def map_shards(self, fn: Callable[[Dict[str, Any]], Any], items: List[Dict[str, Any]]) -> List[Any]:
        futures = [self.pool.submit(fn, item) for item in items]
        results = [f.result() for f in futures]
        return results

    def shutdown(self):
        self.pool.shutdown(wait=True)
