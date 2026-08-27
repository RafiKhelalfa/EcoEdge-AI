import time
from typing import Callable, List, Tuple
import numpy as np
import torch


class LatencyMeter:

    @staticmethod
    def measure_inference_time(
        model_fn: Callable,
        sample_input: torch.Tensor,
        warmup: int = 10,
        runs: int = 100,
    ) -> Tuple[float, float]:
        for _ in range(warmup):
            _ = model_fn(sample_input)

        latencies: List[float] = []

        for _ in range(runs):
            start = time.perf_counter_ns()
            _ = model_fn(sample_input)
            end = time.perf_counter_ns()
            latencies.append((end - start) / 1e6)

        mean_latency = float(np.mean(latencies))
        std_latency = float(np.std(latencies))
        return mean_latency, std_latency
