import os
import psutil
import torch


class MemoryTracker:

    @staticmethod
    def get_process_memory_mb() -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    @staticmethod
    def get_model_size_mb(model: torch.nn.Module) -> float:
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / (1024 * 1024)
