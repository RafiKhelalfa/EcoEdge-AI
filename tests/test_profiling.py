import unittest
import torch
import torchvision
from ecoedge.profiling.latency_meter import LatencyMeter
from ecoedge.profiling.memory_tracker import MemoryTracker


class TestProfiling(unittest.TestCase):

    def setUp(self):
        self.model = torchvision.models.resnet18(num_classes=10)
        self.model.eval()  # Évite les erreurs BatchNorm sur batch_size=1
        self.input_tensor = torch.randn(1, 3, 32, 32)

    def test_latency_meter(self):
        mean_lat, std_lat = LatencyMeter.measure_inference_time(
            self.model, self.input_tensor, warmup=2, runs=5
        )
        self.assertGreater(mean_lat, 0.0)
        self.assertGreaterEqual(std_lat, 0.0)

    def test_memory_tracker(self):
        size_mb = MemoryTracker.get_model_size_mb(self.model)
        self.assertGreater(size_mb, 0.0)


if __name__ == "__main__":
    unittest.main()
