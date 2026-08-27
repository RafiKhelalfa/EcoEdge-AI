import unittest
import torch
import torchvision
from ecoedge.compression.pruner import StructuredPruner
from ecoedge.compression.quantizer import ModelQuantizer


class TestCompression(unittest.TestCase):

    def setUp(self):
        self.model = torchvision.models.resnet18(num_classes=10)

    def test_pruning(self):
        pruner = StructuredPruner(self.model)
        pruned_model = pruner.apply_structured_pruning(amount=0.2)
        self.assertIsNotNone(pruned_model)

    def test_quantization(self):
        quantizer = ModelQuantizer(self.model)
        quant_model = quantizer.quantize_dynamic_int8()
        self.assertIsNotNone(quant_model)


if __name__ == "__main__":
    unittest.main()

