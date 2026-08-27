import torch
import torch.nn as nn


class ModelQuantizer:

    def __init__(self, model: nn.Module):
        self.model = model

    def quantize_dynamic_int8(self) -> nn.Module:
        quantized_model = torch.ao.quantization.quantize_dynamic(
            self.model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
        return quantized_model

    def prepare_static_quantization(
        self, backend: str = "fbgemm"
    ) -> nn.Module:
        torch.backends.quantized.engine = backend
        self.model.eval()
        self.model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
        prepared_model = torch.ao.quantization.prepare(self.model)
        return prepared_model

    @staticmethod
    def convert_static_quantization(prepared_model: nn.Module) -> nn.Module:
        return torch.ao.quantization.convert(prepared_model)
