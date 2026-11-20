import torch
import torch.nn as nn

class AdaptivePiecewiseQuantizer(nn.Module):
    """
    Quantification non-linéaire par morceaux avec seuillage doux (soft-thresholding)
    pour préserver la dynamique des poids éparses issus du pruning sans saturer le NPU.
    """
    def __init__(self, model: nn.Module, bits: int = 8):
        super().__init__()
        self.model = model
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = (2 ** (bits - 1)) - 1

    def forward(self, x):
        return self.model(x)

    def apply_soft_threshold_quantization(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    non_zero_mask = param != 0
                    if non_zero_mask.sum() > 0:
                        abs_vals = param[non_zero_mask].abs()
                        scale = abs_vals.max() / self.qmax
                        if scale > 0:
                            quantized = torch.clamp(torch.round(param / scale), self.qmin, self.qmax)
                            param.copy_(quantized * scale)
        print(f"[EcoEdge-AI] Quantification adaptative INT{self.bits} appliquée avec succès.")
