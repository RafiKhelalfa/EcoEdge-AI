import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class StructuredPruner:

    def __init__(self, model: nn.Module):
        self.model = model

    def apply_structured_pruning(
        self, amount: float = 0.2, n_norm: int = 1, dim: int = 0
    ) -> nn.Module:
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(
                    module, name="weight", amount=amount, n=n_norm, dim=dim
                )
                prune.remove(module, "weight")
        return self.model
