from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ModelTrainer:

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = (
            optimizer
            or torch.optim.Adam(self.model.parameters(), lr=1e-3)
        )

    def train_one_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        return total_loss / len(dataloader.dataset)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(
                    self.device
                )
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        accuracy = (correct / total) * 100.0
        return total_loss / len(dataloader.dataset), accuracy
