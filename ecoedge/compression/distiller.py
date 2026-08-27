import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class KnowledgeDistiller:

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
    ):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kld_loss = nn.KLDivLoss(reduction="batchmean")

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> float:
        self.teacher.eval()
        self.student.train()

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits = self.teacher(inputs)

        student_logits = self.student(inputs)

        loss_ce = self.ce_loss(student_logits, targets)

        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        loss_kld = self.kld_loss(soft_student, soft_teacher) * (
            self.temperature**2
        )

        total_loss = self.alpha * loss_kld + (1.0 - self.alpha) * loss_ce
        total_loss.backward()
        optimizer.step()

        return total_loss.item()
