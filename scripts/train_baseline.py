import torch
import torchvision
import torchvision.transforms as transforms
from ecoedge.core.trainer import ModelTrainer
from ecoedge.core.utils import set_seed, get_device


def main():
    set_seed(42)
    device = get_device()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=False
    )

    model = torchvision.models.resnet18(num_classes=10)
    trainer = ModelTrainer(model=model, device=device)

    print("Training Baseline Model...")
    for epoch in range(1, 6):
        loss = trainer.train_one_epoch(train_loader)
        val_loss, acc = trainer.evaluate(test_loader)
        print(
            f"Epoch {epoch} | Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.2f}%"
        )

    torch.save(model.state_dict(), "baseline_resnet18.pth")
    print("Baseline model saved to baseline_resnet18.pth")


if __name__ == "__main__":
    main()
