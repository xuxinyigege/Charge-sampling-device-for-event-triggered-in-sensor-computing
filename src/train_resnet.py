"""Train the face-recognition ResNet on cropped grayscale faces."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import FaceResNet


class FaceCropDataset(Dataset):
    """ImageFolder-style dataset without an extra torchvision dependency."""

    def __init__(self, root: Path):
        self.samples: list[tuple[Path, int]] = []
        for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            label = int(class_dir.name)
            for image_path in sorted(class_dir.glob("*.png")):
                self.samples.append((image_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("L")
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor, torch.tensor(label, dtype=torch.long)


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = FaceCropDataset(args.train_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = FaceResNet(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()

        print(
            f"epoch {epoch + 1:03d}: "
            f"loss={total_loss / len(dataset):.4f}, "
            f"accuracy={correct / len(dataset):.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the face-recognition network.")
    parser.add_argument("--train-dir", type=Path, default=Path("../data/train_cropped"))
    parser.add_argument("--output", type=Path, default=Path("../models/face_resnet_state_t.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=10)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
