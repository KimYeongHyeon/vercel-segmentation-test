"""Train LRASPP or a compact U-Net on paired image/mask folders."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from models import SegmentationOutput, build_model


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PairedMaskDataset(Dataset):
    """Pair files by stem: images/<split>/x.jpg with masks/<split>/x.png."""

    def __init__(self, root: Path, split: str, image_size: int, augment: bool) -> None:
        self.image_size = image_size
        self.augment = augment
        image_dir = root / "images" / split
        mask_dir = root / "masks" / split
        image_paths = sorted(
            path
            for path in image_dir.glob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
        )
        self.samples = []
        for image_path in image_paths:
            mask_path = mask_dir / f"{image_path.stem}.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
            self.samples.append((image_path, mask_path))
        if not self.samples:
            raise ValueError(f"No paired samples found for split '{split}' under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )
        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
        return image_tensor, mask_tensor


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    valid = targets != 255
    encoded = targets[valid] * num_classes + predictions[valid]
    return torch.bincount(encoded, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )


def multiclass_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    valid = targets != 255
    safe_targets = targets.masked_fill(~valid, 0)
    one_hot = torch.nn.functional.one_hot(
        safe_targets, num_classes=num_classes
    ).permute(0, 3, 1, 2)
    one_hot = one_hot.to(dtype=probabilities.dtype)
    valid_mask = valid.unsqueeze(1)
    probabilities = probabilities * valid_mask
    one_hot = one_hot * valid_mask
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = (2 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> tuple[float, list[float]]:
    model.eval()
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        predictions = model(images).argmax(1)
        matrix += confusion_matrix(predictions, masks, num_classes)
    intersection = matrix.diag().float()
    union = matrix.sum(0) + matrix.sum(1) - intersection
    class_iou = torch.where(union > 0, intersection / union, torch.nan)
    return torch.nanmean(class_iou).item(), class_iou.cpu().tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=["lraspp", "unet"], default="lraspp")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--class-names", required=True, help="Comma-separated names")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; 0 is the safest default on macOS/MPS",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=0.0,
        help="Mix Dice with cross entropy: loss=(1-w)*CE+w*Dice",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, default=Path("checkpoint.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    if len(class_names) != args.num_classes:
        raise ValueError("--class-names count must equal --num-classes")
    if not 0 <= args.dice_weight <= 1:
        raise ValueError("--dice-weight must be between 0 and 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    train_data = PairedMaskDataset(args.data_dir, "train", args.image_size, augment=True)
    val_data = PairedMaskDataset(args.data_dir, "val", args.image_size, augment=False)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = SegmentationOutput(
        build_model(args.architecture, args.num_classes)
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_miou = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            ce_loss = criterion(logits, masks)
            dice_loss = multiclass_dice_loss(logits, masks, args.num_classes)
            loss = (1 - args.dice_weight) * ce_loss + args.dice_weight * dice_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.shape[0]

        miou, class_iou = evaluate(model, val_loader, device, args.num_classes)
        train_loss = running_loss / len(train_data)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "val_miou": round(miou, 6),
                    "class_iou": class_iou,
                },
                ensure_ascii=False,
            )
        )
        if miou > best_miou:
            best_miou = miou
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.module.state_dict()
                    if hasattr(model, "module")
                    else model.state_dict(),
                    "architecture": args.architecture,
                    "num_classes": args.num_classes,
                    "class_names": class_names,
                    "image_size": args.image_size,
                    "val_miou": miou,
                    "dice_weight": args.dice_weight,
                    "seed": args.seed,
                },
                args.output,
            )


if __name__ == "__main__":
    main()
