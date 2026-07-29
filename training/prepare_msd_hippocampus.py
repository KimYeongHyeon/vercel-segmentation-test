"""Convert MSD Task04 Hippocampus volumes to leakage-safe 2D PNG pairs.

The official test set has no public labels, so this script makes a deterministic
patient-level train/validation split from the 260 labelled training volumes.
Slices from one patient never cross the split boundary.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


DATASET_URL = (
    "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar"
)
CLASS_NAMES = ["background", "anterior", "posterior"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Extracted Task04_Hippocampus directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--max-slices-per-volume",
        type=int,
        default=12,
        help="Keep the most foreground-rich slices; 0 keeps every positive slice",
    )
    return parser.parse_args()


def normalize_to_uint8(volume: np.ndarray) -> np.ndarray:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        raise ValueError("Volume has no finite voxels")
    low, high = np.percentile(finite, [0.5, 99.5])
    if high <= low:
        raise ValueError(f"Degenerate intensity range: {low}..{high}")
    clipped = np.clip(volume, low, high)
    return np.round((clipped - low) / (high - low) * 255).astype(np.uint8)


def paired_volumes(source_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for image_path in sorted((source_dir / "imagesTr").glob("*.nii.gz")):
        mask_path = source_dir / "labelsTr" / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing label volume: {mask_path}")
        pairs.append((image_path, mask_path))
    if not pairs:
        raise ValueError(f"No .nii.gz training volumes found under {source_dir}")
    return pairs


def select_slice_indices(mask: np.ndarray, maximum: int) -> list[int]:
    areas = [(index, int(np.count_nonzero(mask[:, :, index]))) for index in range(mask.shape[2])]
    positive = [(index, area) for index, area in areas if area > 0]
    if maximum > 0:
        positive = sorted(positive, key=lambda item: (-item[1], item[0]))[:maximum]
    return sorted(index for index, _ in positive)


def main() -> None:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    pairs = paired_volumes(args.source_dir)
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    val_count = max(1, round(len(pairs) * args.val_fraction))
    val_ids = {image_path.name.removesuffix(".nii.gz") for image_path, _ in pairs[:val_count]}

    manifest: dict[str, object] = {
        "dataset": "Medical Segmentation Decathlon Task04_Hippocampus",
        "source_url": DATASET_URL,
        "license": "CC-BY-SA 4.0",
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "class_names": CLASS_NAMES,
        "patients": {"train": [], "val": []},
        "slices": {"train": 0, "val": 0},
        "pixel_histogram": {"train": Counter(), "val": Counter()},
    }

    for image_path, mask_path in sorted(pairs):
        patient_id = image_path.name.removesuffix(".nii.gz")
        split = "val" if patient_id in val_ids else "train"
        image_volume = np.asarray(nib.load(image_path).dataobj, dtype=np.float32)
        mask_volume = np.asarray(nib.load(mask_path).dataobj, dtype=np.uint8)
        if image_volume.shape != mask_volume.shape:
            raise ValueError(
                f"Shape mismatch for {patient_id}: {image_volume.shape} vs {mask_volume.shape}"
            )
        labels = set(np.unique(mask_volume).tolist())
        if not labels.issubset({0, 1, 2}):
            raise ValueError(f"Unexpected labels for {patient_id}: {sorted(labels)}")

        image_volume = normalize_to_uint8(image_volume)
        indices = select_slice_indices(mask_volume, args.max_slices_per_volume)
        if not indices:
            raise ValueError(f"No foreground slices for {patient_id}")

        image_dir = args.output_dir / "images" / split
        mask_dir = args.output_dir / "masks" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        manifest["patients"][split].append(patient_id)  # type: ignore[index]
        for slice_index in indices:
            stem = f"{patient_id}_z{slice_index:02d}"
            image = Image.fromarray(image_volume[:, :, slice_index], mode="L")
            mask_array = mask_volume[:, :, slice_index]
            mask = Image.fromarray(mask_array, mode="L")
            image.save(image_dir / f"{stem}.png", optimize=True)
            mask.save(mask_dir / f"{stem}.png", optimize=True)
            manifest["slices"][split] += 1  # type: ignore[index]
            histogram = Counter(
                dict(zip(*np.unique(mask_array, return_counts=True), strict=True))
            )
            manifest["pixel_histogram"][split].update(histogram)  # type: ignore[index]

    train_ids = set(manifest["patients"]["train"])  # type: ignore[index]
    held_out_ids = set(manifest["patients"]["val"])  # type: ignore[index]
    if train_ids & held_out_ids:
        raise RuntimeError("Patient leakage detected between train and validation")
    for split in ("train", "val"):
        manifest["patients"][split] = sorted(manifest["patients"][split])  # type: ignore[index]
        manifest["pixel_histogram"][split] = {  # type: ignore[index]
            str(key): int(value)
            for key, value in sorted(manifest["pixel_histogram"][split].items())  # type: ignore[index]
        }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
