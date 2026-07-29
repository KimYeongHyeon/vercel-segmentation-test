"""Export a custom checkpoint or official VOC baseline to browser-ready ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch

from models import SegmentationOutput, build_model


VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--pretrained-voc", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pretrained_voc:
        architecture = "lraspp"
        num_classes = 21
        class_names = VOC_CLASSES
        model = build_model(architecture, num_classes, pretrained_voc=True)
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        architecture = checkpoint["architecture"]
        num_classes = checkpoint["num_classes"]
        class_names = checkpoint["class_names"]
        model = SegmentationOutput(build_model(architecture, num_classes))
        model.load_state_dict(checkpoint["model"])

    if not isinstance(model, SegmentationOutput):
        model = SegmentationOutput(model)
    model.eval()
    dummy = torch.zeros(1, 3, args.image_size, args.image_size, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        args.output,
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    exported = onnx.load(args.output)
    onnx.checker.check_model(exported)
    metadata = {
        "architecture": architecture,
        "num_classes": num_classes,
        "class_names": class_names,
        "image_size": args.image_size,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {args.output} ({args.output.stat().st_size / 1_000_000:.2f} MB)")


if __name__ == "__main__":
    main()
