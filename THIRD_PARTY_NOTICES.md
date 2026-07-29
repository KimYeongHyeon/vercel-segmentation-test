# Third-party notices

## TorchVision LRASPP-MobileNetV3-Large

- Project: https://github.com/pytorch/vision
- Example checkpoint: `LRASPP_MobileNet_V3_Large_Weights.DEFAULT`
- License: BSD 3-Clause
- Exported artifact: `models/lraspp-voc21.onnx`

## ONNX Runtime Web

- Package: `onnxruntime-web@1.27.0`
- Project: https://github.com/microsoft/onnxruntime
- License: MIT
- Redistributed artifacts: `vendor/ort.wasm.min.js`,
  `vendor/ort-wasm-simd-threaded.mjs`, and
  `vendor/ort-wasm-simd-threaded.wasm`

## Medical Segmentation Decathlon Task04 Hippocampus

- Registry: https://registry.opendata.aws/msd/
- License: CC BY-SA 4.0
- Derived training artifact: `models/unet-hippocampus.onnx`
- Derived validation samples: `samples/hippocampus-mri.png` and
  `samples/hippocampus-mask.png`

These derived data/model artifacts remain under CC BY-SA 4.0 and are not
relicensed by the repository's MIT software license.

## Previous benchmark references

Model sizes for MediaPipe, MODNet, DeepLabV3-MobileNetV3, and RMBG listed in the
README come from their official project documentation or published model files.
Those models are not redistributed by this project.
