# MSD Hippocampus U-Net training record

## Claim boundary

이 기록은 교육용 2D slice U-Net의 **5-epoch pilot**입니다. 환자 단위 held-out
validation에서 세 클래스가 모두 예측되고 ONNX 브라우저 경로가 실행되는지를
검증합니다. 3D 임상 성능, 공식 MSD test 성능 또는 임상 사용 가능성을 주장하지
않습니다.

## Dataset and split

- Source: Medical Segmentation Decathlon Task04 Hippocampus
- License: CC-BY-SA 4.0
- Labelled volumes: 260
- Split seed: `20260729`
- Patient split: 208 train / 52 validation
- 2D slices: 2,496 train / 624 validation
- Slice policy: 각 volume에서 foreground 면적이 큰 최대 12개 slice
- Train/validation duplicate PNG hash: 0
- Classes: `0 background`, `1 anterior`, `2 posterior`
- Training pixel histogram: 3,869,498 / 317,419 / 205,875
- Validation pixel histogram: 987,939 / 80,498 / 55,123

## Resolved training configuration

```text
architecture     TinyUNet(base_channels=24)
input            float32 [B,3,256,256]
output           float32 [B,3,256,256]
loss             0.5 CrossEntropy + 0.5 multiclass Dice
optimizer        AdamW
learning_rate    0.0003
batch_size       16
epochs           5
seed             20260729
device           Apple MPS
num_workers      0
```

`num_workers=2`로 시작한 최초 시도는 epoch 로그 없이 정지해 중단했고 artifact를
사용하지 않았습니다. macOS/MPS에서 안전한 `num_workers=0`으로 재실행했습니다.

## Validation results

| Epoch | Train loss | mIoU | Background IoU | Anterior IoU | Posterior IoU |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.736074 | 0.445276 | 0.930506 | 0.404845 | 0.000479 |
| 2 | 0.477908 | 0.646534 | 0.938756 | 0.534424 | 0.466421 |
| 3 | 0.327913 | 0.723510 | 0.957753 | 0.630203 | 0.582574 |
| 4 | 0.242386 | 0.730339 | 0.960839 | 0.622011 | 0.608169 |
| 5 | 0.192528 | **0.735724** | **0.961866** | **0.632059** | **0.613247** |

## Exported artifact

- File: `models/unet-hippocampus.onnx`
- Size: 4.34 MB
- Input: `image float32[1,3,256,256]`
- Output: `logits float32[1,3,256,256]`
- ONNX opset: 17

## Sources

- MedMNIST classification contract: https://github.com/MedMNIST/MedMNIST
- MSD registry and license: https://registry.opendata.aws/msd/
- MSD Task04 download:
  https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar
