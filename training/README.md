# Segmentation training kit

두 과제를 같은 브라우저 입력/출력 계약으로 학습합니다.

- 다중 클래스 일반영상: `--architecture lraspp`
- 의료영상 멀티클래스: `--architecture unet`

> MedMNIST는 이미지 단위 **분류** 데이터라 픽셀 마스크가 없습니다. U-Net
> segmentation 실습에는 MedMNIST를 억지로 변환하지 않고, 공식 마스크가 있는
> Medical Segmentation Decathlon(MSD) Task04를 사용합니다.

## 데이터 구조

```text
dataset/
├── images/
│   ├── train/sample-001.jpg
│   └── val/sample-101.jpg
└── masks/
    ├── train/sample-001.png
    └── val/sample-101.png
```

마스크 PNG의 픽셀값은 클래스 ID `0..C-1`이고, 평가에서 제외할 픽셀은
`255`입니다. 회색조 의료영상도 로더에서 RGB 3채널로 복제됩니다.

## 학습

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 일반영상 4-class 예시
python train.py \
  --data-dir ./dataset \
  --architecture lraspp \
  --num-classes 4 \
  --class-names "background,road,person,vehicle" \
  --output checkpoints/semantic.pt

# 의료영상 3-class 예시
python train.py \
  --data-dir ./medical-dataset \
  --architecture unet \
  --num-classes 3 \
  --class-names "background,anterior,posterior" \
  --dice-weight 0.5 \
  --output checkpoints/hippocampus.pt
```

매 epoch마다 validation mIoU와 클래스별 IoU를 JSON 한 줄로 출력하며, 가장
높은 mIoU checkpoint만 저장합니다.

## ONNX 내보내기

```bash
python export_onnx.py \
  --checkpoint checkpoints/semantic.pt \
  --output exported/custom-model.onnx
```

생성된 `.onnx`와 같은 이름의 `.json`을 웹 실습 앱에서 선택합니다. 계약은
다음과 같습니다.

```text
input  image  float32[1,3,256,256]
output logits float32[1,C,256,256]
```

## MSD Hippocampus 재현

공식 AWS 공개 데이터(약 27.1 MiB)를 내려받고 압축을 풉니다.

```bash
curl -L \
  https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar \
  -o Task04_Hippocampus.tar
tar -xf Task04_Hippocampus.tar

python prepare_msd_hippocampus.py \
  --source-dir ./Task04_Hippocampus \
  --output-dir ./hippocampus-2d \
  --seed 20260729 \
  --max-slices-per-volume 12

python train.py \
  --data-dir ./hippocampus-2d \
  --architecture unet \
  --num-classes 3 \
  --class-names "background,anterior,posterior" \
  --image-size 256 \
  --epochs 5 \
  --batch-size 16 \
  --num-workers 0 \
  --learning-rate 0.0003 \
  --dice-weight 0.5 \
  --seed 20260729 \
  --output checkpoints/unet-hippocampus.pt

python export_onnx.py \
  --checkpoint checkpoints/unet-hippocampus.pt \
  --output exported/unet-hippocampus.onnx
```

전처리 스크립트는 260개 labelled volume을 208 train / 52 validation
patient로 먼저 나눈 뒤 slice를 생성합니다. 같은 환자의 slice가 양쪽 split에
섞이지 않습니다. 이 저장소에서 실제 실행한 5-epoch 결과는
[`../docs/msd-hippocampus-training.md`](../docs/msd-hippocampus-training.md)에
기록했습니다.

데이터 출처:

- MedMNIST(분류 전용): https://github.com/MedMNIST/MedMNIST
- Medical Segmentation Decathlon: https://registry.opendata.aws/msd/
