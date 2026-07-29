# SEG/LAB — Vercel Segmentation Test

학생이 직접 학습한 segmentation 모델을 ONNX로 내보내 브라우저에서 시험하는
완전 정적 실습 앱입니다.

- Production: https://segmentation-static.vercel.app
- Runtime: ONNX Runtime Web + WebAssembly SIMD
- Server inference: 없음
- 학생 모델과 테스트 이미지 업로드: 없음. 브라우저 메모리에서만 처리

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FKimYeongHyeon%2Fvercel-segmentation-test)

## 교육 트랙

### 1. 일반영상 다중 클래스

- 기본 모델: TorchVision LRASPP-MobileNetV3-Large
- 예: `background, road, person, vehicle`
- 손실: multiclass cross entropy
- 평가: validation mIoU 및 클래스별 IoU

### 2. 의료영상 병변·장기

- 기본 모델: compact U-Net
- 배포 예제: MSD Task04 Hippocampus MRI
- 클래스: `background, anterior, posterior`
- 회색조 이미지는 RGB 3채널로 복제
- 마스크 픽셀값은 클래스 ID

MedMNIST는 공식적으로 2D/3D **classification** benchmark이며 배포 NPZ에는
`images`와 이미지 단위 `labels`만 있습니다. segmentation 마스크가 없으므로
U-Net 학습 데이터로 사용하지 않았습니다. 대신 픽셀 마스크와 다중 클래스가
있는 Medical Segmentation Decathlon을 사용했습니다.

두 트랙의 브라우저 계약은 같습니다.

```text
input  image  float32 [1, 3, 256, 256]
output logits float32 [1, C, 256, 256]
mask          uint8   [H, W] class IDs
normalization ImageNet mean/std
```

## 학습과 export

학생용 실행 코드는 [`training/`](./training/)에 있습니다.

```bash
cd training
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py \
  --data-dir ./dataset \
  --architecture lraspp \
  --num-classes 4 \
  --class-names "background,road,person,vehicle" \
  --output checkpoints/custom.pt

python export_onnx.py \
  --checkpoint checkpoints/custom.pt \
  --output exported/custom.onnx
```

의료영상은 `--architecture unet`으로 변경합니다. MSD 다운로드·환자 단위
분리·실제 실행 명령은 [`training/README.md`](./training/README.md), 실제
5-epoch 결과는 [`docs/msd-hippocampus-training.md`](./docs/msd-hippocampus-training.md)에
있습니다.

## Vercel 무료 정적 배포 후보

아래 크기는 원본 또는 공식 배포 파일 기준입니다. 속도는 기기·입력 크기·브라우저
backend의 영향을 크게 받으므로, 확인하지 않은 모델에는 속도를 단정하지 않습니다.

| 모델 | 목적 | 학습 가능성 | 모델 파일 | Vercel 정적 배포 | 브라우저 속도 |
|---|---|---|---:|---|---|
| MediaPipe Selfie Segmentation | 인물 전경 분리 | 공식 학습 코드 없음 | 0.25 MB | 가능 | 이 저장소의 이전 Chrome 측정: warm 239 ms |
| MODNet quantized ONNX | 인물 matting | PyTorch 학습 코드 공개 | 6.63 MB | 가능 | 현재 환경에서 미측정 |
| **LRASPP-MobileNetV3-Large** | 학생용 multiclass semantic segmentation | **TorchVision fine-tuning 가능** | **12.5 MB PyTorch / 12.88 MB ONNX** | **가능** | 아래 실제 검증 기록 참조 |
| DeepLabV3-MobileNetV3-Large | 고정확도 multiclass | TorchVision fine-tuning 가능 | 42.3 MB PyTorch | 가능하나 runtime 포함 용량 주의 | 현재 환경에서 미측정 |
| RMBG-1.4 quantized ONNX | 범용 배경 제거 | 공개 가중치 비상업 제한 | 44.4 MB | 가능하나 runtime 포함 용량 주의 | 현재 환경에서 미측정 |
| BiRefNet 계열 | 고품질 dichotomous segmentation | 모델별 상이 | 대개 수백 MB 이상 | Hobby 100 MB source 제한에 부적합 | 해당 없음 |

### 실제 속도 기록

속도는 `docs/benchmark-results.md`에 실행 날짜, 브라우저, 입력 크기, cold/warm
여부와 함께 기록합니다. 표의 다른 모델과 공정 비교하려면 동일 이미지와 같은
입력 크기로 다시 측정해야 합니다.

## 정적 배포 크기

현재 핵심 payload:

```text
LRASPP ONNX example       12.88 MB
Tiny U-Net MRI example     4.34 MB
ONNX Runtime WASM         13.48 MB
ONNX Runtime loader       0.05 MB
```

Vercel Hobby의 CLI source upload 제한은 100 MB입니다. 모델 학습은 Vercel에서
수행하지 않고, 정적 앱은 결과 ONNX의 추론만 브라우저에 위임합니다.

## 로컬 실행

```bash
npm run check
npm run serve
```

http://localhost:4173 에서 실행합니다.

## 라이선스

- 직접 작성한 앱과 학습 코드는 [MIT](./LICENSE)
- TorchVision LRASPP와 ONNX Runtime 재배포물은 각 upstream 라이선스
- MSD 기반 U-Net 모델과 MRI sample은 CC BY-SA 4.0

자세한 attribution은 [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md)에
기록했습니다.

## 출처

- TorchVision LRASPP: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html
- TorchVision DeepLabV3-MobileNetV3: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_mobilenet_v3_large.html
- MedMNIST (classification): https://github.com/MedMNIST/MedMNIST
- Medical Segmentation Decathlon: https://registry.opendata.aws/msd/
- ONNX Runtime Web deployment: https://onnxruntime.ai/docs/tutorials/web/deploy.html
- Vercel limits: https://vercel.com/docs/limits
