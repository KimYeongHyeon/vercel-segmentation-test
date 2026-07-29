# Browser benchmark results

속도 표기 규칙:

- Cold: 모델 다운로드와 session 초기화를 포함
- Warm: 같은 session에서 모델 초기화 이후의 `session.run()`만 측정
- 모든 비교는 입력 크기, 브라우저, 실행 backend를 함께 기록

## 2026-07-29 — Apple Silicon Mac / Chrome

| 모델 | 위치 | 입력 | Backend | 모델 준비 | 첫 추론 | Warm |
|---|---|---:|---|---:|---:|---:|
| MediaPipe Selfie Segmentation | 이전 Vercel 배포 | 256×256 내부 resize | WASM | 3,374–8,345 ms | — | 239 ms |
| LRASPP-MobileNetV3-Large ONNX | localhost | 256×256 | WASM, 1 thread | 223 ms (cached) | 56 ms | 45 ms |
| LRASPP-MobileNetV3-Large ONNX | Vercel production | 256×256 | WASM SIMD | 2,331 ms | 17 ms | **15 ms** |
| Tiny U-Net Hippocampus 3-class | localhost | 256×256 | WASM, 1 thread | 미기록 | 306 ms | 291 ms |
| Tiny U-Net Hippocampus 3-class | Vercel production | 256×256 | WASM SIMD | 2,855 ms | 92 ms | **77 ms** |

MediaPipe와 LRASPP는 목적과 출력이 달라 정확도 비교 대상으로 사용하지 않습니다.
이 표의 시간은 배포 feasibility를 판단하기 위한 runtime 기록입니다.

Production URL: https://segmentation-static.vercel.app

- 모델 준비: 버튼 click부터 ONNX session 생성 완료 UI까지의 wall-clock
- 첫/Warm 추론: 앱에 표시된 `session.run()` 시간. 이미지 전처리와 canvas
  rendering은 제외
- Production 응답은 COOP `same-origin`, COEP `require-corp`를 확인
- U-Net 결과: `1×3×256×256`, 세 클래스 모두 활성
- LRASPP 결과: `1×21×256×256`
- Chrome console error 0, CDP `Network.loadingFailed` 0, HTTP 4xx/5xx 0

두 모델의 출력 해상도는 같지만 계산 graph가 다릅니다. 이 실행에서는
LRASPP가 U-Net보다 빨랐습니다. 수치는 Apple Silicon Mac의 단일 브라우저
측정이며 모바일이나 저사양 교육용 PC의 보장값이 아닙니다.
