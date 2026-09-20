# BridgePRAG

[🇰🇷 한국어](README.md) | [🇺🇸 English](README.en.md)

**디코더 전용 검색 증강 생성을 위한 FiD 기반 질문 조건부 HyperKV 메모리.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Research Preview](https://img.shields.io/badge/status-research%20preview-purple)](#연구-상태)
[![Paper](https://img.shields.io/badge/paper-accepted%20%40%20KDCS-0a7c66)](#연구-상태)

BridgePRAG는 검색된 근거를 학습 가능한 **K/V 메모리 슬롯**으로 인코딩하고,
이를 동결된 디코더 전용 LLM에 주입하기 위한 간결한 연구 코드베이스입니다.
MergePRAG/HyperKV 방향에서 출발해, Fusion-in-Decoder 방식의 질문-패시지
경계와 보정을 위한 경량 선형 **KV 어댑터**를 사용하도록 메모리 인코더를
변경했습니다.

이 프로젝트를 기반으로 한 논문은 **한국디지털콘텐츠학회**에 채택되었습니다.
프로시딩이 공개되면 출판 세부 정보를 추가할 예정입니다.

<div align="center">
  <img src="assets/BridgePRAG_model.png" alt="BridgePRAG 아키텍처: 질문 조건부 근거가 보정된 K/V 메모리 슬롯으로 인코딩됩니다" width="960">
  <br>
  <sub><b>그림 1.</b> FiD 방식의 질문-패시지 인코딩이 디코더 전용 RAG를 위한 보정된 K/V 메모리 슬롯을 생성합니다.</sub>
</div>

## 한눈에 보기

BridgePRAG는 간단한 질문에서 출발합니다. 검색된 패시지를 사용자의 질문을
이미 인지하는 작고 학습 가능한 K/V 메모리로 압축할 수 있을까요?

BridgePRAG는 패시지만으로 메모리를 생성하는 대신 `question + passage`를
인코딩하고, HyperKV 슬롯을 만든 뒤 경량 어댑터로 보정하여 그 결과 메모리를
동결된 디코더 전용 모델에 주입합니다. 목표는 디코더 전용 RAG 메모리를
탐구하는 연구자와 엔지니어를 위한 간결한 참조 구현을 제공하는 것입니다.

**현재 검증 결과 요약**

- 엔터티 방식 검증 설정에서 패시지 전용 메모리보다 적중 정확도 +43.00.
- 패시지 전용 메모리보다 F1 점수 +24.67.
- 보고된 비교에서 더 빠른 평균 추론 시간.
- 테스트 세트 및 학습/검증 시각화에서 패시지 전용 기준선보다 빠른 수렴과 높은 정확도 확인.

## BridgePRAG를 사용하는 이유

- **질문 조건부 메모리**: HyperKV 슬롯을 생성하기 전에 `question + passage`를 인코딩합니다.
- **FiD 기반 경계**: 패시지 전용 메모리에서 질의 인지형 근거 인코딩으로 확장합니다.
- **KV 어댑터 보정**: 생성된 key/value 슬롯에 대한 선형 잔차 보정을 학습합니다.
- **디코더 전용 모델 호환**: 기본 LLM을 미세 조정하지 않고 선택한 트랜스포머 레이어를 통해 메모리를 주입합니다.
- **재현 가능한 연구 구성**: 설치 가능한 패키지, 예제 데이터, 학습 CLI, 추론 CLI, 테스트, 그림, 인용 파일을 제공합니다.

## 설치

```bash
git clone https://github.com/SeungMin2001/BridgePRAG.git
cd BridgePRAG
python -m pip install -e ".[dev]"
```

GPU에서 실행하려면 먼저 CUDA 버전에 맞는 PyTorch 빌드를 설치하세요.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[dev]"
```

## 빠른 시작

작은 BridgePRAG 메모리 생성기를 학습합니다.

```bash
bridgeprag train \
  --data examples/data/tiny_memory.jsonl \
  --output runs/tiny_bridgeprag.pt \
  --model Qwen/Qwen2.5-0.5B \
  --num-kv 8 \
  --hidden-dim 512 \
  --critical-layer 8 \
  --epochs 1 \
  --max-steps 3
```

질문 조건부 메모리 추론을 실행합니다.

```bash
bridgeprag infer \
  --checkpoint runs/tiny_bridgeprag.pt \
  --question "What does the KV adapter correct?" \
  --passage "The KV adapter applies lightweight linear corrections to generated key and value slots."
```

Python API:

```python
from bridgeprag import BridgePRAG

bridge = BridgePRAG.from_checkpoint("runs/tiny_bridgeprag.pt")
answer = bridge.generate(
    question="What does the KV adapter correct?",
    passages=[
        "The KV adapter applies lightweight linear corrections to generated key and value slots."
    ],
)
print(answer)
```

## 방법

BridgePRAG는 기본 LLM을 동결한 상태에서 작은 HyperKV 생성기만 학습합니다.

1. 검색된 근거를 `passage` 또는 `question + passage`로 인코딩합니다.
2. 어텐티브 풀링으로 토큰 특징을 슬롯별 은닉 상태에 투영합니다.
3. `K`와 `V` 메모리 슬롯을 생성합니다.
4. 선형 KV 어댑터를 적용합니다: `K = K + A_k(K)`, `V = V + A_v(V)`.
5. 교차 어텐션으로 선택한 디코더 레이어에 메모리를 주입합니다.
6. 직교 슬롯 합성을 통해 여러 패시지를 병합합니다.

주요 차이는 다음과 같습니다.

| 변형 | 메모리 인코더 입력 | 슬롯 보정 | 의도한 효과 |
| --- | --- | --- | --- |
| 패시지 전용 HyperKV | passage | 없음 | 간결한 외부 메모리 |
| BridgePRAG | question + passage | 선형 KV 어댑터 | 질의 인지형 보정 메모리 |

[docs/method.md](docs/method.md)에서 전체 연구 노트를 확인할 수 있습니다.

## 결과 요약

현재 BridgePRAG 브랜치는 엔터티 방식 검증 설정에서 다음과 같은
질문+패시지 메모리와 패시지 전용 메모리의 비교 결과를 보고합니다.

| 지표 | 질문+패시지 | 패시지 전용 | 차이 |
| --- | ---: | ---: | ---: |
| 정확도 (적중) | 84.00 | 41.00 | +43.00 |
| 정밀도 | 69.78 | 45.58 | +24.21 |
| 재현율 | 86.73 | 63.28 | +23.45 |
| F1 점수 | 76.19 | 51.52 | +24.67 |
| QA Score | 38.26 | 26.16 | +12.10 |
| 평균 시간(초) | 4.17 | 4.83 | -0.66 |

**핵심 결론.** 질문 조건부 메모리는 검색 근거 기반 답변 품질과 효율을
모두 개선합니다. 이 검증 결과에서 더 높은 적중 정확도, F1, QA 점수와
더 낮은 평균 지연 시간을 보였습니다.

<details>
<summary>렌더링된 지표 표</summary>

<div align="center">
  <img src="assets/bridgeprag_evaluation_metrics_table.png" alt="질문-패시지 메모리와 패시지 전용 메모리를 비교한 BridgePRAG 평가 지표 표" width="560">
  <br>
  <sub><b>표 1.</b> 발표 및 공유를 위해 렌더링한 지표 표입니다.</sub>
</div>

</details>

### 테스트 세트 학습 곡선

원본 테스트 세트 시각화를 주요 비교 그림으로 유지했습니다.
질문 조건부 BridgePRAG 모델의 정확도가 가파르게 상승하는 동안
패시지 전용 기준선은 더 느리게 개선되는 모습을 보여줍니다. 손실 곡선도
같은 양상을 보입니다. BridgePRAG가 더 일찍 낮은 손실에 도달하는 것은
질문 인지형 메모리 슬롯을 패시지 전용 메모리보다 효과적으로 사용함을 시사합니다.

<div align="center">
  <img src="assets/bridgeprag_training_curves.png" alt="BridgePRAG 질문-패시지 메모리와 패시지 전용 메모리를 비교한 테스트 세트 학습 곡선" width="960">
  <br>
  <sub><b>그림 2.</b> 질문 조건부 BridgePRAG 메모리와 패시지 전용 메모리의 테스트 세트 비교입니다.</sub>
</div>

### 학습 곡선 비교

아래 두 학습 시각화는 동일한 에폭 예산에서 BridgePRAG와 패시지 전용
기준선을 비교합니다. BridgePRAG는 검증 손실을 더 빠르게 줄이고 거의
포화된 검증 정확도에 도달하지만, 패시지 전용 기준선은 더 느리게 개선되며
정확도에 한계를 보입니다.

| BridgePRAG: 질문 조건부 K/V 메모리 | 패시지 전용 HyperKV 메모리 |
| --- | --- |
| <img src="assets/bridgeprag_train_val_loss_accuracy.png" alt="BridgePRAG 학습 및 검증 손실/정확도 곡선" width="480"> | <img src="assets/passage_only_train_val_loss_accuracy.png" alt="패시지 전용 학습 및 검증 손실/정확도 곡선" width="480"> |

<div align="center">
  <sub><b>그림 3.</b> 질문 조건부 메모리는 더 가파른 손실 곡선을 학습하고 패시지 전용 메모리보다 훨씬 높은 검증 정확도에 도달합니다.</sub>
</div>

## 데이터셋 형식

최소 JSONL 형식:

```json
{"source_id":"ex1","passage":"...","question":"...","answer":"...","full_answer":"..."}
```

확장된 행에는 `qas`, `atomic_qas`, `final_qas`, `hard_negatives`가
포함될 수 있으며, 로더는 이를 독립적인 메모리 지도 학습 예제로 펼칩니다.

## 저장소 구성

```text
bridgeprag/          # 설치 가능한 연구 패키지
  config.py          # BridgePRAGConfig
  memory.py          # HyperKV 생성기, FiD 방식 인코딩, 주입 훅
  runtime.py         # 고수준 BridgePRAG 래퍼
  trainer.py         # 간결한 동결 LLM 학습 루프
  cli.py             # bridgeprag train / infer
examples/            # 실행 가능한 작은 예제
docs/                # 방법 및 재현 노트
assets/              # 아키텍처 및 실험 그림
tests/               # 빠른 CI를 위한 형태/데이터 테스트
```

## 연구 상태

이 저장소는 연구 프리뷰입니다. 코드는 BridgePRAG 메커니즘을 쉽게 검토하고,
실행하고, 수정할 수 있도록 구성했습니다. 대형 체크포인트는 Git에 저장하지
않으며, 배포 체크포인트는 GitHub Releases 또는 Hugging Face Hub에 첨부해야 합니다.

논문 상태:

- 한국디지털콘텐츠학회에 채택되었습니다.
- 출판 정보, 권·호, 페이지 및 DOI는 아직 확정되지 않았습니다.
- 공식 프로시딩 항목이 공개되면 인용 블록을 업데이트할 예정입니다.

## 로드맵

- [ ] 소형 사전 학습 BridgePRAG 체크포인트 공개.
- [ ] 패시지 전용 메모리와 질문+패시지 메모리를 비교하는 전체 벤치마크 스크립트 추가.
- [ ] 공개 체크포인트용 Hugging Face 모델 카드 메타데이터 추가.
- [ ] 채택 논문 출판 후 인용 정보 업데이트.

## 인용

```bibtex
@software{bridgeprag2026,
  title = {BridgePRAG: FiD-Inspired Question-Conditioned HyperKV Memory for Decoder-Only RAG},
  author = {SeungMin2001},
  year = {2026},
  url = {https://github.com/SeungMin2001/BridgePRAG}
}
```

## 참고 자료

- 검색 증강 생성을 위한 MergePRAG 방식의 HyperKV 메모리.
- 검색된 근거를 위한 Fusion-in-Decoder 방식의 질문-패시지 인코딩.
- 디코더 전용 KV 캐시 및 교차 어텐션 메모리 주입.

## 라이선스

MIT. 자세한 내용은 [LICENSE](LICENSE)를 참고하세요.
