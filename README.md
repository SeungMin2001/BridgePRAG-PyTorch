# BridgePRAG

[🇰🇷 한국어](README.md) | [🇺🇸 English](README.en.md)

**디코더 전용 RAG를 위한 FiD 방식의 질문 조건부 HyperKV 메모리**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

BridgePRAG는 검색된 근거를 학습 가능한 **K/V 메모리 슬롯**으로 압축해 동결된 디코더 전용 LLM에 주입하는 연구 코드베이스다. MergePRAG/HyperKV 계열을 바탕으로, `질문 + 패시지` 경계의 Fusion-in-Decoder 스타일 인코딩과 경량 선형 **KV 어댑터**를 사용한다.

이 프로젝트 기반 논문은 **한국디지털콘텐츠학회**에 채택되었으며, 학회 프로시딩 공개 후 서지 정보를 보완할 예정이다.

<div align="center">
  <img src="assets/BridgePRAG_model.png" alt="질문 조건부 근거를 보정된 K/V 메모리 슬롯으로 인코딩하는 BridgePRAG 아키텍처" width="960">
  <br>
  <sub><b>그림 1.</b> FiD 스타일 질문-패시지 인코딩이 디코더 전용 RAG용 K/V 메모리를 생성한다.</sub>
</div>

## 핵심 아이디어

패시지만 인코딩하는 대신, BridgePRAG는 `question + passage`를 먼저 인코딩한다. HyperKV 슬롯을 생성하고 선형 어댑터로 보정한 뒤 선택한 트랜스포머 레이어에 주입한다.

- **질문 조건부 메모리**: `question + passage`에서 HyperKV 슬롯을 생성한다.
- **FiD 스타일 경계**: 패시지 전용 메모리를 질문 인지형 근거 인코딩으로 확장한다.
- **KV 어댑터 보정**: 생성된 key/value 슬롯에 선형 잔차 보정을 학습한다.
- **디코더 전용 호환**: 기본 LLM을 미세조정하지 않고 특정 레이어에 메모리를 주입한다.

## 설치

```bash
git clone https://github.com/SeungMin2001/BridgePRAG-PyTorch.git
cd BridgePRAG-PyTorch
python -m pip install -e ".[dev]"
```

GPU에서는 CUDA 버전에 맞는 PyTorch를 먼저 설치한다.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[dev]"
```

## 빠른 시작

```bash
bridgeprag train --data examples/data/tiny_memory.jsonl --output runs/tiny_bridgeprag.pt --model Qwen/Qwen2.5-0.5B --num-kv 8 --hidden-dim 512 --critical-layer 8 --epochs 1 --max-steps 3
bridgeprag infer --checkpoint runs/tiny_bridgeprag.pt --question "What does the KV adapter correct?" --passage "The KV adapter applies lightweight linear corrections to generated key and value slots."
```

## 동작 방식

1. 검색 근거를 `passage` 또는 `question + passage`로 인코딩한다.
2. 어텐티브 풀링으로 토큰 특징을 슬롯별 은닉 상태로 투영한다.
3. K와 V 메모리 슬롯을 생성하고, 선형 KV 어댑터를 적용한다.
4. 선택한 디코더 레이어에서 cross-attention으로 메모리를 주입한다.
5. 여러 패시지는 직교 슬롯 합성으로 병합한다.

| 변형 | 메모리 인코더 입력 | 슬롯 보정 | 기대 효과 |
| --- | --- | --- | --- |
| 패시지 전용 HyperKV | passage | 없음 | 외부 메모리 압축 |
| BridgePRAG | question + passage | 선형 KV 어댑터 | 질문 인지형·보정된 메모리 |

전체 연구 노트는 [docs/method.md](docs/method.md)에서 볼 수 있다.

## 검증 결과 스냅샷

현재 엔터티 스타일 검증 세트에서 질문+패시지 메모리는 패시지 전용 메모리보다 더 높은 정확도와 F1, 더 짧은 평균 지연 시간을 보였다.

| 지표 | 질문+패시지 | 패시지 전용 | 차이 |
| --- | ---: | ---: | ---: |
| 정확도 (Hit) | 84.00 | 41.00 | +43.00 |
| Precision | 69.78 | 45.58 | +24.21 |
| Recall | 86.73 | 63.28 | +23.45 |
| F1 Score | 76.19 | 51.52 | +24.67 |
| QA Score | 38.26 | 26.16 | +12.10 |
| 평균 시간 (초) | 4.17 | 4.83 | -0.66 |

<div align="center">
  <img src="assets/bridgeprag_training_curves.png" alt="BridgePRAG와 패시지 전용 메모리의 테스트 학습 곡선" width="960">
  <br>
  <sub><b>그림 2.</b> 질문 조건부 메모리는 패시지 전용 기준선보다 빠르게 수렴하고 더 높은 정확도에 도달한다.</sub>
</div>

## 데이터와 구성

최소 JSONL 형식:

```json
{"source_id":"ex1","passage":"...","question":"...","answer":"...","full_answer":"..."}
```

```text
bridgeprag/          # 설치 가능한 연구 패키지
examples/            # 실행 가능한 작은 예제
docs/                # 방법 및 재현 노트
assets/              # 아키텍처와 실험 그림
tests/               # 형태·데이터 테스트
```

## 연구 상태 및 인용

이 저장소는 연구 프리뷰다. 대형 체크포인트는 Git에 넣지 않으며, 배포 시 GitHub Releases 또는 Hugging Face Hub를 사용한다.

```bibtex
@software{bridgeprag2026,
  title = {BridgePRAG: FiD-Inspired Question-Conditioned HyperKV Memory for Decoder-Only RAG},
  author = {SeungMin2001},
  year = {2026},
  url = {https://github.com/SeungMin2001/BridgePRAG-PyTorch}
}
```

MIT 라이선스이며 자세한 내용은 [LICENSE](LICENSE)를 참고한다.

