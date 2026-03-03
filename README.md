# CoRes-Embedding (Compositional Residual Embedding)

**"The Unreasonable Effectiveness of Decomposition: Latent Stability in Compositional Residual Representations"**

분해의 비합리적 효율성: 합성 잔차 표현의 잠재 공간 안정성 연구

## 가설 (Hypothesis)

제한된 Latent Dimension $D$가 주어졌을 때,  
정보를 $z_{global} \in \mathbb{R}^D$ (통짜)로 압축하는 것보다,  
$z_{total} = \sum z_{concepts} + z_{residual}$ (구조화된 개념 + 잔차)로 분해하여 저장하는 것이 **노이즈에 대한 내성(Stability)**과 **새로운 데이터에 대한 적응력(Few-shot Generalization)**이 더 높다.

## 프로젝트 구조

```
cores-embedding/
├── configs/                # 실험 설정 파일
│   └── default.yaml
├── data/                   # 데이터 로더
│   ├── __init__.py
│   ├── clevr.py
│   ├── mpi3d.py
│   └── celeba.py
├── models/                 # 모델 아키텍처
│   ├── __init__.py
│   ├── backbone.py         # ResNet-18 Shared Feature Extractor
│   ├── baseline.py         # Monolithic Baseline Model
│   ├── cores.py            # CoRes (Proposed) Model
│   └── components.py       # Concept Branch, Residual Branch
├── evaluation/             # 평가 프로토콜
│   ├── __init__.py
│   ├── perturbation.py     # 실험 1: 섭동 안정성
│   ├── fewshot.py          # 실험 2: Few-shot Generalization
│   └── manifold.py         # 실험 3: Manifold Smoothness
├── training/               # 학습 루프
│   ├── __init__.py
│   ├── trainer.py
│   └── losses.py
├── visualization/          # 결과 시각화
│   ├── __init__.py
│   └── plots.py
├── scripts/                # 실행 스크립트
│   ├── train.py
│   ├── evaluate.py
│   └── run_all_experiments.py
├── requirements.txt
└── README.md
```

## 실행 방법

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 학습
```bash
# Baseline 모델 학습
python scripts/train.py --config configs/default.yaml --model baseline

# CoRes 모델 학습
python scripts/train.py --config configs/default.yaml --model cores
```

### 3. 평가
```bash
# 전체 평가 실행
python scripts/evaluate.py --config configs/default.yaml

# 모든 실험 자동 실행 (학습 + 평가 + 시각화)
python scripts/run_all_experiments.py --config configs/default.yaml
```

## 평가 프로토콜

1. **섭동 안정성 (Perturbation Stability)**: 노이즈에 대한 임베딩 코사인 유사도 유지력
2. **Few-shot Generalization**: 소량 데이터로 새로운 태스크 적응력
3. **Manifold Smoothness**: 잠재 공간 보간 안정성
