# Machine Learning Algorithms Documentation

This document provides detailed explanations of all machine learning algorithms used in the Bitcoin Price Prediction System.

---

## Table of Contents

1. [OneR (One Rule)](#1-oner-one-rule)
2. [Naive Bayes](#2-naive-bayes)
3. [Decision Tree (J48/C4.5)](#3-decision-tree-j48c45)
4. [Random Forest](#4-random-forest)
5. [SVM (Support Vector Machine)](#5-svm-support-vector-machine)
6. [Algorithm Comparison](#6-algorithm-comparison)

---

## 1. OneR (One Rule)

### 개요
OneR은 **가장 단순한 규칙 기반 분류기**로, 단 하나의 속성(attribute)만을 사용하여 분류를 수행합니다. WEKA에서 제공하는 베이스라인 알고리즘으로, 복잡한 모델과의 성능 비교를 위해 자주 사용됩니다.

### 동작 원리

1. **속성별 규칙 생성**
   - 각 속성에 대해 간단한 분류 규칙을 생성
   - Numeric 속성: 구간(bin)으로 나누어 규칙 생성
   - Nominal 속성: 각 값에 대해 규칙 생성

2. **최적 속성 선택**
   - 각 속성의 오류율(error rate) 계산
   - 가장 낮은 오류율을 가진 속성을 선택

3. **예측**
   - 선택된 속성의 규칙만 사용하여 분류

### 계산 과정 (예시)

**데이터셋:**
```
| Volume Spike | MA Cross | RSI Signal | Price Direction |
|--------------|----------|------------|-----------------|
| high         | golden   | neutral    | UP              |
| normal       | neutral  | overbought | DOWN            |
| high         | golden   | neutral    | UP              |
| low          | dead     | oversold   | DOWN            |
| normal       | neutral  | neutral    | STABLE          |
```

**Step 1: 각 속성의 오류율 계산**

- **Volume Spike 규칙:**
  - high → UP (2/2 correct)
  - normal → STABLE (1/2 correct)
  - low → DOWN (1/1 correct)
  - **Error rate: 1/5 = 20%**

- **MA Cross 규칙:**
  - golden → UP (2/2 correct)
  - neutral → STABLE (1/2 correct)
  - dead → DOWN (1/1 correct)
  - **Error rate: 1/5 = 20%**

- **RSI Signal 규칙:**
  - neutral → UP (2/3 correct)
  - overbought → DOWN (1/1 correct)
  - oversold → DOWN (1/1 correct)
  - **Error rate: 1/5 = 20%**

**Step 2: 최적 규칙 선택**
- 모든 속성이 동일한 오류율을 가지므로, 첫 번째 속성(Volume Spike)을 선택

**Step 3: 예측**
- 새 데이터의 Volume Spike 값만 보고 예측
- 예: Volume Spike = high → **예측: UP**

### 장단점

**장점:**
- 매우 간단하고 해석하기 쉬움
- 빠른 학습 속도
- 베이스라인 모델로 유용

**단점:**
- 정확도가 낮음 (단일 속성만 사용)
- 복잡한 패턴을 포착하지 못함
- 속성 간 상호작용을 고려하지 않음

### 비트코인 예측에서의 역할
- 어떤 기술적 지표가 가장 중요한지 파악
- 다른 알고리즘의 성능 평가 기준점
- 예상 정확도: **50-60%** (랜덤보다 약간 나은 수준)

---

## 2. Naive Bayes

### 개요
Naive Bayes는 **베이즈 정리(Bayes' Theorem)**를 기반으로 하는 확률적 분류기입니다. 모든 속성이 독립적이라는 "naive"한 가정을 하지만, 실제로는 많은 경우에 잘 작동합니다.

### 동작 원리

**베이즈 정리:**
```
P(Y|X) = P(X|Y) × P(Y) / P(X)
```

- P(Y|X): 속성 X가 주어졌을 때 클래스 Y의 확률 (사후 확률)
- P(X|Y): 클래스 Y일 때 속성 X가 나타날 확률 (우도, likelihood)
- P(Y): 클래스 Y의 사전 확률 (prior)
- P(X): 속성 X의 확률 (정규화 상수)

### 계산 과정 (예시)

**학습 데이터:**
```
Price Direction 분포:
- UP: 26/199 = 13.1%
- DOWN: 27/199 = 13.6%
- STABLE: 146/199 = 73.4%
```

**새 데이터:**
```
close = 165,000,000원
volume = 50.5
ma_cross = golden
rsi_signal = neutral
volume_spike = high
```

**Step 1: 사전 확률 (Prior)**
```
P(UP) = 0.131
P(DOWN) = 0.136
P(STABLE) = 0.734
```

**Step 2: 우도 (Likelihood) 계산**

UP 클래스에 대한 우도:
```
P(ma_cross=golden | UP) = 0.65  (17/26)
P(rsi_signal=neutral | UP) = 0.54  (14/26)
P(volume_spike=high | UP) = 0.46  (12/26)
P(close=165M | UP) ~ Gaussian(μ=168M, σ=3M)
P(volume=50.5 | UP) ~ Gaussian(μ=45, σ=10)
```

**Step 3: 사후 확률 계산**
```
P(UP | X) ∝ P(UP) × P(ma_cross=golden|UP) × P(rsi_signal=neutral|UP) × ...
         = 0.131 × 0.65 × 0.54 × 0.46 × ... = 0.00523

P(DOWN | X) ∝ ... = 0.00234
P(STABLE | X) ∝ ... = 0.01456
```

**Step 4: 정규화 및 예측**
```
Total = 0.00523 + 0.00234 + 0.01456 = 0.02213

P(UP | X) = 0.00523 / 0.02213 = 23.6%
P(DOWN | X) = 0.00234 / 0.02213 = 10.6%
P(STABLE | X) = 0.01456 / 0.02213 = 65.8%

예측: STABLE (가장 높은 확률)
```

### 장단점

**장점:**
- 빠르고 효율적
- 소량의 학습 데이터로도 작동
- 확률 값 제공 (예측 신뢰도)
- 다중 클래스 분류에 적합

**단점:**
- 속성 독립 가정이 현실적이지 않음
- 연속형 변수는 정규분포를 가정 (GaussianNB)

### 비트코인 예측에서의 성능
- 예상 정확도: **68-72%**
- 클래스 불균형에 민감
- 확률 기반으로 불확실성 정량화 가능

---

## 3. Decision Tree (J48/C4.5)

### 개요
Decision Tree는 데이터를 **트리 구조**로 분할하여 분류하는 알고리즘입니다. J48은 WEKA에서 구현한 **C4.5 알고리즘**으로, ID3의 개선 버전입니다.

### 동작 원리

1. **정보 이득(Information Gain) 계산**
   - 각 속성이 데이터를 얼마나 잘 분할하는지 측정
   - 엔트로피(Entropy) 감소량 계산

2. **노드 분할**
   - 가장 높은 정보 이득을 가진 속성 선택
   - 해당 속성으로 데이터 분할

3. **재귀적 분할**
   - 각 자식 노드에 대해 1-2 단계 반복
   - 종료 조건: 순수 노드, 최소 샘플 수, 최대 깊이

4. **가지치기(Pruning)**
   - 과적합 방지를 위해 불필요한 가지 제거

### 계산 과정 (예시)

**Step 1: 엔트로피 계산**

전체 데이터의 엔트로피:
```
P(UP) = 26/199 = 0.131
P(DOWN) = 27/199 = 0.136
P(STABLE) = 146/199 = 0.734

H(전체) = -[0.131×log₂(0.131) + 0.136×log₂(0.136) + 0.734×log₂(0.734)]
        = -[-0.381 - 0.387 - 0.274]
        = 1.042 bits
```

**Step 2: 속성별 정보 이득 계산**

MA Cross로 분할 시:
```
MA Cross = golden (50개):
  - UP: 17개, DOWN: 5개, STABLE: 28개
  - H(golden) = 1.164

MA Cross = dead (40개):
  - UP: 3개, DOWN: 18개, STABLE: 19개
  - H(dead) = 1.089

MA Cross = neutral (109개):
  - UP: 6개, DOWN: 4개, STABLE: 99개
  - H(neutral) = 0.454

정보 이득 = H(전체) - [P(golden)×H(golden) + P(dead)×H(dead) + P(neutral)×H(neutral)]
          = 1.042 - [(50/199)×1.164 + (40/199)×1.089 + (109/199)×0.454]
          = 1.042 - 0.801
          = 0.241 bits
```

**Step 3: 트리 구축**
```
Root: MA Cross
├─ golden (50) → UP 확률 높음
│   ├─ volume_spike = high → UP (12)
│   └─ volume_spike = normal → STABLE (28)
├─ dead (40) → DOWN 확률 높음
│   └─ rsi_signal = oversold → DOWN (18)
└─ neutral (109) → STABLE (99)
```

### J48 파라미터

- **minNumObj (최소 객체 수):** 노드 분할을 위한 최소 인스턴스 수 (기본 2)
- **confidenceFactor:** 가지치기 신뢰도 (기본 0.25)
- **unpruned:** 가지치기 여부 (기본 false)

### 장단점

**장점:**
- 해석이 매우 쉬움 (시각화 가능)
- 비선형 패턴 포착 가능
- 속성 중요도 파악 가능
- 범주형/연속형 데이터 모두 처리

**단점:**
- 과적합 경향
- 불안정성 (데이터 변화에 민감)
- 클래스 불균형에 취약

### 비트코인 예측에서의 성능
- 예상 정확도: **56-62%**
- 기술적 지표 간 관계 파악에 유용
- 트리 깊이 제한으로 과적합 방지 필요

---

## 4. Random Forest

### 개요
Random Forest는 **여러 개의 Decision Tree를 앙상블(ensemble)**한 알고리즘입니다. Bagging(Bootstrap Aggregating) 기법을 사용하여 과적합을 줄이고 일반화 성능을 향상시킵니다.

### 동작 원리

1. **부트스트랩 샘플링**
   - 원본 데이터에서 중복을 허용하여 N개의 서브셋 생성
   - 각 서브셋은 원본 데이터의 약 63%를 포함

2. **랜덤 속성 선택**
   - 각 노드에서 √(전체 속성 개수)만큼 랜덤하게 속성 선택
   - 예: 8개 속성 → 각 노드에서 √8 ≈ 3개 속성만 고려

3. **다수결 투표 (Majority Voting)**
   - 각 트리의 예측 결과를 수집
   - 가장 많이 투표된 클래스를 최종 예측

### 계산 과정 (예시)

**100개 트리 생성 (n_estimators=100):**

```
트리 1: 데이터 [1, 3, 5, 7, ...] → 예측: STABLE
트리 2: 데이터 [2, 2, 6, 8, ...] → 예측: UP
트리 3: 데이터 [1, 4, 4, 9, ...] → 예측: STABLE
...
트리 100: 데이터 [3, 5, 7, 10, ...] → 예측: STABLE
```

**투표 결과:**
```
UP: 18표
DOWN: 12표
STABLE: 70표

최종 예측: STABLE
확률: P(UP)=18%, P(DOWN)=12%, P(STABLE)=70%
```

### 하이퍼파라미터

- **n_estimators:** 트리 개수 (기본 100)
- **max_depth:** 트리 최대 깊이 (None = 제한 없음)
- **min_samples_split:** 노드 분할 최소 샘플 수 (기본 2)
- **max_features:** 각 분할에서 고려할 최대 속성 수 (기본 √n)

### 장단점

**장점:**
- 높은 정확도
- 과적합 방지
- 속성 중요도 자동 계산
- 병렬 처리 가능

**단점:**
- 해석이 어려움 (블랙박스)
- 메모리 사용량 큼
- 학습 시간 오래 걸림

### 비트코인 예측에서의 성능
- 예상 정확도: **68-73%**
- 앙상블 효과로 안정적인 성능
- 본 프로젝트에서 2위 성능

---

## 5. SVM (Support Vector Machine)

### 개요
SVM은 **최적의 결정 경계(Decision Boundary)**를 찾는 알고리즘입니다. 데이터를 고차원 공간으로 매핑하여 비선형 분류 문제를 해결합니다.

### 동작 원리

1. **서포트 벡터 찾기**
   - 각 클래스의 경계에 가장 가까운 데이터 포인트
   - 이 점들이 결정 경계를 정의

2. **마진 최대화**
   - 결정 경계와 서포트 벡터 사이의 거리(마진)를 최대화
   - 일반화 성능 향상

3. **커널 트릭 (Kernel Trick)**
   - RBF(Radial Basis Function) 커널 사용
   - 비선형 패턴을 고차원 공간에서 선형으로 분리

### 계산 과정 (예시)

**Step 1: 데이터 정규화**
```
X = StandardScaler().fit_transform(X)
```

**Step 2: RBF 커널 적용**
```
K(x₁, x₂) = exp(-γ × ||x₁ - x₂||²)

γ (gamma): 커널 계수 (기본 1/n_features = 1/8 = 0.125)
```

**Step 3: 최적화 문제 해결**
```
minimize: (1/2)||w||² + C×Σξᵢ

subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ
            ξᵢ ≥ 0

w: 가중치 벡터
b: 편향(bias)
C: 규제 파라미터 (기본 1.0)
ξ: 오류 허용 변수
```

**Step 4: 다중 클래스 분류 (One-vs-One)**
```
3개 클래스 → 3개 이진 분류기:
1. UP vs DOWN
2. UP vs STABLE
3. DOWN vs STABLE

투표를 통해 최종 클래스 결정
```

### SVM을 선택한 이유

1. **비선형 패턴 포착**
   - 비트코인 가격은 복잡한 비선형 관계
   - RBF 커널로 효과적으로 모델링

2. **고차원 데이터에 강함**
   - 8개 속성의 상호작용 포착
   - 과적합 방지 메커니즘 내장

3. **클래스 불균형 처리**
   - STABLE 클래스가 73%를 차지
   - class_weight='balanced' 옵션으로 대응 가능

4. **일반화 성능**
   - 마진 최대화로 새 데이터에 대한 예측 성능 우수
   - 본 프로젝트에서 **75% 정확도로 1위**

### 하이퍼파라미터

- **C (규제 파라미터):** 오류 허용 정도 (기본 1.0)
- **kernel:** 커널 함수 ('rbf', 'linear', 'poly')
- **gamma:** RBF 커널 계수 ('scale', 'auto', 또는 float)
- **probability:** 확률 추정 활성화 (기본 False)

### 장단점

**장점:**
- 높은 정확도
- 비선형 패턴 포착
- 고차원 데이터에 효과적
- 과적합 방지

**단점:**
- 학습 시간 오래 걸림 (특히 대규모 데이터)
- 하이퍼파라미터 튜닝 필요
- 해석이 어려움

### 비트코인 예측에서의 성능
- **정확도: 75.0%** (최고 성능)
- 교차 검증: 74.5% ± 4.6%
- STABLE 클래스 완벽 예측 (36/36)
- UP/DOWN 예측은 어려움 (샘플 부족)

---

## 6. Algorithm Comparison

### 성능 비교표

| Algorithm | Accuracy | CV Score | Training Time | Interpretability | Complexity |
|-----------|----------|----------|---------------|------------------|------------|
| **OneR** | 50-55% | 52% ± 8% | 매우 빠름 (< 0.1s) | ⭐⭐⭐⭐⭐ | 매우 낮음 |
| **Naive Bayes** | 68-72% | 70% ± 6% | 매우 빠름 (< 0.1s) | ⭐⭐⭐⭐ | 낮음 |
| **Decision Tree** | 56-62% | 58% ± 8% | 빠름 (< 0.5s) | ⭐⭐⭐⭐⭐ | 중간 |
| **Random Forest** | 68-73% | 70% ± 5% | 중간 (2-3s) | ⭐⭐ | 높음 |
| **SVM** | **73-77%** | **75% ± 5%** | 느림 (5-10s) | ⭐ | 매우 높음 |

### 알고리즘 선택 가이드

**1. 해석 가능성이 중요한 경우:**
- Decision Tree (J48)
- OneR

**2. 빠른 프로토타이핑:**
- Naive Bayes
- OneR

**3. 최고 정확도가 필요한 경우:**
- **SVM** (본 프로젝트 최종 선택)
- Random Forest

**4. 안정적인 성능:**
- Random Forest
- SVM

### 비트코인 예측에서의 특징

**클래스 불균형 문제:**
```
STABLE: 146개 (73.4%) - 압도적 다수
UP: 26개 (13.1%)
DOWN: 27개 (13.6%)
```

**영향:**
- 모든 모델이 STABLE 예측에 편향
- UP/DOWN 예측 정확도 낮음
- 전체 정확도는 높지만 실용성 떨어짐

**개선 방안:**
1. 더 많은 데이터 수집 (1년치 → 8,760개)
2. SMOTE 등 샘플링 기법 적용
3. class_weight 조정
4. 임계값 조정 (0.3% → 0.5%)

---

## 참고 문헌

1. Holte, R. C. (1993). "Very simple classification rules perform well on most commonly used datasets". Machine Learning, 11, 63-91.
2. Mitchell, T. M. (1997). Machine Learning. McGraw-Hill.
3. Quinlan, J. R. (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann.
4. Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32.
5. Cortes, C., & Vapnik, V. (1995). "Support-vector networks". Machine Learning, 20(3), 273-297.

---

**Last Updated:** 2025-11-04
**Author:** Jeong Won Ryeol
**Course:** Data Mining (Fall 2025)
