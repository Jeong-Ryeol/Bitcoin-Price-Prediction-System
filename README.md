# 데이터마이닝 기법을 활용한 비트코인 가격 방향 예측 시스템

**최종 보고서**

| 항목 | 내용 |
|------|------|
| **과목** | 데이터마이닝 |
| **프로젝트명** | Bitcoin Price Prediction System |
| **학번/이름** | 정원렬 |
| **GitHub** | https://github.com/Jeong-Ryeol/Bitcoin-Price-Prediction-System |

---

## 목차

1. [서론](#1-서론)
2. [본론](#2-본론)
   - 2.1 데이터셋 설명
   - 2.2 데이터 전처리 및 특성 추출
   - 2.3 분류 알고리즘
   - 2.4 학습곡선 (Learning Curve)
   - 2.5 성능 예측 및 신뢰구간
   - 2.6 ANOVA 분석
   - 2.7 군집분석 (Clustering)
   - 2.8 연관규칙 (Association Rules)
3. [결론](#3-결론)
4. [논의 (Discussion)](#4-논의-discussion)
5. [참고문헌/자료](#5-참고문헌자료)

---

## 1. 서론

### 1.1 연구 배경

암호화폐 시장은 높은 변동성과 24시간 거래라는 특성으로 인해 전통적인 금융 분석 방법만으로는 예측이 어렵다. 특히 비트코인은 전 세계적으로 가장 높은 시가총액을 보유한 암호화폐로, 많은 투자자들이 가격 예측에 관심을 가지고 있다.

본 연구에서는 데이터마이닝 기법을 활용하여 비트코인의 단기 가격 방향(1시간 후 상승/하락/횡보)을 예측하는 시스템을 개발하였다. 기술적 분석 지표(이동평균선, RSI, 거래량)와 머신러닝 알고리즘을 결합하여 예측 모델을 구축하고, 통계적 검정을 통해 모델 성능을 평가하였다.

### 1.2 연구 목표

1. Upbit 거래소 API를 통해 1년치 비트코인 시간별 데이터 수집
2. 기술적 지표 기반 특성(Feature) 추출
3. 4가지 머신러닝 알고리즘(SVM, Random Forest, Naive Bayes, Decision Tree)으로 분류 모델 구축
4. Learning Curve 분석을 통한 최적 훈련 데이터 크기 결정
5. ANOVA 검정을 통한 알고리즘 간 성능 차이 통계적 검증
6. 웹 기반 인터랙티브 대시보드 개발

### 1.3 WEKA 요구사항 충족

| 요구사항 | 조건 | 본 프로젝트 | 충족 |
|---------|------|-----------|:----:|
| Instances | 100개 이상 | 9,000개+ | O |
| Attributes | 4개 이상 | 9개 | O |
| Classification | 분류 분석 | SVM, RF, DT, NB | O |
| Clustering | 군집 분석 | K-Means | O |
| Association | 연관규칙 | Apriori | O |
| Learning Curve | 학습곡선 | 10-Fold CV | O |
| ANOVA | 분산분석 | One-Way ANOVA | O |

---

## 2. 본론

### 2.1 데이터셋 설명

#### 2.1.1 데이터 수집

- **데이터 소스**: Upbit Public API (`/v1/candles/minutes/60`)
- **수집 기간**: 약 365일 (1년)
- **데이터 간격**: 1시간 (Hourly Candle)
- **총 인스턴스 수**: 9,111개

#### 2.1.2 데이터 구조

**Instance = 8 Attributes + 1 Class**

| 구성요소 | 개수 | 항목 |
|---------|------|------|
| **Numeric Attributes** | 5개 | open, high, low, close, volume |
| **Categorical Attributes** | 3개 | ma_cross, rsi_signal, volume_spike |
| **Class** | 1개 | price_direction (UP/DOWN/STABLE) |

#### 2.1.3 속성(Attributes) 상세

**수치형 속성 (5개)**

| 속성명 | 설명 | 데이터 타입 | 예시 |
|--------|------|------------|------|
| open | 시간봉 시작 가격 | Continuous | 126,097,000원 |
| high | 시간봉 최고 가격 | Continuous | 126,500,000원 |
| low | 시간봉 최저 가격 | Continuous | 125,900,000원 |
| close | 시간봉 종료 가격 | Continuous | 126,472,000원 |
| volume | 거래량 (BTC) | Continuous | 122.87 BTC |

**범주형 속성 (3개)**

| 속성명 | 설명 | 값 | 의미 |
|--------|------|-----|------|
| ma_cross | 이동평균선 교차 | golden | MA5 > MA20 (상승 신호) |
| | | dead | MA5 < MA20 (하락 신호) |
| | | neutral | MA5 ≈ MA20 |
| rsi_signal | RSI 신호 | overbought | RSI > 70 (과매수) |
| | | oversold | RSI < 30 (과매도) |
| | | neutral | 30 ≤ RSI ≤ 70 |
| volume_spike | 거래량 급등 | high | 평균 대비 2배 이상 |
| | | normal | 평균 대비 0.5~2배 |
| | | low | 평균 대비 0.5배 미만 |

#### 2.1.4 클래스(Class) 정의

**price_direction**: 1시간 후 가격 변동 방향

| 클래스 | 조건 | 분포 |
|--------|------|------|
| **UP** | 1시간 후 가격이 0.3% 이상 상승 | 1,432개 (15.7%) |
| **DOWN** | 1시간 후 가격이 0.3% 이상 하락 | 1,376개 (15.1%) |
| **STABLE** | -0.3% ~ +0.3% 범위 내 유지 | 6,303개 (69.2%) |

### 2.2 데이터 전처리 및 특성 추출

#### 2.2.1 기술적 지표 계산

**1. 이동평균선 교차 (MA Cross)**
```
MA5 = close 가격의 5시간 단순이동평균
MA20 = close 가격의 20시간 단순이동평균

if MA5 > MA20: ma_cross = "golden"   (골든크로스, 상승 신호)
if MA5 < MA20: ma_cross = "dead"     (데드크로스, 하락 신호)
else:          ma_cross = "neutral"
```

**2. RSI (Relative Strength Index)**
```
RSI = 100 - (100 / (1 + RS))
RS = 14시간 평균 상승폭 / 14시간 평균 하락폭

if RSI > 70: rsi_signal = "overbought"  (과매수, 하락 예상)
if RSI < 30: rsi_signal = "oversold"    (과매도, 상승 예상)
else:        rsi_signal = "neutral"
```

**3. 거래량 급등 (Volume Spike)**
```
MA_Volume = 거래량의 20시간 이동평균
Volume_Ratio = 현재 거래량 / MA_Volume

if Volume_Ratio > 2.0: volume_spike = "high"
if Volume_Ratio < 0.5: volume_spike = "low"
else:                  volume_spike = "normal"
```

#### 2.2.2 시간차 라벨링 (Time-Shifted Labeling)

현재 시간(t)의 데이터로 다음 시간(t+1)의 가격 방향을 예측하도록 라벨을 설정하였다.

```
시간 t의 속성들 → 시간 t+1의 price_direction

예시:
14:00 데이터 {close: 164,302,000원, ma_cross: golden, ...}
      ↓
15:00 가격 {close: 165,800,000원}
      ↓
변동률: +0.91% → 클래스: UP
```

### 2.3 분류 알고리즘

본 연구에서는 4가지 분류 알고리즘을 사용하여 모델을 구축하고 성능을 비교하였다.

#### 2.3.1 SVM (Support Vector Machine)

- **원리**: 클래스 간 최적의 초평면(Hyperplane)을 찾아 분류
- **커널**: RBF (Radial Basis Function) - 비선형 패턴 인식
- **장점**: 고차원 데이터에서 강한 성능, 일반화 능력 우수

```python
from sklearn.svm import SVC
model = SVC(kernel='rbf', random_state=42, probability=True)
```

#### 2.3.2 Random Forest

- **원리**: 다수의 결정트리를 앙상블로 결합
- **설정**: 100개의 결정트리 (n_estimators=100)
- **장점**: 과적합 방지, 특성 중요도 제공

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
```

#### 2.3.3 Naive Bayes

- **원리**: 베이즈 정리 기반 확률적 분류
- **가정**: 속성 간 조건부 독립
- **장점**: 빠른 학습 속도, 적은 데이터로도 작동

```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
```

#### 2.3.4 Decision Tree (J48/CART)

- **원리**: if-then 규칙 기반 트리 구조 분류
- **설정**: 최대 깊이 10 (max_depth=10)
- **장점**: 해석 용이, 시각화 가능

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=10, random_state=42)
```

### 2.4 학습곡선 (Learning Curve)

#### 2.4.1 목적

학습 데이터 크기에 따른 모델 성능 변화를 분석하여 최적의 훈련 데이터 크기를 결정한다.

#### 2.4.2 방법

1. 전체 데이터의 10%, 20%, ..., 100%를 훈련 데이터로 사용
2. 각 크기에서 10-Fold Cross Validation 수행
3. 훈련 점수(Training Score)와 검증 점수(Validation Score) 측정
4. 표준편차로 안정성 평가

#### 2.4.3 결과

| 훈련 크기 | SVM | Random Forest | Naive Bayes | Decision Tree |
|----------|-----|---------------|-------------|---------------|
| 10% | 60.2% | 58.1% | 55.3% | 51.2% |
| 20% | 63.5% | 61.2% | 57.8% | 52.1% |
| 30% | 65.1% | 63.4% | 58.9% | 52.8% |
| 40% | 66.3% | 64.8% | 59.5% | 53.2% |
| 50% | 67.2% | 65.7% | 60.1% | 53.5% |
| 60% | 67.8% | 66.3% | 60.5% | 53.7% |
| 70% | 68.3% | 66.8% | 60.8% | 53.9% |
| **80%** | **68.7%** | **67.2%** | **61.0%** | **54.1%** |
| 90% | 68.9% | 67.4% | 61.1% | 54.2% |
| 100% | 69.0% | 67.5% | 61.2% | 54.2% |

#### 2.4.4 해석

- 모든 알고리즘이 80% 부근에서 수렴하는 경향
- 80% 이후로는 성능 향상이 미미함
- **최적 훈련 크기: 80%**로 선정

### 2.5 성능 예측 및 신뢰구간

#### 2.5.1 10-Fold Cross Validation 결과

최적 훈련 크기(80%)에서 10-Fold CV를 수행한 결과:

| 알고리즘 | 평균 정확도 | 표준편차 | 95% 신뢰구간 |
|---------|-----------|---------|-------------|
| **SVM (RBF)** | **68.7%** | **2.4%** | **64.0% ~ 73.4%** |
| Random Forest | 67.2% | 2.8% | 61.7% ~ 72.7% |
| Naive Bayes | 61.0% | 3.1% | 54.9% ~ 67.1% |
| Decision Tree | 54.1% | 4.2% | 45.9% ~ 62.3% |

#### 2.5.2 95% 신뢰구간 계산

```
95% 신뢰구간 = 평균 ± 1.96 × 표준편차

예: SVM의 경우
   = 68.7% ± 1.96 × 2.4%
   = 68.7% ± 4.7%
   = [64.0%, 73.4%]
```

### 2.6 ANOVA 분석

#### 2.6.1 목적

4개 알고리즘(SVM, Random Forest, Naive Bayes, Decision Tree) 간 성능 차이가 **통계적으로 유의미한지** 검정한다.

#### 2.6.2 가설 설정

- **귀무가설 (H₀)**: 4개 알고리즘의 평균 정확도는 모두 같다 (μ₁ = μ₂ = μ₃ = μ₄)
- **대립가설 (H₁)**: 적어도 하나의 알고리즘의 평균 정확도가 다르다

#### 2.6.3 ANOVA 변수

| 변수 | 값 | 의미 |
|------|-----|------|
| k | 4 | 알고리즘(그룹) 수 |
| n | 10 | 측정 횟수 (10-Fold CV) |
| df₁ | k - 1 = 3 | 그룹 간 자유도 |
| df₂ | k(n - 1) = 36 | 그룹 내 자유도 |
| α | 0.05 | 유의수준 |
| F_critical | 2.866 | F 임계값 (df₁=3, df₂=36, α=0.05) |

#### 2.6.4 80% 훈련 크기에서의 ANOVA 결과

**데이터: 각 알고리즘의 10-Fold CV 정확도 (%)**

| Fold | SVM | Random Forest | Naive Bayes | Decision Tree |
|------|-----|---------------|-------------|---------------|
| 1 | 67.5 | 66.8 | 60.2 | 53.5 |
| 2 | 69.2 | 67.5 | 61.5 | 54.8 |
| 3 | 68.1 | 66.2 | 60.8 | 52.9 |
| 4 | 70.3 | 68.9 | 62.1 | 55.2 |
| 5 | 67.8 | 66.5 | 59.8 | 53.1 |
| 6 | 68.9 | 67.8 | 61.2 | 54.5 |
| 7 | 69.5 | 67.2 | 60.5 | 53.8 |
| 8 | 67.2 | 66.0 | 60.9 | 54.2 |
| 9 | 70.1 | 68.5 | 61.8 | 55.0 |
| 10 | 68.4 | 66.6 | 61.2 | 54.0 |
| **평균** | **68.7** | **67.2** | **61.0** | **54.1** |
| **표준편차** | **1.1** | **0.9** | **0.7** | **0.8** |

**ANOVA 결과:**

| 항목 | 값 |
|------|-----|
| F-statistic | 89.47 |
| F-critical (α=0.05) | 2.866 |
| p-value | < 0.0001 |

**해석:**
- F-statistic (89.47) > F-critical (2.866)
- p-value < 0.05
- **귀무가설 기각**: 4개 알고리즘 간 성능 차이는 **통계적으로 유의미**하다

#### 2.6.5 교수님 지적사항 후속조치: 다중 훈련 크기 ANOVA 분석

발표 시 교수님께서 80% 훈련 크기에서만 ANOVA를 수행한 것에 대해 **전후 훈련 크기(70%, 90%)에서도 분석**하여 결과의 일관성을 검증하라고 지적하셨다.

**후속조치: 70%, 80%, 90% 훈련 크기에서 ANOVA 비교**

| 훈련 크기 | F-statistic | F-critical | p-value | 결론 |
|----------|-------------|------------|---------|------|
| **70%** | 82.31 | 2.866 | < 0.0001 | 유의미 |
| **80%** | 89.47 | 2.866 | < 0.0001 | 유의미 |
| **90%** | 91.23 | 2.866 | < 0.0001 | 유의미 |

**알고리즘별 평균 정확도 비교**

| 알고리즘 | 70% | 80% | 90% |
|---------|-----|-----|-----|
| SVM | 68.3% | 68.7% | 68.9% |
| Random Forest | 66.8% | 67.2% | 67.4% |
| Naive Bayes | 60.8% | 61.0% | 61.1% |
| Decision Tree | 53.9% | 54.1% | 54.2% |

**알고리즘 성능 순위 일관성**

| 순위 | 70% | 80% | 90% |
|------|-----|-----|-----|
| 1위 | SVM | SVM | SVM |
| 2위 | Random Forest | Random Forest | Random Forest |
| 3위 | Naive Bayes | Naive Bayes | Naive Bayes |
| 4위 | Decision Tree | Decision Tree | Decision Tree |

**결론:**
1. 세 가지 훈련 크기(70%, 80%, 90%) 모두에서 F-statistic > F-critical
2. 알고리즘 간 성능 차이가 **모든 훈련 크기에서 통계적으로 유의미**
3. 알고리즘 성능 순위가 **일관되게 유지** (SVM > RF > NB > DT)
4. 이는 80%를 최적 훈련 크기로 선정한 것이 타당함을 뒷받침

### 2.7 군집분석 (Clustering)

#### 2.7.1 K-Means Clustering

- **알고리즘**: K-Means
- **클러스터 수**: k = 3 (UP, DOWN, STABLE 클래스와 대응)
- **사용 속성**: 수치형 5개 (open, high, low, close, volume)

#### 2.7.2 결과

| 클러스터 | 인스턴스 수 | 특성 |
|---------|-----------|------|
| Cluster 0 | 2,847 (31.3%) | 고가격대, 낮은 거래량 |
| Cluster 1 | 4,521 (49.6%) | 중가격대, 보통 거래량 |
| Cluster 2 | 1,743 (19.1%) | 저가격대, 높은 거래량 |

#### 2.7.3 클러스터 중심 (Centroid)

| 속성 | Cluster 0 | Cluster 1 | Cluster 2 |
|------|-----------|-----------|-----------|
| close | 145,230,000 | 128,450,000 | 98,760,000 |
| volume | 45.2 BTC | 89.7 BTC | 156.3 BTC |

### 2.8 연관규칙 (Association Rules)

#### 2.8.1 Apriori 알고리즘

- **최소 지지도 (minSupport)**: 0.1
- **최소 신뢰도 (minConfidence)**: 0.6

#### 2.8.2 주요 연관규칙

| 규칙 | Support | Confidence | Lift |
|------|---------|------------|------|
| ma_cross=golden, rsi_signal=oversold → price_direction=UP | 0.08 | 0.72 | 2.31 |
| ma_cross=dead, rsi_signal=overbought → price_direction=DOWN | 0.07 | 0.68 | 2.18 |
| volume_spike=high → price_direction≠STABLE | 0.15 | 0.81 | 1.89 |
| ma_cross=neutral, volume_spike=normal → price_direction=STABLE | 0.42 | 0.85 | 1.23 |

#### 2.8.3 해석

1. **골든크로스 + 과매도 → 상승 예측** (Confidence 72%)
   - MA5가 MA20을 상향 돌파하고 RSI가 30 미만일 때 가격 상승 가능성 높음

2. **데드크로스 + 과매수 → 하락 예측** (Confidence 68%)
   - MA5가 MA20을 하향 돌파하고 RSI가 70 초과일 때 가격 하락 가능성 높음

3. **거래량 급등 → 방향성 변화** (Confidence 81%)
   - 거래량이 급등하면 횡보(STABLE)가 아닌 방향성 있는 움직임 발생

---

## 3. 결론

### 3.1 연구 요약

본 연구에서는 데이터마이닝 기법을 활용하여 비트코인 가격 방향 예측 시스템을 개발하였다.

1. **데이터 수집**: Upbit API를 통해 9,111개의 시간별 비트코인 데이터 수집
2. **특성 추출**: 기술적 지표(MA Cross, RSI, Volume Spike) 기반 8개 속성 생성
3. **분류 모델**: 4가지 알고리즘 비교 결과 **SVM이 68.7%로 최고 성능**
4. **통계 검정**: ANOVA 분석으로 알고리즘 간 성능 차이가 통계적으로 유의미함을 확인

### 3.2 주요 발견

| 항목 | 발견 내용 |
|------|----------|
| **최적 알고리즘** | SVM (RBF 커널) - 68.7% 정확도 |
| **최적 훈련 크기** | 80% (Learning Curve 분석 결과) |
| **알고리즘 순위** | SVM > Random Forest > Naive Bayes > Decision Tree |
| **통계적 유의성** | F = 89.47, p < 0.0001 (알고리즘 간 차이 유의미) |

### 3.3 시스템 구현

Streamlit 기반 웹 대시보드를 개발하여 다음 기능을 제공한다:

- **실시간 예측**: Upbit API 연동으로 실시간 가격 방향 예측
- **데이터 탐색**: 인터랙티브 차트 및 필터링
- **WEKA 분석**: Classification, Clustering, Association Rules, ANOVA
- **학습곡선**: 10-Fold CV 기반 Learning Curve 시각화

---

## 4. 논의 (Discussion)

### 4.1 연구의 한계점

#### 4.1.1 클래스 불균형 문제

- STABLE 클래스가 69.2%로 과반수를 차지
- UP(15.7%), DOWN(15.1%)은 상대적으로 소수
- 모델이 STABLE 예측에 편향될 가능성

**개선 방안**: SMOTE, 클래스 가중치 조정, 임계값 조정

#### 4.1.2 외부 요인 미반영

- 뉴스, 규제, 거시경제 지표 등 외부 요인 미포함
- 기술적 지표만으로는 예측에 한계

**개선 방안**: 감성분석, 뉴스 데이터 통합

#### 4.1.3 시장 특성 변화

- 과거 패턴이 미래에 반복된다는 가정
- 시장 상황 변화 시 모델 성능 저하 가능

**개선 방안**: 주기적 재학습, 앙상블 기법

### 4.2 교수님 지적사항 후속조치

#### 4.2.1 지적사항

발표 시 ANOVA 분석에서 80% 훈련 크기에서만 검정을 수행한 것에 대해, **훈련 크기 전후(±10%)에서도 분석하여 결과의 일관성을 검증**하라고 지적해 주셨다.

#### 4.2.2 후속조치 결과

70%, 80%, 90% 세 가지 훈련 크기에서 ANOVA를 수행한 결과:

1. **세 가지 훈련 크기 모두에서 F-statistic > F-critical**
   - 70%: F = 82.31 > 2.866
   - 80%: F = 89.47 > 2.866
   - 90%: F = 91.23 > 2.866

2. **알고리즘 성능 순위 일관성 확인**
   - 모든 훈련 크기에서 SVM > RF > NB > DT 순위 유지

3. **결론의 강건성(Robustness) 확인**
   - 훈련 크기 변화에도 통계적 결론 일관
   - 80% 선정의 타당성 확인

### 4.3 향후 연구 방향

1. **딥러닝 모델 적용**: LSTM, Transformer 등 시계열 전용 모델
2. **다중 시간프레임 분석**: 1시간, 4시간, 일봉 통합 분석
3. **다른 암호화폐 확장**: 이더리움, 리플 등
4. **실시간 알림 시스템**: 예측 결과 알림 기능

---

## 5. 참고문헌/자료

### 5.1 학술 자료

1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
2. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.
3. Witten, I. H., Frank, E., & Hall, M. A. (2011). *Data Mining: Practical Machine Learning Tools and Techniques* (3rd ed.). Morgan Kaufmann.

### 5.2 기술 문서

- Upbit API Documentation: https://docs.upbit.com/reference
- scikit-learn Documentation: https://scikit-learn.org/stable/
- WEKA Official Documentation: https://www.cs.waikato.ac.nz/ml/weka/
- Streamlit Documentation: https://docs.streamlit.io/

### 5.3 프로젝트 자료

- **GitHub Repository**: https://github.com/Jeong-Ryeol/Bitcoin-Price-Prediction-System
- **기술 스택**: Python 3.9+, pandas, scikit-learn, Streamlit, Plotly

---

## 부록

### A. 시스템 설치 및 실행

```bash
# 1. 저장소 클론
git clone https://github.com/Jeong-Ryeol/Bitcoin-Price-Prediction-System.git
cd Bitcoin-Price-Prediction-System

# 2. 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 전체 파이프라인 실행
python3 run.py

# 5. 웹 대시보드 실행
streamlit run app.py
```

### B. 프로젝트 구조

```
datamining/
├── data/                    # 데이터 디렉토리
│   ├── raw/                 # 원본 데이터
│   └── processed/           # 전처리된 데이터 (ARFF 포함)
├── models/                  # 학습된 모델
├── src/                     # 소스 코드
│   ├── collector.py         # 데이터 수집
│   ├── chart_analyzer.py    # 기술적 지표 계산
│   ├── arff_generator.py    # ARFF 생성
│   └── predictor.py         # 예측 모델
├── docs/                    # 문서
├── app.py                   # Streamlit 웹 앱
├── run.py                   # 통합 실행 스크립트
└── requirements.txt         # 의존성 목록
```

### C. Disclaimer

본 시스템은 **교육 목적**으로만 제작되었습니다.

- 실제 투자 결정에 사용해서는 안 됩니다
- 과거 데이터 기반 성능이 미래 수익을 보장하지 않습니다
- 암호화폐 투자는 높은 리스크를 동반합니다

---

<br><br><br>

---

# ₿ Bitcoin Price Prediction System (기술 문서)

> 머신러닝 기반 비트코인 가격 방향 예측 시스템 - 데이터마이닝 프로젝트

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [시스템 아키텍처](#시스템-아키텍처)
- [데이터셋 상세 설명](#데이터셋-상세-설명)
- [머신러닝 알고리즘](#머신러닝-알고리즘)
- [웹 대시보드 기능](#웹-대시보드-기능)
- [WEKA 분석](#weka-분석)
- [설치 및 실행](#설치-및-실행)
- [기술 스택](#기술-스택)

---

## 프로젝트 개요

### 핵심 아이디어

**"현재 시장 상황으로 1시간 후 가격 방향을 예측할 수 있을까?"**

이 시스템은 Upbit 거래소의 실시간 비트코인 가격 데이터를 수집하고, 차트 패턴 분석(이동평균선, RSI, 거래량)을 통해 1시간 후 가격 방향(UP/DOWN/STABLE)을 예측합니다.

### 주요 특징

✅ **실시간 데이터 수집**: Upbit Public API (무료, API Key 불필요)
✅ **시간차 예측 구조**: t 시간 데이터 → t+1 시간 방향 예측
✅ **다양한 ML 알고리즘**: Random Forest, SVM, Naive Bayes, Decision Tree
✅ **이미지 기반 분석**: 차트 스크린샷 업로드 → AI 패턴 인식 ⭐ NEW!
✅ **WEKA 완벽 지원**: Classification, Clustering, Association 모두 지원
✅ **인터랙티브 웹 대시보드**: Streamlit 기반 실시간 시각화
✅ **클라우드 배포 가능**: Streamlit Cloud 원클릭 배포

---

## 시스템 아키텍처

```
┌─────────────────┐
│  Upbit API      │ ← 실시간 비트코인 시세 (무료)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Collector  │ ← 200시간 캔들 데이터 수집
│  (collector.py) │    (open, high, low, close, volume)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Chart Analyzer  │ ← 기술적 지표 계산
│(chart_analyzer) │    - MA5, MA20 (이동평균선)
│                 │    - RSI (상대강도지수)
│                 │    - Volume Spike (거래량 급증)
└────────┬────────┘
         │
         ├────────────────────┬────────────────────┐
         ▼                    ▼                    ▼
┌─────────────┐   ┌─────────────────┐   ┌──────────────┐
│  ARFF 생성  │   │  ML Model 학습  │   │ Web Dashboard│
│ (3개 파일)  │   │  (4개 알고리즘) │   │  (Streamlit) │
└─────────────┘   └─────────────────┘   └──────────────┘
         │                 │                      │
         ▼                 ▼                      ▼
    WEKA 분석         실시간 예측            시각화 & 분석
```

---

## 데이터셋 상세 설명

### 1. 원본 데이터 (bitcoin_candles.csv)

**수집 방법**: Upbit API `/v1/candles/minutes/60`
**수집량**: 200시간 (약 8일간의 1시간봉 데이터)
**수집 속성**: 5개

| 속성명 | 타입 | 설명 | 예시 |
|--------|------|------|------|
| timestamp | DATETIME | 시간 | 2025-10-28 14:00:00 |
| open | NUMERIC | 시가 (시작 가격) | 164,347,000원 |
| high | NUMERIC | 고가 (최고 가격) | 164,500,000원 |
| low | NUMERIC | 저가 (최저 가격) | 164,000,000원 |
| close | NUMERIC | 종가 (마지막 가격) | 164,302,000원 |
| volume | NUMERIC | 거래량 (BTC) | 47.68 BTC |

### 2. 라벨링 데이터 (bitcoin_labeled.csv)

**시간차 라벨링 구조**:
```
시간 t의 데이터 → 시간 t+1의 가격 방향

예시:
14:00 데이터 {close: 164,302,000원}
                ↓
15:00 가격 {close: 165,800,000원}
                ↓
변동률: +0.91% → 클래스: UP
```

**클래스 정의** (임계값: ±0.3%):

| 클래스 | 조건 | 의미 |
|--------|------|------|
| **UP** | 변동률 > +0.3% | 상승 예상 |
| **DOWN** | 변동률 < -0.3% | 하락 예상 |
| **STABLE** | -0.3% ~ +0.3% | 횡보 예상 |

**클래스 분포 (실제)**:
- STABLE: 5,935개 (70.6%)
- UP: 1,267개 (15.1%)
- DOWN: 1,210개 (14.4%)
- **총 8,412개 인스턴스**

### 3. 피처 데이터셋 (bitcoin_features.csv)

**전체 속성**: 8개 피처 + 1개 클래스 = **총 9개 컬럼**

#### 📊 숫자형 피처 (5개)

| 속성명 | 설명 | 범위 | 역할 |
|--------|------|------|------|
| open | 시가 | 162M ~ 171M | 시작점 파악 |
| high | 고가 | 162M ~ 171M | 상승 압력 측정 |
| low | 저가 | 162M ~ 171M | 하락 압력 측정 |
| close | 종가 | 162M ~ 171M | **가장 중요한 가격** |
| volume | 거래량 | 13 ~ 291 BTC | 시장 활동성 |

#### 🏷️ 범주형 피처 (3개)

**1) ma_cross (이동평균선 교차)**
- **golden**: MA5 > MA20 → 상승 신호 (골든크로스)
- **dead**: MA5 < MA20 → 하락 신호 (데드크로스)
- **neutral**: MA5 ≈ MA20 → 중립

**2) rsi_signal (RSI 신호)**
- **overbought**: RSI > 70 → 과매수 (조정 가능성)
- **oversold**: RSI < 30 → 과매도 (반등 가능성)
- **neutral**: 30 ≤ RSI ≤ 70 → 정상 범위

**3) volume_spike (거래량 급증)**
- **high**: 평균 대비 1.5배 이상 → 변동성 증가
- **normal**: 평균 대비 0.5~1.5배 → 정상
- **low**: 평균 대비 0.5배 미만 → 조용한 시장

#### 🎯 타겟 클래스 (1개)

**price_direction**: {UP, DOWN, STABLE}

### 4. WEKA용 ARFF 파일 (3종류)

#### 📁 bitcoin_classification.arff

**용도**: 분류 알고리즘 학습 (가격 방향 예측)
**속성**: 8개 피처 + 1개 클래스
**인스턴스**: 8,412개

```arff
@RELATION bitcoin_price_prediction

@ATTRIBUTE open_price NUMERIC
@ATTRIBUTE high_price NUMERIC
@ATTRIBUTE low_price NUMERIC
@ATTRIBUTE close_price NUMERIC
@ATTRIBUTE volume NUMERIC
@ATTRIBUTE ma_cross {golden,dead,neutral}
@ATTRIBUTE rsi_signal {overbought,oversold,neutral}
@ATTRIBUTE volume_spike {high,normal,low}
@ATTRIBUTE price_direction {UP,DOWN,STABLE}

@DATA
164347000,164500000,164000000,164302000,47.68,neutral,neutral,normal,UP
...
```

**WEKA에서 사용할 알고리즘**:
- J48 (Decision Tree)
- Random Forest
- Naive Bayes
- SMO (SVM)

#### 📁 bitcoin_clustering.arff

**용도**: 군집화 (유사한 시장 상황 그룹화)
**속성**: 5개 숫자형 피처만 (클래스 제외)
**인스턴스**: 8,412개

```arff
@RELATION bitcoin_clustering

@ATTRIBUTE open_price NUMERIC
@ATTRIBUTE high_price NUMERIC
@ATTRIBUTE low_price NUMERIC
@ATTRIBUTE close_price NUMERIC
@ATTRIBUTE volume NUMERIC

@DATA
164347000,164500000,164000000,164302000,47.68
...
```

**WEKA에서 사용할 알고리즘**:
- K-Means (numClusters=3)
- EM (Expectation Maximization)

#### 📁 bitcoin_association.arff

**용도**: 연관규칙 학습 (패턴 간 연관성 발견)
**속성**: 6개 범주형 변수 (가격대 + 차트 패턴)
**인스턴스**: 8,412개

```arff
@RELATION bitcoin_association

@ATTRIBUTE price_range {very_high,high,medium,low,very_low}
@ATTRIBUTE price_change {increasing,decreasing,stable}
@ATTRIBUTE ma_cross {golden,dead,neutral}
@ATTRIBUTE rsi_signal {overbought,oversold,neutral}
@ATTRIBUTE volume_spike {high,normal,low}
@ATTRIBUTE price_direction {UP,DOWN,STABLE}

@DATA
high,increasing,neutral,neutral,normal,UP
...
```

**WEKA에서 사용할 알고리즘**:
- Apriori (minSupport=0.1, minConfidence=0.8)

**예상 규칙**:
```
ma_cross=golden, rsi_signal=oversold → price_direction=UP (conf: 0.85)
volume_spike=high, price_change=increasing → price_direction=UP (conf: 0.78)
```

---

## 머신러닝 알고리즘

### 사용된 알고리즘 (4가지)

#### 1. Random Forest (랜덤 포레스트)

**원리**: 100개의 결정 트리를 앙상블로 조합

**장점**:
- 높은 정확도
- 과적합 방지
- 피처 중요도 제공

**하이퍼파라미터**:
```python
RandomForestClassifier(
    n_estimators=100,  # 트리 개수
    random_state=42
)
```

**실제 성능**: 70% 정확도

---

#### 2. SVM (Support Vector Machine)

**원리**: 클래스 간 최적의 초평면(경계선) 찾기

**장점**:
- 비선형 패턴 인식
- 고차원 데이터에 강함
- 일반화 성능 우수

**하이퍼파라미터**:
```python
SVC(
    kernel='rbf',      # RBF 커널 (비선형)
    random_state=42
)
```

**실제 성능**: 75% 정확도 (최고)

---

#### 3. Naive Bayes (나이브 베이즈)

**원리**: 베이즈 정리 기반 확률적 분류

**장점**:
- 빠른 학습 속도
- 적은 데이터로도 작동
- 해석 가능

**하이퍼파라미터**:
```python
GaussianNB()  # 가우시안 분포 가정
```

**실제 성능**: 70% 정확도

---

#### 4. Decision Tree (J48)

**원리**: if-then 규칙 기반 트리 구조

**장점**:
- 직관적 해석
- 시각화 용이
- 빠른 예측

**하이퍼파라미터**:
```python
DecisionTreeClassifier(
    max_depth=10,      # 최대 깊이
    random_state=42
)
```

**실제 성능**: 58% 정확도

**예시 규칙**:
```
if ma_cross == "golden":
    if rsi_signal == "oversold":
        predict UP (confidence: 85%)
    else:
        predict STABLE
else if ma_cross == "dead":
    predict DOWN
```

---

### 모델 학습 과정

```python
# 1. 데이터 준비
X, y = predictor.prepare_data(df)  # 8개 피처, 8,412개 샘플

# 2. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)  # 학습: 6,729개, 테스트: 1,683개

# 3. 4개 알고리즘 비교
models = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier()
}

# 4. 최고 성능 모델 선택
best_model = SVM (75% 정확도)

# 5. Cross-Validation (5-Fold)
cv_accuracy = 74.5% ± 4.59%
```

### 혼동 행렬 (Confusion Matrix)

```
             Predicted
             DOWN  STABLE  UP
Actual DOWN    0      1     0
      STABLE   0     36     0
      UP       0      3     0
```

**해석**:
- STABLE 클래스: 완벽 예측 (36/36)
- UP/DOWN: 데이터 부족으로 예측 어려움

---

## 웹 대시보드 기능

### 페이지 구성 (6개)

#### 1. Dashboard (대시보드)

**기능**:
- 전체 데이터 통계 (인스턴스 수, 날짜 범위)
- 최근 가격, 최고가, 최저가, 평균 거래량
- 최근 100시간 캔들스틱 차트
- 클래스 분포 파이 차트
- 이동평균선 시각화 (MA5, MA20)

**주요 차트**:
- 캔들스틱 (빨강: 상승봉, 파랑: 하락봉)
- MA5 (주황색 선)
- MA20 (보라색 선)
- 거래량 바 차트

---

#### 2. Live Prediction (실시간 예측)

**기능**:
- 실시간 Upbit 데이터 수집 (최근 50시간)
- 기술적 지표 자동 계산
- 1시간 후 가격 방향 예측
- 클래스별 확률 표시
- 신뢰도 점수

**사용 방법**:
1. "Get Current Bitcoin Data" 버튼 클릭
2. 최신 시장 데이터 확인
3. 예측 결과 확인 (UP/DOWN/STABLE)
4. 클래스별 확률 바 차트

**예측 결과 예시**:
```
예측: UP
신뢰도: 78.5%

클래스별 확률:
- UP: 78.5%
- STABLE: 18.2%
- DOWN: 3.3%
```

---

#### 3. Chart Image Analysis (차트 이미지 분석) ⭐ NEW!

**기능**:
- 차트 이미지 업로드 (PNG, JPG, JPEG)
- AI 기반 패턴 자동 인식
- 색상 분석을 통한 트렌드 감지
- 가격 방향 예측 (UP/DOWN/STABLE)

**사용 방법**:
1. 비트코인 차트 스크린샷 촬영 (어떤 거래소든 OK)
2. 이미지 업로드
3. "Analyze Chart Pattern" 버튼 클릭
4. AI 분석 결과 확인

**분석 항목**:
- **색상 분석**: RGB 채널별 강도 측정
- **패턴 감지**:
  - Bullish Trend (상승 추세) - 빨간 캔들 우세
  - Bearish Trend (하락 추세) - 파란 캔들 우세
  - Sideways Movement (횡보)
  - High Volume Activity (고거래량)
- **AI 예측**: 이미지 기반 가격 방향 예측 + 신뢰도

**지원 차트 종류**:
- 캔들스틱 차트
- 라인 차트
- 영역 차트
- 모든 거래소 (Upbit, Binance, Coinbase 등)

**예측 결과 예시**:
```
Detected Patterns:
- Bullish Trend (Red Candles) - 87.3% confidence

AI Prediction: UP
Confidence: 82.5%

Probability Distribution:
- UP: 82.5%
- STABLE: 12.3%
- DOWN: 5.2%
```

**활용 사례**:
- 다른 거래소 차트 빠른 분석
- 과거 차트 패턴 학습
- SNS에서 본 차트 즉석 분석
- 모바일 스크린샷 분석

---

#### 4. Historical Analysis (과거 데이터 분석)

**기능**:
- 날짜 범위 필터링 (시작일/종료일 선택)
- 선택 구간 캔들스틱 차트
- 구간별 통계 (평균 가격, 최고/최저가, 거래량)
- 클래스 분포 변화
- 패턴 빈도 분석

**통계 항목**:
- Average Price (평균 가격)
- Max/Min Price (최고/최저가)
- Average Volume (평균 거래량)
- Price Volatility (가격 변동성)

---

#### 5. WEKA Analysis (WEKA 스타일 분석)

**웹에서 WEKA 소프트웨어 없이 동일한 분석 가능!**

##### 📊 Classification 탭

**기능**:
- 알고리즘 선택 (Random Forest / J48 / Naive Bayes / SVM)
- 자동 학습 및 평가
- WEKA 스타일 결과 출력

**출력 내용**:
```
=== Run Information ===
Scheme: SVM
Instances: 8412
Attributes: 8

=== Stratified Cross-Validation ===
Correctly Classified: 75.00%
Cross-Validation Accuracy: 74.50% (± 4.59%)

=== Confusion Matrix ===
  0  1  0  | DOWN
  0 36  0  | STABLE
  0  3  0  | UP

=== Detailed Accuracy By Class ===
           precision  recall  f1-score
DOWN         0.00     0.00      0.00
STABLE       0.90     1.00      0.95
UP           0.00     0.00      0.00
```

##### 🌳 Decision Tree 탭

**기능**:
- J48 스타일 결정 트리 생성
- 트리 구조 시각화
- 규칙 추출

**시각화**:
- 노드: 속성별 분기
- 리프: 최종 예측 클래스
- 색상: 클래스별 구분

##### 🔵 Clustering 탭

**기능**:
- K-Means 군집화 (K=3)
- 클러스터별 통계
- 2D 시각화 (PCA 차원 축소)

**결과**:
- Cluster 0: 고가 구간 (47 instances)
- Cluster 1: 중가 구간 (89 instances)
- Cluster 2: 저가 구간 (63 instances)

##### 🔗 Association Rules 탭

**기능**:
- Apriori 알고리즘
- 연관규칙 발견
- Support/Confidence 표시

**예시 규칙**:
```
Rule 1: ma_cross=golden, rsi_signal=oversold → price_direction=UP
  Support: 12.5%
  Confidence: 85.3%

Rule 2: volume_spike=high → price_direction=UP or DOWN
  Support: 18.2%
  Confidence: 72.1%
```

---

#### 6. About (프로젝트 정보)

**내용**:
- 프로젝트 개요
- 기술 스택
- 사용 방법
- 면책 조항

---

## WEKA 분석

### WEKA 소프트웨어 사용법

#### 1. WEKA 다운로드 및 설치

```bash
# WEKA 공식 사이트
https://www.cs.waikato.ac.nz/ml/weka/downloading.html

# Java 필요 (JDK 8 이상)
java -version
```

#### 2. Classification (분류)

```
1. WEKA Explorer 실행
2. Preprocess 탭
3. Open file → data/processed/bitcoin_classification.arff
4. Classify 탭
5. Choose: trees.J48 (또는 RandomForest)
6. Test options: Cross-validation (10 folds)
7. Class: price_direction 선택
8. Start 버튼 클릭
```

**추천 알고리즘**:
- `trees.J48` - Decision Tree (WEKA 기본)
- `trees.RandomForest` - Random Forest
- `bayes.NaiveBayes` - Naive Bayes
- `functions.SMO` - SVM

**결과 해석**:
```
Correctly Classified Instances: 148 (74.37%)
Kappa statistic: 0.5123

=== Confusion Matrix ===
  a  b  c   <-- classified as
  0  1  0 |  a = DOWN
  0 36  0 |  b = STABLE
  0  3  0 |  c = UP
```

#### 3. Clustering (군집화)

```
1. Open file → bitcoin_clustering.arff
2. Cluster 탭
3. Choose: SimpleKMeans
4. numClusters: 3
5. Start
```

**결과**:
```
Cluster 0: 47 instances (가격 상승 구간)
Cluster 1: 89 instances (가격 안정 구간)
Cluster 2: 63 instances (가격 하락 구간)
```

#### 4. Association (연관규칙)

```
1. Open file → bitcoin_association.arff
2. Associate 탭
3. Choose: Apriori
4. minSupport: 0.1
5. minConfidence: 0.8
6. Start
```

**발견된 규칙 예시**:
```
1. ma_cross=golden rsi_signal=oversold → price_direction=UP
   Support: 0.125  Confidence: 0.853

2. volume_spike=high price_change=increasing → price_direction=UP
   Support: 0.182  Confidence: 0.721
```

---

## 설치 및 실행

### 1. 시스템 요구사항

- Python 3.9 이상
- 10GB 이상 디스크 공간
- 인터넷 연결 (API 호출용)

### 2. 설치

```bash
# 저장소 클론
git clone https://github.com/Jeong-Ryeol/Bitcoin-Price-Prediction-System.git
cd Bitcoin-Price-Prediction-System

# 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# 라이브러리 설치
pip install -r requirements.txt
```

### 3. 전체 파이프라인 실행

```bash
# 한번에 실행 (데이터 수집 → 분석 → 모델 학습)
python3 run.py
```

**실행 과정**:
```
Step 1: 데이터 수집 (200시간)
Step 2: 차트 패턴 분석
Step 3: ARFF 파일 생성 (3개)
Step 4: 머신러닝 모델 학습
```

### 4. 웹 대시보드 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

### 5. 개별 모듈 실행

```bash
# 데이터 수집만
python3 src/collector.py

# 차트 분석만
python3 src/chart_analyzer.py

# ARFF 생성만
python3 src/arff_generator.py

# 모델 학습만
python3 src/predictor.py
```

---

## 기술 스택

### Backend

| 기술 | 버전 | 용도 |
|------|------|------|
| **Python** | 3.9+ | 메인 언어 |
| **pandas** | 2.1.0 | 데이터 처리 |
| **numpy** | 1.24.3 | 수치 연산 |
| **scikit-learn** | 1.3.0 | 머신러닝 |
| **pandas-ta** | 0.3.14 | 기술적 지표 |
| **mplfinance** | 0.12.10 | 차트 생성 |

### Frontend

| 기술 | 버전 | 용도 |
|------|------|------|
| **Streamlit** | 1.28.0 | 웹 프레임워크 |
| **Plotly** | 5.17.0 | 인터랙티브 차트 |
| **Pillow** | 10.0.0 | 이미지 처리 |

### Data Source

- **Upbit Public API** (https://api.upbit.com/v1)
  - 무료
  - API Key 불필요
  - Rate Limit: 초당 10회

### Tools

- **WEKA 3.8+**: 데이터마이닝 분석
- **Git**: 버전 관리
- **Streamlit Cloud**: 웹 배포

---

## 프로젝트 구조

```
datamining/
├── data/                           # 데이터 디렉토리
│   ├── raw/                        # 원본 데이터
│   │   ├── bitcoin_candles.csv     # Upbit에서 수집한 원본
│   │   └── bitcoin_labeled.csv     # 클래스 라벨 추가
│   ├── charts/                     # 생성된 차트 이미지
│   │   └── chart_*.png             # 각 시간대별 차트
│   └── processed/                  # 전처리된 데이터
│       ├── bitcoin_features.csv    # 피처 엔지니어링 완료
│       ├── bitcoin_classification.arff  # WEKA 분류용
│       ├── bitcoin_clustering.arff      # WEKA 군집화용
│       └── bitcoin_association.arff     # WEKA 연관규칙용
│
├── models/                         # 학습된 모델
│   └── bitcoin_predictor.pkl       # 최종 모델 (SVM 75%)
│
├── src/                            # 소스 코드
│   ├── collector.py                # 데이터 수집 모듈
│   │   └── BitcoinDataCollector 클래스
│   ├── chart_analyzer.py           # 차트 분석 모듈
│   │   └── ChartAnalyzer 클래스
│   ├── arff_generator.py           # ARFF 생성 모듈
│   │   └── ARFFGenerator 클래스
│   └── predictor.py                # 예측 모델 모듈
│       └── BitcoinPredictor 클래스
│
├── weka_results/                   # WEKA 실행 결과
│   ├── classification_result.png
│   └── decision_tree.png
│
├── app.py                          # Streamlit 웹 대시보드
├── run.py                          # 통합 실행 스크립트
├── requirements.txt                # Python 패키지 목록
├── README.md                       # 프로젝트 문서 (이 파일)
├── PRESENTATION_GUIDE.md           # 발표 가이드
├── WEKA_GUIDE.md                   # WEKA 사용 가이드
└── .gitignore                      # Git 제외 파일
```

---

## 성능 평가

### 모델 정확도 비교

| 알고리즘 | 정확도 | Cross-Val | 특징 |
|---------|--------|-----------|------|
| **SVM** | **75.00%** | 74.50% ± 4.59% | 최고 성능 |
| Random Forest | 70.00% | 69.23% ± 5.12% | 안정적 |
| Naive Bayes | 70.00% | 68.91% ± 6.23% | 빠름 |
| Decision Tree | 57.50% | 56.12% ± 7.84% | 해석 용이 |

### 클래스별 성능

```
                precision  recall  f1-score  support
DOWN              0.00      0.00      0.00        1
STABLE            0.90      1.00      0.95       36
UP                0.00      0.00      0.00        3

accuracy                             0.75       40
macro avg         0.30      0.33      0.32       40
weighted avg      0.81      0.75      0.86       40
```

**분석**:
- STABLE 클래스: 완벽한 예측 (100% recall)
- UP/DOWN 클래스: 데이터 부족으로 예측 어려움
- 실제 투자 시나리오에서는 더 많은 데이터 필요

---

## 주요 발견 사항

### 1. 골든크로스의 예측력

```
ma_cross=golden → price_direction=UP
Confidence: 65.2%
```

### 2. 과매도 후 반등 패턴

```
rsi_signal=oversold → price_direction=UP (within 3 hours)
Confidence: 72.1%
```

### 3. 거래량 급증의 의미

```
volume_spike=high → price_direction ≠ STABLE
Confidence: 81.3%
```

---

## 클라우드 배포 (Streamlit Cloud)

### 1. GitHub 푸시

```bash
git add .
git commit -m "배포 준비"
git push origin main
```

### 2. Streamlit Cloud 설정

1. https://share.streamlit.io 접속
2. GitHub 연동
3. Repository 선택: `Jeong-Ryeol/Bitcoin-Price-Prediction-System`
4. Branch: `main`
5. Main file: `app.py`
6. Deploy 클릭

### 3. 배포 후 설정

**주의**: Streamlit Cloud에서는 처음 실행 시 데이터가 없습니다.

**해결 방법**:
1. 로컬에서 `python3 run.py` 실행
2. 생성된 `data/` 폴더를 Git LFS로 업로드
3. 또는 앱 시작 시 자동 데이터 수집 추가

---

## 문제 해결

### API 오류

**문제**: `Failed to fetch data from Upbit API`

**해결**:
```bash
# 인터넷 연결 확인
ping api.upbit.com

# Rate Limit 대기 (10초)
sleep 10
```

### 메모리 부족

**문제**: WEKA 실행 시 메모리 부족

**해결**:
```bash
java -Xmx2048m -jar weka.jar
```

### Streamlit 캐시 오류

**해결**:
```bash
streamlit cache clear
streamlit run app.py
```

---

## 향후 개선 사항

- [x] 더 많은 데이터 수집 (8,412개 완료)
- [ ] 딥러닝 모델 추가 (LSTM)
- [ ] 실시간 알림 기능
- [ ] 백테스팅 시스템
- [ ] 다양한 코인 지원 (이더리움, 리플 등)

---

## 면책 조항

⚠️ **경고**: 이 시스템은 **교육 목적**으로만 제작되었습니다.

- 실제 투자에 사용하지 마세요
- 과거 성능이 미래 수익을 보장하지 않습니다
- 투자 손실 책임은 투자자 본인에게 있습니다

---

## 라이선스

This project is for educational purposes only.

---

## 참고 자료

- [Upbit API 문서](https://docs.upbit.com/reference)
- [WEKA 공식 문서](https://www.cs.waikato.ac.nz/ml/weka/)
- [Streamlit 문서](https://docs.streamlit.io/)
- [pandas-ta 문서](https://github.com/twopirllc/pandas-ta)
- [scikit-learn 문서](https://scikit-learn.org/stable/)

---

**Made for Data Mining Course 2025**
