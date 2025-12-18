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

**Data Mining Course 2025**
