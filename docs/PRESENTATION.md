# Bitcoin Price Prediction System

## 데이터마이닝 기법을 활용한 비트코인 가격 방향 예측

---

**Team Project - Data Mining 2025**

**GitHub**: https://github.com/Jeong-Ryeol/Bitcoin-Price-Prediction-System

---

## 목차

1. Project Overview
2. Data Structure
3. Features Guide
4. Data Mining Methods
5. Installation & Academic Purpose

---

# 1. Project Overview

## 프로젝트 개요

비트코인 가격 예측 시스템은 **데이터마이닝 기법**을 활용하여 암호화폐 시장의 가격 방향을 예측하는 머신러닝 프로젝트입니다.

## 프로젝트 목표

- Upbit Public API를 통해 **1년치 시간별 데이터**를 수집
- 기술적 지표를 분석하여 **4가지 머신러닝 알고리즘**으로 가격 방향 예측
- **WEKA 스타일** 데이터마이닝 분석을 웹에서 구현

## Dataset Statistics

| 항목 | 값 |
|------|-----|
| Total Instances | 9,000개+ |
| Total Attributes | 9개 (8 features + 1 class) |
| Data Period | 약 365일 |
| Data Interval | 1시간 (Hourly Candle) |
| Data Source | Upbit API |

## Technology Stack

| Category | Technologies |
|----------|-------------|
| **Data Collection** | Upbit Public API (RESTful) |
| **Data Processing** | pandas, numpy |
| **Technical Analysis** | pandas-ta, RSI, Moving Average |
| **Machine Learning** | scikit-learn (SVM, RF, DT, NB) |
| **Statistical Analysis** | scipy (ANOVA) |
| **Data Mining** | mlxtend (Apriori), K-Means |
| **Visualization** | Streamlit, Plotly |

---

# 2. Data Structure

## Instance = Attribute + Class

| 구성요소 | 개수 | 항목 |
|---------|------|------|
| **Attributes** | 8개 | open, high, low, close, volume, ma_cross, rsi_signal, volume_spike |
| **Class** | 1개 | price_direction |

## Attributes (속성) - 8개

### Numeric Attributes (수치형) - 5개

| Attribute | Description | Type | Example |
|-----------|-------------|------|---------|
| **open** | 시간봉 시작 가격 | Continuous | 126,097,000 원 |
| **high** | 시간봉 최고 가격 | Continuous | 126,500,000 원 |
| **low** | 시간봉 최저 가격 | Continuous | 125,900,000 원 |
| **close** | 시간봉 종료 가격 | Continuous | 126,472,000 원 |
| **volume** | 거래량 (BTC) | Continuous | 122.87 BTC |

### Categorical Attributes (범주형) - 3개

| Attribute | Description | Values |
|-----------|-------------|--------|
| **ma_cross** | 이동평균선 교차 신호 | golden, dead, neutral |
| **rsi_signal** | RSI 과매수/과매도 신호 | overbought, oversold, neutral |
| **volume_spike** | 거래량 급등/급락 신호 | high, normal, low |

## Class (클래스) - 1개

| Class | Description | Threshold |
|-------|-------------|-----------|
| **price_direction** | 1시간 후 가격 방향 | UP/DOWN/STABLE |

**분류 기준:**
- **UP**: 1시간 후 가격이 0.3% 이상 상승
- **DOWN**: 1시간 후 가격이 0.3% 이상 하락
- **STABLE**: -0.3% ~ +0.3% 범위 내 유지

## Technical Indicators 상세

**1. MA Cross (Moving Average Crossover)**
- 단기 이동평균(5시간)과 장기 이동평균(20시간) 비교
- **Golden Cross**: MA5 > MA20 (상승 신호)
- **Dead Cross**: MA5 < MA20 (하락 신호)

**2. RSI Signal (Relative Strength Index)**
- 14시간 기준 상대강도지수
- **Overbought**: RSI > 70 (과매수 → 하락 예상)
- **Oversold**: RSI < 30 (과매도 → 상승 예상)

**3. Volume Spike**
- 20시간 평균 거래량 대비 비율
- **High**: 2배 이상 (급등)
- **Low**: 0.5배 이하 (급락)

---

# 3. Features Guide

## 각 페이지 기능 설명

### 1. Dashboard
**목적**: 데이터셋 전체 개요 및 통계 확인

**주요 기능:**
- 총 인스턴스 수, 속성 수, 데이터 기간 표시
- 비트코인 가격 시계열 차트 (Plotly Interactive)
- 클래스(UP/DOWN/STABLE) 분포 파이 차트
- 기술적 지표별 분포 막대 차트

---

### 2. Live Prediction
**목적**: 실시간 데이터로 즉시 예측

**주요 기능:**
- Upbit API에서 실시간 200시간 데이터 수집
- 기술적 지표 자동 계산
- SVM 모델로 1시간 후 가격 방향 예측
- 클래스별 확률 분포 표시

---

### 3. Manual Prediction
**목적**: 사용자 입력 데이터로 예측

**주요 기능:**
- 개별 입력: OHLCV + 기술적 지표 직접 입력
- 일괄 예측: CSV 파일 업로드 후 일괄 예측
- 예측 결과 다운로드

---

### 4. Dataset Explorer
**목적**: 전체 데이터셋 탐색 및 분석

**주요 기능:**
- 인스턴스 필터링 (클래스별, 날짜별)
- 속성 분포 시각화 (히스토그램, 박스플롯)
- CSV 다운로드

---

### 5. Chart Image Analysis
**목적**: 차트 이미지 분석 (실험적 기능)

**주요 기능:**
- 차트 스크린샷 업로드
- 색상 분석으로 트렌드 감지
- 간단한 패턴 인식

---

### 6. Historical Analysis
**목적**: 특정 기간 데이터 분석

**주요 기능:**
- 날짜 범위 선택
- 캔들스틱 차트 표시
- 기간별 통계 (평균, 최고, 최저 가격)

---

### 7. WEKA Analysis
**목적**: WEKA 스타일 데이터마이닝 분석

| Tab | 기능 | 알고리즘 |
|-----|------|---------|
| **Classification** | 분류 분석 | NB, DT, RF, SVM |
| **Decision Tree** | 의사결정나무 시각화 | J48 (CART) |
| **Clustering** | 군집 분석 | K-Means (k=3) |
| **Association Rules** | 연관규칙 마이닝 | Apriori |
| **ANOVA** | 알고리즘 성능 비교 | One-Way ANOVA |

---

### 8. Model Performance
**목적**: 모델 성능 비교 및 분석

**주요 기능:**
- 4개 알고리즘 정확도 비교
- Learning Curve 시각화 (10-Fold CV)
- 95% 신뢰구간 표시
- 최적 모델 선택 근거 제시

---

# 4. Data Mining Methods

## 분석 방법론

### 1. Classification (분류)

**목적**: 새로운 인스턴스의 클래스를 예측

**사용 알고리즘:**

| Algorithm | Description | 특징 |
|-----------|-------------|------|
| **Naive Bayes** | 베이즈 정리 기반 확률적 분류 | 빠른 학습, 독립성 가정 |
| **Decision Tree (J48)** | 규칙 기반 트리 구조 분류 | 해석 용이, 과적합 위험 |
| **Random Forest** | 다수의 결정트리 앙상블 | 높은 정확도, 과적합 방지 |
| **SVM (RBF)** | 최적 결정 경계 탐색 | 고차원에 강함, 최고 성능 |

**평가 지표:**
- Accuracy (정확도)
- Precision, Recall, F1-Score
- Cross-Validation (10-Fold)

---

### 2. Learning Curve (학습곡선)

**목적**: 학습 데이터 크기에 따른 모델 성능 변화 분석

**방법:**
- 10%, 20%, ..., 100% 학습 데이터로 모델 학습
- 10-Fold Cross Validation으로 성능 측정
- 표준편차로 안정성 평가

**해석:**
- 학습곡선이 수렴하면 충분한 데이터
- 학습/검증 점수 차이가 크면 과적합

---

### 3. Performance Confidence Interval (성능 신뢰구간)

**목적**: 모델 성능의 신뢰 범위 추정

**방법:**
- 10-Fold Cross Validation 수행
- 각 Fold의 정확도 수집
- 평균 및 표준편차 계산
- 95% 신뢰구간: 평균 ± 1.96 × 표준편차

**해석:**
> "SVM 모델의 정확도는 95% 신뢰수준에서 64% ~ 74% 범위입니다."

---

### 4. ANOVA (분산분석) - 알고리즘 성능 비교

**목적**: 4개 알고리즘(SVM, RF, NB, DT) 간 성능 차이가 통계적으로 유의미한지 검정

**데이터 수집 방법:**
1. Learning Curve를 그려 최적 훈련 크기(80%)를 선정
2. 최적 훈련 크기(80%)에서 각 알고리즘을 **10번 반복 실행** (10-Fold CV)
3. 각 알고리즘마다 10개의 정확도 측정값 수집

**ANOVA 변수:**
- k = 4 (알고리즘 수)
- n = 10 (측정 횟수 = 10-Fold CV)
- 자유도: df₁ = k-1 = 3, df₂ = k(n-1) = 36

**해석:**
- F 통계량 > F 임계값: 알고리즘 간 성능 차이가 **통계적으로 유의미**
- F 통계량 ≤ F 임계값: 차이가 우연에 의한 것일 수 있음

---

### 5. Clustering (군집분석)

**목적**: 유사한 인스턴스들을 그룹화

**알고리즘**: K-Means (k=3)
- 3개 클러스터로 시장 상황 분류
- 각 클러스터의 중심(Centroid) 분석
- 클러스터별 특성 파악

---

### 6. Association Rules (연관규칙)

**목적**: 속성 간 연관 패턴 발견

**알고리즘**: Apriori

**주요 지표:**
- **Support**: 규칙이 전체 데이터에서 나타나는 빈도
- **Confidence**: 선행 조건 발생 시 결과가 나타날 확률
- **Lift**: 규칙의 유용성 (1보다 크면 유의미)

**예시:**
> "IF ma_cross=golden AND volume_spike=high THEN price_direction=UP (Confidence: 75%, Lift: 2.1)"

---

# 5. Installation & Academic Purpose

## Prerequisites

```
Python 3.9 or higher
pip (Python package manager)
```

## Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/Jeong-Ryeol/Bitcoin-Price-Prediction-System.git
cd Bitcoin-Price-Prediction-System

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the complete pipeline
python3 run.py

# 5. Launch web application
streamlit run app.py
```

---

## Academic Purpose

이 프로젝트는 **데이터마이닝 과목**의 과제로 진행되었습니다.

### 학습 목표
- 실제 금융 데이터 수집 및 전처리
- 시계열 데이터 분석 및 특성 추출
- WEKA 데이터마이닝 도구 활용
- 다양한 분류 알고리즘 비교 및 평가
- 웹 기반 대시보드 개발

---

## WEKA 요구사항 충족

| 요구사항 | 조건 | 본 프로젝트 | 충족 |
|---------|------|-----------|------|
| Instances | 100개 이상 | 9,000개+ | O |
| Attributes | 4개 이상 | 9개 | O |
| Classification | 분류 분석 | SVM, RF, DT, NB | O |
| Clustering | 군집 분석 | K-Means | O |
| Association | 연관규칙 | Apriori | O |
| Learning Curve | 학습곡선 | 10-Fold CV | O |
| ANOVA | 분산분석 | One-Way ANOVA | O |

---

## Disclaimer

이 시스템은 **교육 목적**으로만 제작되었습니다.
실제 투자 결정에 사용해서는 안 됩니다.
과거 데이터는 미래 수익을 보장하지 않으며, 암호화폐 투자는 높은 리스크를 동반합니다.

---

## Developer

**Jeong Won Ryeol**
Department of Computer Science and Engineering

GitHub: https://github.com/Jeong-Ryeol

---

**Data Mining Project 2025**
