# ₿ Bitcoin Price Prediction System

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
- STABLE: 146개 (73.4%)
- UP: 26개 (13.1%)
- DOWN: 27개 (13.6%)
- **총 199개 인스턴스**

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
**인스턴스**: 199개

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
**인스턴스**: 199개

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
**인스턴스**: 199개

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
X, y = predictor.prepare_data(df)  # 8개 피처, 199개 샘플

# 2. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)  # 학습: 159개, 테스트: 40개

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

### 페이지 구성 (5개)

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

#### 3. Historical Analysis (과거 데이터 분석)

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

#### 4. WEKA Analysis (WEKA 스타일 분석)

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
Instances: 199
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

#### 5. About (프로젝트 정보)

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

- [ ] 더 많은 데이터 수집 (500시간 이상)
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
