# 발표 대본 (8분)

## About 페이지를 보여주면서 발표

---

## 오프닝 (30초)

안녕하세요. 비트코인 가격 예측 시스템을 발표하겠습니다.

이 프로젝트는 데이터마이닝 기법을 활용해서 비트코인의 1시간 후 가격 방향을 예측하는 시스템입니다.

지금부터 About 페이지를 통해 프로젝트를 설명드리겠습니다.

---

## Tab 1: Project Overview (1분 30초)

*[Project Overview 탭 클릭]*

먼저 프로젝트 개요입니다.

### 프로젝트 목표
- Upbit API를 통해 1년치 시간별 비트코인 데이터를 수집했습니다.
- 이동평균선, RSI 같은 기술적 지표를 계산하고
- 4가지 머신러닝 알고리즘으로 가격 방향을 예측합니다.

### Dataset Statistics
화면에 보시는 것처럼 총 9,000개 이상의 인스턴스를 수집했고, 8개의 속성과 1개의 클래스로 구성되어 있습니다. 데이터 간격은 1시간 단위입니다.

WEKA 요구사항이 최소 100개 인스턴스, 4개 속성인데 저희는 이를 충분히 충족합니다.

### Technology Stack
데이터 수집은 Upbit API, 분석은 pandas와 scikit-learn, 시각화는 Streamlit과 Plotly를 사용했습니다.

아래 파이 차트는 클래스 분포를 보여줍니다. UP, DOWN, STABLE 세 가지 클래스가 있습니다.

---

## Tab 2: Data Structure (1분 30초)

*[Data Structure 탭 클릭]*

다음은 데이터 구조입니다.

### Instance 구조
하나의 인스턴스는 8개의 속성과 1개의 클래스로 구성됩니다.

### Attributes - 수치형 5개
- open, high, low, close: 시간봉의 시가, 고가, 저가, 종가
- volume: 거래량

### Attributes - 범주형 3개
- ma_cross: 이동평균선 교차 신호 (golden, dead, neutral)
- rsi_signal: RSI 신호 (overbought, oversold, neutral)
- volume_spike: 거래량 급등/급락 신호 (high, normal, low)

### Class
price_direction이 클래스입니다.
- 1시간 후 가격이 0.3% 이상 오르면 UP
- 0.3% 이상 내리면 DOWN
- 그 사이면 STABLE로 분류합니다.

### Technical Indicators
MA Cross는 단기 5시간 이동평균과 장기 20시간 이동평균을 비교합니다. 단기가 장기를 상향 돌파하면 Golden Cross로 상승 신호입니다.

RSI는 14시간 기준으로 70 이상이면 과매수, 30 이하면 과매도로 판단합니다.

아래 샘플 데이터에서 실제 인스턴스 예시를 확인할 수 있습니다.

---

## Tab 3: Features Guide (1분 30초)

*[Features Guide 탭 클릭]*

이 시스템의 주요 기능들입니다.

### Dashboard
전체 데이터셋의 통계와 가격 차트, 클래스 분포를 한눈에 볼 수 있습니다.

### Live Prediction
실시간으로 Upbit API에서 데이터를 가져와서 즉시 예측합니다. 버튼 하나로 현재 시점의 1시간 후 가격 방향을 예측할 수 있습니다.

### Manual Prediction
사용자가 직접 데이터를 입력하거나 CSV 파일을 업로드해서 예측할 수 있습니다.

### Dataset Explorer
전체 데이터셋을 필터링하고 분석할 수 있습니다.

### Chart Image Analysis
차트 스크린샷을 업로드하면 색상 분석으로 트렌드를 감지합니다.

분석 로직은 다음과 같습니다:
- 이미지를 RGB 채널로 분리해서 각 채널의 평균값을 계산합니다
- 캔들스틱 차트에서 빨간 캔들은 상승, 파란 캔들은 하락을 의미하므로
- 빨간색 비율이 높으면 상승 추세, 파란색 비율이 높으면 하락 추세로 판단합니다
- 실험적인 기능이지만 이미지 기반 분석의 가능성을 보여줍니다

### Historical Analysis
특정 기간을 선택해서 캔들스틱 차트와 통계를 확인할 수 있습니다.

### WEKA Analysis
가장 중요한 기능입니다. WEKA의 주요 분석 기능을 웹에서 직접 실행할 수 있습니다.
- Classification: 4가지 알고리즘으로 분류
- Decision Tree: 의사결정나무 시각화
- Clustering: K-Means 군집 분석
- Association Rules: Apriori 연관규칙
- ANOVA: 알고리즘 간 성능 비교

### Model Performance
Learning Curve와 95% 신뢰구간을 통해 모델 성능을 분석합니다.

---

## Tab 4: Data Mining Methods (2분)

*[Data Mining Methods 탭 클릭]*

사용한 데이터마이닝 방법론입니다.

### Classification
4가지 알고리즘을 사용했습니다.
- Naive Bayes: 베이즈 정리 기반, 빠른 학습
- Decision Tree: 규칙 기반, 해석이 쉬움
- Random Forest: 앙상블 기법, 과적합 방지
- SVM: 최적 결정 경계 탐색, 가장 좋은 성능

10-Fold Cross Validation으로 평가했습니다.

### Learning Curve
학습 데이터 크기를 10%부터 100%까지 늘려가며 성능 변화를 분석했습니다. 학습곡선이 수렴하면 데이터가 충분하다는 의미입니다.

### Performance Confidence Interval
10-Fold CV 결과로 95% 신뢰구간을 계산했습니다. 예를 들어 SVM의 정확도는 95% 신뢰수준에서 64%~74% 범위입니다.

### ANOVA
4개 알고리즘의 성능 차이가 통계적으로 유의미한지 검정했습니다.

방법은 다음과 같습니다:
1. Learning Curve로 최적 훈련 크기 80%를 선정
2. 각 알고리즘을 10-Fold CV로 10번 측정
3. F 통계량과 F 임계값을 비교

F 통계량이 임계값보다 크면 알고리즘 간 성능 차이가 통계적으로 유의미합니다.

### Clustering
K-Means로 3개 클러스터를 만들어 시장 상황을 분류했습니다.

### Association Rules
Apriori 알고리즘으로 속성 간 연관 패턴을 찾았습니다. Support, Confidence, Lift 지표를 사용합니다.

---

## Tab 5: Installation & 마무리 (1분)

*[Installation 탭 클릭]*

마지막으로 설치 방법입니다.

Python 3.9 이상이 필요하고, git clone으로 저장소를 받은 후 가상환경을 만들고 requirements를 설치하면 됩니다.

run.py를 실행하면 데이터 수집부터 모델 학습까지 전체 파이프라인이 자동 실행되고, streamlit run app.py로 웹 앱을 실행할 수 있습니다.

### WEKA 요구사항 충족
표에서 보시는 것처럼 모든 요구사항을 충족했습니다.
- 인스턴스 100개 이상: 9,000개 이상 수집
- 속성 4개 이상: 9개 사용
- Classification, Clustering, Association, Learning Curve, ANOVA 모두 구현

### 마무리
이 시스템은 교육 목적으로 제작되었습니다. 실제 투자에는 사용하지 마시기 바랍니다.

이상으로 발표를 마치겠습니다. 감사합니다.

---

## 시간 체크

| 구간 | 시간 | 누적 |
|------|------|------|
| 오프닝 | 0:30 | 0:30 |
| Tab 1: Project Overview | 1:30 | 2:00 |
| Tab 2: Data Structure | 1:30 | 3:30 |
| Tab 3: Features Guide | 1:30 | 5:00 |
| Tab 4: Data Mining Methods | 2:00 | 7:00 |
| Tab 5: Installation & 마무리 | 1:00 | 8:00 |

**총 8분**
