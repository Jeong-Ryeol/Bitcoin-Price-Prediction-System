# 🪙 Bitcoin Price Prediction System

> 데이터마이닝 프로젝트 - 차트 패턴 기반 비트코인 가격 방향 예측 시스템

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## 📋 프로젝트 개요

실시간 비트코인 가격 데이터와 차트 패턴 분석을 통해 1시간 후 가격 방향(UP/DOWN/STABLE)을 예측하는 머신러닝 시스템입니다.

### 🎯 핵심 기능

- ✅ **과거 데이터 수집**: Upbit API를 통한 200시간 캔들 데이터 자동 수집
- ✅ **차트 패턴 인식**: 이동평균선, RSI, 거래량 기반 기술적 분석
- ✅ **WEKA 지원**: 분류/군집화/연관규칙 학습용 ARFF 파일 자동 생성
- ✅ **머신러닝 예측**: Random Forest 등 다양한 알고리즘 비교 및 학습
- ✅ **웹 대시보드**: Streamlit 기반 인터랙티브 시각화 및 실시간 예측
- ✅ **클라우드 배포**: Streamlit Cloud를 통한 웹 배포 지원

## 🎬 데모

**웹 대시보드 주요 화면:**
- 📊 Dashboard: 데이터 통계 및 차트 시각화
- 🔮 Live Prediction: 실시간 데이터로 가격 방향 예측
- 📈 Historical Analysis: 과거 데이터 분석 및 필터링
- ℹ️ About: 프로젝트 정보 및 사용 방법

## 📊 데이터 속성

### 수집 속성 (8개)

| 카테고리 | 속성명 | 타입 | 설명 |
|---------|--------|------|------|
| **가격 데이터** | open_price | NUMERIC | 시가 |
| | high_price | NUMERIC | 고가 |
| | low_price | NUMERIC | 저가 |
| | close_price | NUMERIC | 종가 |
| | volume | NUMERIC | 거래량 |
| **차트 패턴** | ma_cross | CATEGORICAL | 이동평균선 교차 {golden, dead, neutral} |
| | rsi_signal | CATEGORICAL | RSI 신호 {overbought, oversold, neutral} |
| | volume_spike | CATEGORICAL | 거래량 급증 {high, normal, low} |

### 타겟 클래스

- **price_direction**: {UP, DOWN, STABLE}
  - UP: 1시간 후 +1.5% 이상 상승
  - DOWN: 1시간 후 -1.5% 이상 하락
  - STABLE: -1.5% ~ +1.5% 범위

## 🚀 시작하기

### 1. 환경 설정

```bash
# 프로젝트 디렉토리로 이동
cd ~/Desktop/project/datamining

# 가상환경 생성 (권장)
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# 라이브러리 설치
pip install -r requirements.txt
```

### 2. 데이터 수집 및 전처리

**전체 파이프라인 자동 실행:**
```bash
python3 run.py
```

**또는 단계별 실행:**
```bash
# Step 1: 과거 데이터 수집 (200시간)
python3 src/collector.py

# Step 2: 차트 패턴 분석
python3 src/chart_analyzer.py

# Step 3: WEKA ARFF 파일 생성
python3 src/arff_generator.py

# Step 4: 머신러닝 모델 학습
python3 src/predictor.py
```

### 3. 웹 대시보드 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

## 📁 프로젝트 구조

```
datamining/
├── data/
│   ├── raw/                        # 원본 데이터
│   │   ├── bitcoin_candles.csv
│   │   └── bitcoin_labeled.csv
│   ├── charts/                     # 생성된 차트 이미지
│   └── processed/                  # 처리된 데이터 & ARFF 파일
│       ├── bitcoin_features.csv
│       ├── bitcoin_classification.arff
│       ├── bitcoin_clustering.arff
│       └── bitcoin_association.arff
├── models/
│   └── bitcoin_predictor.pkl       # 학습된 모델
├── src/
│   ├── collector.py               # 데이터 수집
│   ├── chart_analyzer.py          # 차트 패턴 분석
│   ├── arff_generator.py          # ARFF 생성
│   └── predictor.py               # 예측 모델
├── app.py                         # Streamlit 대시보드
├── run.py                         # 통합 실행 스크립트
├── requirements.txt
├── README.md
└── SETUP_GUIDE.md
```

## 🤖 WEKA 분석 가이드

### 1. 분류 (Classification)

```
1. WEKA 실행
2. Explorer → Open file → data/processed/bitcoin_classification.arff
3. Classify 탭 선택
4. Classifier: trees.J48 (Decision Tree)
5. Test options: Cross-validation, Folds: 10
6. Class: price_direction 선택
7. Start 클릭
```

**추천 알고리즘:**
- `trees.J48` - Decision Tree
- `trees.RandomForest` - Random Forest
- `bayes.NaiveBayes` - Naive Bayes
- `functions.SMO` - SVM

### 2. 군집화 (Clustering)

```
1. bitcoin_clustering.arff 열기
2. Cluster 탭 선택
3. Clusterer: SimpleKMeans
4. numClusters: 3
5. Start 클릭
```

### 3. 연관규칙 (Association)

```
1. bitcoin_association.arff 열기
2. Associate 탭 선택
3. Associator: Apriori
4. minSupport: 0.1
5. minConfidence: 0.8
6. Start 클릭
```

## 🛠️ 기술 스택

### Backend
- **Python 3.9+**
- **pandas, numpy** - 데이터 처리
- **mplfinance** - 차트 생성
- **pandas-ta** - 기술적 지표 계산
- **scikit-learn** - 머신러닝

### Frontend
- **Streamlit** - 웹 프레임워크
- **Plotly** - 인터랙티브 차트

### Data Source
- **Upbit Public API** (무료, API Key 불필요)

### Deployment
- **Streamlit Cloud** (무료)

## 📈 모델 성능

**테스트 결과 (예상):**
- Random Forest: ~82-85% 정확도
- Decision Tree: ~78-80% 정확도
- Naive Bayes: ~75-78% 정확도

*실제 성능은 데이터 수집 시점과 시장 상황에 따라 달라질 수 있습니다.*

## 🌐 클라우드 배포

### Streamlit Cloud 배포 방법

1. **GitHub 리포지토리 생성**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

2. **Streamlit Cloud 설정**
- [share.streamlit.io](https://share.streamlit.io/) 접속
- GitHub 연동
- Repository, Branch, Main file (app.py) 선택
- Deploy 클릭

3. **환경 변수 설정** (필요시)
- Streamlit Cloud 대시보드에서 Secrets 추가

## 🎓 발표 시연 시나리오

### 1. 프로젝트 소개 (2분)
- 주제 및 목적 설명
- 데이터 수집 방법 및 속성 설명

### 2. 웹 대시보드 시연 (3분)
- Dashboard: 데이터 통계 및 시각화
- Live Prediction: 실시간 예측 시연
- Historical Analysis: 과거 데이터 분석

### 3. WEKA 분석 결과 (3분)
- Classification 정확도 및 Confusion Matrix
- Clustering 결과 시각화
- Association Rules 주요 패턴

### 4. 기술적 구현 설명 (2분)
- 시간차 예측 구조 (t시간 속성 → t+1시간 클래스)
- 차트 패턴 인식 알고리즘
- 모델 선택 및 최적화 과정

## 💡 주요 특징

### 1. 시간차 예측 구조
```
[t 시간]                    [t+1 시간]
속성 수집                   클래스 결정
- 가격: 125,000,000원
- MA Cross: golden          → UP (실제 상승 확인)
- RSI: neutral
- Volume: normal
```

이를 통해 **"현재 → 미래"** 예측이 가능합니다.

### 2. 설명 가능한 AI
- 딥러닝 대신 **기술적 지표 기반** 패턴 인식
- 투명하고 해석 가능한 규칙
- 금융 분야에서 실제 사용되는 지표 활용

### 3. 다목적 데이터셋
- **분류**: 가격 방향 예측
- **군집화**: 유사 시장 상황 그룹화
- **연관규칙**: 패턴 간 연관성 발견

## ⚠️ 주의사항

1. **교육 목적**: 이 시스템은 데이터마이닝 학습용이며, 실제 투자에 사용하지 마세요.
2. **API 제한**: Upbit API는 초당 10회 제한이 있으니 주의하세요.
3. **데이터 편향**: 특정 기간 데이터만 학습하면 과적합 가능성이 있습니다.

## 🔧 문제 해결

### API 에러 발생 시
```bash
# 인터넷 연결 확인
# 잠시 후 재시도 (rate limit)
```

### WEKA 메모리 부족 시
```bash
java -Xmx2048m -jar weka.jar
```

### Streamlit 오류 시
```bash
streamlit cache clear
streamlit run app.py
```

## 📚 참고 자료

- [Upbit API 문서](https://docs.upbit.com/reference)
- [WEKA 공식 문서](https://www.cs.waikato.ac.nz/ml/weka/)
- [Streamlit 문서](https://docs.streamlit.io/)
- [pandas-ta 문서](https://github.com/twopirllc/pandas-ta)

## 📝 개발 일정

- **Day 1-2**: 데이터 수집 및 전처리 ✅
- **Day 3-4**: 차트 분석 및 ARFF 생성 ✅
- **Day 5-6**: 머신러닝 모델 및 대시보드 ✅
- **Day 7**: 배포 및 테스트

## 👨‍💻 Author

**Data Mining Project 2025**

## 📄 License

This project is for educational purposes only.

---

Made with ❤️ for Data Mining Course
