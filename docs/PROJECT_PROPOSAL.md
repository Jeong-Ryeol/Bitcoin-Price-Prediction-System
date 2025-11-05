# 비트코인 가격 예측 시스템 프로젝트 제안서
## Bitcoin Price Prediction System - Project Proposal

---

## 1. 프로젝트 개요 (Project Overview)

### 1.1 프로젝트 제목
**비트코인 가격 예측 시스템 (Bitcoin Price Prediction System)**

### 1.2 프로젝트 목표
데이터마이닝 기법과 머신러닝 알고리즘을 활용하여 암호화폐 시장의 비트코인 가격 방향을 예측하는 시스템을 개발한다. 실시간 데이터 수집, 기술적 지표 분석, 다양한 분류 알고리즘 비교를 통해 1시간 후 가격 방향(상승/하락/유지)을 예측하며, 사용자 친화적인 웹 대시보드를 제공한다.

### 1.3 프로젝트 배경
암호화폐 시장은 높은 변동성과 불확실성을 가지고 있어, 데이터 기반 예측 모델의 필요성이 크다. 본 프로젝트는 다음과 같은 학습 목표를 달성한다:

- 실제 금융 데이터 수집 및 전처리 경험
- 시계열 데이터의 특성 이해 및 분석
- WEKA 도구를 활용한 데이터 마이닝 실습
- 다양한 머신러닝 알고리즘 비교 및 성능 평가
- 웹 기반 대시보드 개발 및 배포

---

## 2. 데이터셋 (Dataset)

### 2.1 데이터 소스
- **API**: Upbit Public API (한국 최대 암호화폐 거래소)
- **Market**: KRW-BTC (원화 비트코인 시장)
- **API Key 요구사항**: 불필요 (Public API 사용)
- **수집 주기**: 1시간 단위 (hourly candles)

### 2.2 데이터 수집 범위
- **총 인스턴스 수**: 8,412개
- **데이터 기간**: 약 1년 (365일 × 24시간)
- **수집 날짜**: 2024년 11월 ~ 2025년 11월

### 2.3 데이터 속성 (Attributes)

#### 2.3.1 가격 데이터 (5개 속성)
| 속성명 | 타입 | 설명 |
|--------|------|------|
| `open` | Numeric | 시간대 시작 가격 (원화) |
| `high` | Numeric | 시간대 최고 가격 (원화) |
| `low` | Numeric | 시간대 최저 가격 (원화) |
| `close` | Numeric | 시간대 종료 가격 (원화) - 예측에 가장 중요 |
| `volume` | Numeric | 거래량 (BTC) - 시장 활성도 지표 |

#### 2.3.2 기술적 지표 (3개 속성)
| 속성명 | 타입 | 가능한 값 | 설명 |
|--------|------|-----------|------|
| `ma_cross` | Nominal | golden, dead, neutral | 이동평균 교차 신호 |
| `rsi_signal` | Nominal | overbought, oversold, neutral | 상대 강도 지수 신호 |
| `volume_spike` | Nominal | high, normal, low | 거래량 급등 여부 |

**기술적 지표 상세 설명:**

1. **MA Cross (Moving Average Crossover)**
   - Golden Cross: 단기 MA(5시간) > 장기 MA(20시간) → 상승 신호
   - Death Cross: 단기 MA < 장기 MA → 하락 신호
   - Neutral: 교차 없음

2. **RSI Signal (Relative Strength Index)**
   - Overbought: RSI > 70 → 과매수 상태, 하락 가능성
   - Oversold: RSI < 30 → 과매도 상태, 상승 가능성
   - Neutral: 30 ≤ RSI ≤ 70

3. **Volume Spike**
   - High: 거래량 > 평균 × 1.5
   - Low: 거래량 < 평균 × 0.5
   - Normal: 평균 수준

#### 2.3.3 타겟 클래스 (Target Class)
| 속성명 | 타입 | 가능한 값 | 설명 |
|--------|------|-----------|------|
| `price_direction` | Nominal | UP, DOWN, STABLE | 1시간 후 가격 방향 |

**클래스 정의:**
- **UP**: 현재 대비 0.3% 이상 상승
- **DOWN**: 현재 대비 0.3% 이상 하락
- **STABLE**: -0.3% ~ +0.3% 범위 내 유지

**클래스 분포 (8,412 인스턴스):**
- STABLE: 70.6% (5,939개)
- UP: 15.1% (1,270개)
- DOWN: 14.4% (1,203개)

### 2.4 데이터 품질
- **결측치**: 없음 (API에서 완전한 데이터 제공)
- **데이터 타입**: 정확히 정의됨 (5개 numeric, 4개 nominal)
- **시간 순서**: 과거에서 현재로 정렬
- **데이터 일관성**: API에서 검증된 실제 거래 데이터

---

## 3. 데이터마이닝 방법론 (Methodology)

### 3.1 데이터 수집 파이프라인
```
1. Upbit API 호출 → 2. 1시간 캔들 데이터 수집 → 3. CSV 저장
   ↓
4. 기술적 지표 계산 → 5. 패턴 탐지 → 6. 미래 라벨 생성
   ↓
7. 특성 엔지니어링 → 8. ARFF 파일 생성 (WEKA용)
```

### 3.2 데이터 전처리
1. **시계열 정렬**: 타임스탬프 기준 오름차순 정렬
2. **이동평균 계산**: MA5, MA20 (pandas 사용)
3. **RSI 계산**: 14일 기준 RSI (pandas-ta 라이브러리)
4. **거래량 평균**: 최근 50시간 이동평균
5. **레이블 생성**: 1시간 후 가격 변동률 계산 → 클래스 할당
6. **인코딩**: 범주형 변수 → LabelEncoder 적용

### 3.3 사용 알고리즘

#### 3.3.1 분류 알고리즘 (Classification)
| 알고리즘 | 라이브러리 | 주요 파라미터 | 성능 |
|----------|-----------|---------------|------|
| **Naive Bayes** | scikit-learn | GaussianNB() | 64% |
| **Decision Tree (J48)** | scikit-learn | max_depth=10 | 67% |
| **Random Forest** | scikit-learn | n_estimators=100 | 67% |
| **SVM** | scikit-learn | kernel='rbf' | **69%** |

**최종 선택 모델: SVM (Support Vector Machine)**

**참고**: 초기에는 OneR 알고리즘도 구현하였으나, 단일 속성 기반 분류의 한계로 인해 최종 버전에서는 제외하였습니다.

#### 3.3.2 군집화 알고리즘 (Clustering)
- **K-Means**: 3-5개 클러스터
- **PCA**: 2D 시각화

#### 3.3.3 연관규칙 마이닝 (Association Rules)
- **Apriori 알고리즘**: mlxtend 라이브러리
- **최소 지지도**: 0.1
- **최소 신뢰도**: 0.6

### 3.4 모델 평가 방법
1. **Train/Test Split**: 80% / 20% 비율
2. **Cross-Validation**: 5-Fold CV
3. **평가 지표**:
   - Accuracy (정확도)
   - Confusion Matrix (혼동 행렬)
   - Precision, Recall, F1-Score (클래스별)
   - CV Mean ± Std (교차 검증 평균 및 표준편차)

---

## 4. 시스템 아키텍처 (System Architecture)

### 4.1 기술 스택

| 계층 | 기술 |
|------|------|
| **데이터 수집** | Upbit Public API, requests |
| **데이터 처리** | pandas, numpy |
| **기술적 분석** | pandas-ta, mplfinance |
| **머신러닝** | scikit-learn, mlxtend |
| **데이터마이닝 도구** | WEKA (ARFF 파일 생성) |
| **웹 프레임워크** | Streamlit |
| **시각화** | Plotly, Matplotlib |
| **이미지 처리** | Pillow (PIL) |
| **배포** | Streamlit Cloud, GitHub |

### 4.2 프로젝트 구조
```
Bitcoin-Price-Prediction-System/
├── src/                           # 소스 코드
│   ├── collector.py              # 데이터 수집 (Upbit API)
│   ├── chart_analyzer.py         # 기술적 지표 계산
│   ├── predictor.py              # 머신러닝 모델
│   ├── arff_generator.py         # WEKA ARFF 파일 생성
│   └── performance_tracker.py    # 모델 성능 추적
│
├── data/                          # 데이터 디렉토리
│   ├── raw/                      # 원본 데이터
│   │   └── bitcoin_labeled.csv   # 8,412개 인스턴스
│   ├── processed/                # 전처리 데이터
│   │   ├── bitcoin_features.csv  # 특성 추가
│   │   ├── bitcoin_classification.arff
│   │   ├── bitcoin_clustering.arff
│   │   └── bitcoin_association.arff
│   └── charts/                   # 차트 이미지
│
├── models/                        # 학습된 모델
│   └── bitcoin_predictor.pkl     # SVM 모델 (69% 정확도)
│
├── docs/                          # 문서
│   ├── ALGORITHMS.md             # 알고리즘 설명
│   └── PROJECT_PROPOSAL.md       # 프로젝트 제안서 (본 문서)
│
├── app.py                         # Streamlit 웹 대시보드
├── run.py                         # 전체 파이프라인 실행 스크립트
├── requirements.txt               # 의존성 패키지
└── README.md                      # 프로젝트 README
```

### 4.3 실행 파이프라인
```bash
# 1. 데이터 수집 (1년치 8,412개 인스턴스)
python3 src/collector.py

# 2. 기술적 지표 계산 및 패턴 탐지
python3 src/chart_analyzer.py

# 3. WEKA ARFF 파일 생성
python3 src/arff_generator.py

# 4. 머신러닝 모델 학습
python3 src/predictor.py

# 5. 웹 대시보드 실행
streamlit run app.py
```

---

## 5. 주요 기능 (Key Features)

### 5.1 대시보드 페이지

#### 1) **Dashboard**
- 데이터셋 개요 및 통계
- 실시간 가격 차트 (캔들스틱 + 이동평균선)
- 클래스 분포 파이 차트
- 기술적 지표 분포

#### 2) **Live Prediction**
- 실시간 비트코인 데이터 수집
- 최신 데이터로 즉시 예측
- 클래스별 확률 분포 시각화
- 신뢰도 표시

#### 3) **Manual Prediction**
- **개별 입력 탭**: 8개 속성 수동 입력
- **CSV 업로드 탭**: 일괄 예측 (여러 인스턴스)
- 최소/최대 범위 표시 (입력 가이드)
- 예측 결과 CSV 다운로드

#### 4) **Model Performance**
- **알고리즘별 비교 막대 그래프**
- **시간대별 정확도 변화 라인 차트**
- **Confusion Matrix 히트맵**
- **교차 검증 박스플롯 및 바이올린 플롯**

#### 5) **Dataset Explorer**
- 전체 인스턴스 목록 (8,412개)
- 필터링 기능 (클래스, MA Cross 등)
- 속성 분포 분석 (파이 차트, 히스토그램)
- 시계열 그래프 (가격 추이, 거래량)

#### 6) **Chart Image Analysis**
- 비트코인 차트 스크린샷 업로드
- AI 기반 패턴 인식 (색상 분석)
- 트렌드 탐지 (상승/하락/횡보)
- 예측 신뢰도 표시

#### 7) **Historical Analysis**
- 날짜 범위 선택 필터
- 선택 기간 캔들스틱 차트
- 통계 요약 (평균, 최대, 최소, 표준편차)
- Dataset Explorer와 역할 분리로 차트 분석에 집중

#### 8) **WEKA Analysis**
- **Classification**: 4개 알고리즘 선택 실행
- **Decision Tree**: J48 스타일 트리 시각화
- **Clustering**: K-Means (2-5 클러스터)
- **Association Rules**: Apriori 알고리즘
  - 빈발 항목집합 탐지
  - 연관규칙 생성 (지지도, 신뢰도, Lift)
  - 상위 20개 규칙 표시
  - 규칙 상세 분석
- ARFF 파일 다운로드

#### 9) **About**
- 프로젝트 개요
- 기술 스택
- 데이터 속성 설명
- 사용 가이드
- GitHub 저장소 링크

### 5.2 핵심 기능

#### 5.2.1 실시간 데이터 업데이트
- Upbit API를 통한 최신 데이터 수집
- 캐시 기능 (5분 TTL)
- API rate limit 고려 (0.15초 딜레이)

#### 5.2.2 다중 알고리즘 비교
- 4개 알고리즘 동시 평가
- 성능 지표 시각화
- 시간대별 정확도 추적

#### 5.2.3 수동 데이터 입력
- Form 기반 입력 (범위 가이드 제공)
- CSV 파일 일괄 업로드
- 예측 결과 다운로드

#### 5.2.4 WEKA 통합
- ARFF 파일 자동 생성 (3종류)
- 웹에서 WEKA 스타일 분석
- 교차 검증 및 혼동 행렬

---

## 6. 기대 효과 및 활용 방안

### 6.1 학습 목표 달성
1. **실제 데이터 수집 경험**: API 활용, 대용량 데이터 처리
2. **시계열 데이터 분석**: 시간 의존성, 트렌드 탐지
3. **WEKA 실습**: ARFF 파일 생성 및 분석
4. **알고리즘 비교**: 4개 분류 알고리즘 성능 평가
5. **웹 개발**: Streamlit을 활용한 대시보드 구축
6. **배포 경험**: GitHub, Streamlit Cloud 배포

### 6.2 확장 가능성
- 다른 암호화폐 추가 (이더리움, 리플 등)
- LSTM, GRU 등 딥러닝 모델 통합
- 실시간 알림 기능 (Telegram Bot)
- 백테스팅 시뮬레이션
- 포트폴리오 최적화

### 6.3 실용적 가치
- 암호화폐 투자 참고 자료
- 데이터마이닝 교육 자료
- 웹 대시보드 개발 템플릿
- 시계열 예측 프로젝트 예시

---

## 7. 프로젝트 일정 (Timeline)

| 주차 | 작업 내용 | 상태 |
|------|----------|------|
| 1주차 | 요구사항 분석, 데이터 소스 조사 | ✅ 완료 |
| 2주차 | 데이터 수집 모듈 개발 (collector.py) | ✅ 완료 |
| 3주차 | 기술적 지표 계산 (chart_analyzer.py) | ✅ 완료 |
| 4주차 | 머신러닝 모델 구현 (predictor.py) | ✅ 완료 |
| 5주차 | WEKA 통합 (arff_generator.py) | ✅ 완료 |
| 6주차 | 웹 대시보드 개발 (app.py) | ✅ 완료 |
| 7주차 | 성능 추적 모듈 (performance_tracker.py) | ✅ 완료 |
| 8주차 | 1년치 데이터 수집 (8,412 인스턴스) | ✅ 완료 |
| 9주차 | 테스트 및 디버깅, 문서화 | ✅ 완료 |
| 10주차 | 배포 및 최종 발표 준비 | ⏳ 진행 중 |

---

## 8. 기술적 도전 과제 및 해결 방법

### 8.1 클래스 불균형 문제
- **문제**: STABLE 클래스 70.6%, UP/DOWN 각 15%
- **해결**:
  - train_test_split에서 stratify 제거 (전체 분포 유지)
  - 클래스별 정확도 개별 평가
  - Confusion Matrix로 세부 성능 분석

### 8.2 API Rate Limit
- **문제**: Upbit API 호출 제한 (초당 10회)
- **해결**:
  - 각 요청 사이 0.15초 딜레이
  - 200개씩 배치 수집 (44 batches)
  - 에러 핸들링 및 재시도 로직

### 8.3 시계열 데이터 특성
- **문제**: 시간 순서 의존성
- **해결**:
  - 과거 → 현재 순서 유지
  - 이동평균 등 시계열 특성 반영
  - 테스트 세트는 최신 데이터 사용

### 8.4 모델 성능 향상
- **시도한 방법**:
  - 특성 엔지니어링 (MA Cross, RSI, Volume Spike)
  - 하이퍼파라미터 튜닝
  - 여러 알고리즘 비교 (SVM이 최고 69%)
- **한계**:
  - 암호화폐 시장의 높은 노이즈
  - 외부 요인 미반영 (뉴스, 규제 등)

### 8.5 Streamlit Cloud 배포 에러
- **문제**: deprecated 함수 및 import 누락
- **해결**:
  - `st.experimental_rerun()` 제거 (Model Performance 페이지)
  - `import numpy as np` 추가 (Confusion Matrix 에러 해결)
  - OneR 알고리즘 제거 (안정성 향상)
  - 페이지별 역할 명확화 (Historical Analysis vs Dataset Explorer)

---

## 9. 프로젝트 성과

### 9.1 정량적 성과
- **데이터셋**: 8,412개 인스턴스 (WEKA 요구사항: 100+ 충족)
- **속성**: 9개 (8 features + 1 class, WEKA 요구사항: 4+ 충족)
- **최고 정확도**: 69% (SVM)
- **웹 페이지**: 9개 기능 페이지
- **분류 알고리즘**: 4개 (Naive Bayes, Decision Tree, Random Forest, SVM)
- **기타 알고리즘**: 군집화 1개 (K-Means), 연관규칙 1개 (Apriori)

### 9.2 정성적 성과
- 실제 금융 데이터 수집 및 분석 경험
- WEKA 도구 이해 및 활용 능력
- 웹 대시보드 개발 능력
- GitHub을 통한 협업 및 배포 경험
- 데이터마이닝 프로젝트 전체 파이프라인 경험

---

## 10. 결론

본 프로젝트는 실제 암호화폐 시장 데이터를 활용하여 데이터마이닝 기법을 종합적으로 학습하고 적용한 사례이다. Upbit Public API를 통해 1년치 데이터를 수집하고, 기술적 지표를 계산하여 특성을 엔지니어링하였으며, 다양한 머신러닝 알고리즘을 비교 분석하였다.

특히 WEKA 도구와의 통합을 통해 전통적인 데이터마이닝 방법론을 실습하고, Streamlit을 활용하여 사용자 친화적인 웹 대시보드를 구축함으로써 실무 적용 가능성을 높였다.

비록 암호화폐 시장의 높은 변동성으로 인해 정확도 69%의 한계가 있지만, 이는 교육적 목적의 프로젝트로서 데이터마이닝의 전체 프로세스를 이해하고 실습하는 데 충분한 가치를 제공한다.

---

## 11. 참고 자료 (References)

### 11.1 API 및 라이브러리
- Upbit Public API Documentation: https://docs.upbit.com/
- scikit-learn Documentation: https://scikit-learn.org/
- Streamlit Documentation: https://docs.streamlit.io/
- pandas-ta Library: https://github.com/twopirllc/pandas-ta
- mlxtend (Apriori): http://rasbt.github.io/mlxtend/

### 11.2 기술적 분석
- Moving Average Crossover Strategy
- RSI (Relative Strength Index) Indicator
- Volume Analysis in Trading

### 11.3 머신러닝
- Naive Bayes Classifier
- Decision Tree (J48/C4.5 Algorithm)
- Random Forest Ensemble Method
- Support Vector Machine (SVM)
- K-Means Clustering
- Apriori Algorithm for Association Rules

---

## 12. 부록 (Appendix)

### 12.1 GitHub 저장소
**Repository URL**: https://github.com/Jeong-Ryeol/Bitcoin-Price-Prediction-System

### 12.2 라이브 데모
**Streamlit Cloud**: (배포 후 URL 추가 예정)

### 12.3 프로젝트 개발자
**이름**: 정원렬 (Jeong Won Ryeol)
**소속**: 컴퓨터공학과
**GitHub**: https://github.com/Jeong-Ryeol

### 12.4 면책 조항 (Disclaimer)
본 시스템은 **교육 목적으로만** 제작되었으며, 실제 투자 결정에 사용해서는 안 됩니다. 과거 데이터는 미래 수익을 보장하지 않으며, 암호화폐 투자는 높은 리스크를 동반합니다. 투자의 모든 책임은 투자자 본인에게 있습니다.

---

**작성일**: 2025년 11월 5일
**버전**: 1.0
**문서 유형**: 프로젝트 제안서 (Project Proposal)
