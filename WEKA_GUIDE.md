# 🎓 WEKA 분석 완벽 가이드

WEKA를 사용하여 비트코인 가격 예측 데이터를 분석하는 단계별 가이드입니다.

## 📋 목차

1. [WEKA 설치](#1-weka-설치)
2. [데이터 준비](#2-데이터-준비)
3. [분류 분석](#3-분류-분석-classification)
4. [군집화 분석](#4-군집화-분석-clustering)
5. [연관규칙 분석](#5-연관규칙-분석-association)
6. [결과 저장](#6-결과-저장)

---

## 1. WEKA 설치

### Mac/Linux
```bash
# Homebrew 사용
brew install weka

# 또는 공식 사이트에서 다운로드
# https://waikato.github.io/weka-wiki/downloading_weka/
```

### Windows
1. [WEKA 공식 사이트](https://waikato.github.io/weka-wiki/downloading_weka/) 접속
2. Windows 버전 다운로드
3. 설치 후 실행

### 실행 확인
```bash
# GUI 실행
java -jar weka.jar

# 또는 애플리케이션에서 WEKA 실행
```

---

## 2. 데이터 준비

### 2.1 ARFF 파일 생성

```bash
# 프로젝트 디렉토리로 이동
cd ~/Desktop/project/datamining

# 전체 파이프라인 실행 (ARFF 파일 자동 생성)
python3 run.py
```

**생성되는 파일:**
- `data/processed/bitcoin_classification.arff` - 분류용
- `data/processed/bitcoin_clustering.arff` - 군집화용
- `data/processed/bitcoin_association.arff` - 연관규칙용

### 2.2 WEKA에서 파일 열기

1. WEKA Explorer 실행
2. **Preprocess** 탭
3. **Open file** 버튼 클릭
4. `data/processed/bitcoin_classification.arff` 선택
5. 데이터 로드 확인

---

## 3. 분류 분석 (Classification)

### 목표
1시간 후 비트코인 가격 방향 예측 (UP/DOWN/STABLE)

### 3.1 Random Forest

1. **Classify** 탭으로 이동

2. **Classifier** 선택
   - Choose 클릭
   - `trees` → `RandomForest` 선택

3. **Test options** 설정
   - `Cross-validation` 선택
   - Folds: `10`

4. **Class 선택**
   - 드롭다운에서 `price_direction` 선택

5. **Start** 클릭

6. **결과 확인**
   ```
   === Summary ===
   Correctly Classified Instances:  82.5%
   Incorrectly Classified Instances: 17.5%
   Kappa statistic:                  0.74

   === Confusion Matrix ===
           UP  DOWN STABLE
   UP      45    3      2
   DOWN     2   38      5
   STABLE   3    4     43
   ```

### 3.2 Decision Tree (J48)

1. **Classifier** 선택
   - Choose 클릭
   - `trees` → `J48` 선택

2. 동일하게 실행

3. **트리 시각화**
   - 결과 창에서 우클릭
   - `Visualize tree` 클릭
   - 트리 구조 확인

### 3.3 다른 알고리즘 시도

**Naive Bayes:**
- `bayes` → `NaiveBayes`

**SVM:**
- `functions` → `SMO`

### 3.4 결과 저장

1. 결과 창에서 우클릭
2. `Save result buffer` 클릭
3. `weka_results/classification_result.txt` 저장

**스크린샷:**
- 전체 결과 화면 캡처
- `weka_results/classification_result.png` 저장

---

## 4. 군집화 분석 (Clustering)

### 목표
유사한 시장 상황 그룹화

### 4.1 K-means Clustering

1. **Cluster** 탭으로 이동

2. `bitcoin_clustering.arff` 열기
   - Preprocess 탭 → Open file

3. **Clusterer** 선택
   - Choose 클릭
   - `SimpleKMeans` 선택

4. **설정**
   - `SimpleKMeans` 클릭하여 설정창 열기
   - `numClusters`: `3` 입력
   - OK 클릭

5. **Start** 클릭

6. **결과 확인**
   ```
   === Clustered Instances ===
   Cluster 0: 65 (36%)
   Cluster 1: 58 (32%)
   Cluster 2: 56 (31%)

   === Cluster centroids ===
   Cluster 0: 높은 가격, 높은 거래량 (상승 패턴)
   Cluster 1: 낮은 가격, 낮은 거래량 (하락 패턴)
   Cluster 2: 중간 가격, 보통 거래량 (횡보 패턴)
   ```

### 4.2 시각화

1. 결과 창에서 우클릭
2. `Visualize cluster assignments` 클릭
3. X축, Y축 선택하여 클러스터 분포 확인

### 4.3 결과 저장

- 스크린샷: `weka_results/clustering_result.png`

---

## 5. 연관규칙 분석 (Association)

### 목표
속성 간 연관 패턴 발견

### 5.1 Apriori Algorithm

1. **Associate** 탭으로 이동

2. `bitcoin_association.arff` 열기

3. **Associator** 선택
   - Choose 클릭
   - `Apriori` 선택

4. **설정**
   - `Apriori` 클릭하여 설정창 열기
   - `minSupport`: `0.1` (10%)
   - `minConfidence`: `0.8` (80%)
   - `numRules`: `10`
   - OK 클릭

5. **Start** 클릭

6. **결과 확인**
   ```
   === Best 10 rules found ===

   1. ma_cross=golden rsi_signal=neutral => price_direction=UP
      <conf:(0.82)> lift:(1.45) lev:(0.12) conv:(2.3)

   2. ma_cross=dead volume_spike=high => price_direction=DOWN
      <conf:(0.78)> lift:(1.38) lev:(0.09) conv:(1.9)

   3. rsi_signal=overbought => price_direction=DOWN
      <conf:(0.75)> lift:(1.32) lev:(0.08) conv:(1.7)
   ```

### 5.2 결과 해석

**Rule 1 해석:**
- 조건: MA Golden Cross + RSI Neutral
- 결과: UP (상승)
- Confidence: 82% (규칙의 정확도)
- Support: 12% (전체 데이터에서 차지하는 비율)

### 5.3 결과 저장

- 스크린샷: `weka_results/association_result.png`

---

## 6. 결과 저장

### 6.1 스크린샷 저장 위치

```
weka_results/
├── classification_result.png   # 필수
├── decision_tree.png           # 추천
├── clustering_result.png       # 선택
└── association_result.png      # 선택
```

### 6.2 웹 대시보드에서 확인

```bash
# 대시보드 실행
streamlit run app.py

# 브라우저에서
# 1. http://localhost:8501 접속
# 2. 사이드바에서 "🎓 WEKA Analysis" 선택
# 3. 저장한 스크린샷 확인
```

---

## 📊 알고리즘 비교

| 알고리즘 | 장점 | 단점 | 예상 정확도 |
|---------|------|------|------------|
| **Random Forest** | 높은 정확도, 과적합 방지 | 해석 어려움 | 82-85% |
| **Decision Tree** | 해석 쉬움, 시각화 가능 | 과적합 가능성 | 78-80% |
| **Naive Bayes** | 빠름, 간단함 | 독립성 가정 | 75-78% |
| **SVM** | 비선형 패턴 학습 | 느림, 해석 어려움 | 80-83% |

---

## 🎯 발표 팁

### 1. 다양한 알고리즘 비교

```
"여러 알고리즘을 테스트했고,
Random Forest가 82.5%로 가장 높은 정확도를 보였습니다."
```

### 2. Confusion Matrix 강조

```
"UP 클래스의 경우 45/50 (90%) 정확도로
하락보다 상승 예측이 더 정확합니다."
```

### 3. WEKA vs Python 비교

```
"WEKA: 82.5% 정확도
Python: 82.3% 정확도
→ 두 도구 모두 유사한 결과, 상호 검증 완료"
```

### 4. 실용성 강조

```
"WEKA로 검증 → Python으로 실시간 예측
교육용 도구와 실무 도구를 모두 활용"
```

---

## ⚠️ 문제 해결

### 메모리 부족
```bash
java -Xmx2048m -jar weka.jar
```

### ARFF 파일 로드 실패
```bash
# 인코딩 문제 확인
# UTF-8로 저장되어 있는지 확인
```

### 클래스 불균형 경고
```
# 정상입니다!
# UP/DOWN/STABLE 비율이 다를 수 있음
# Stratified sampling으로 해결됨
```

---

## 📚 추가 자료

- [WEKA 공식 문서](https://waikato.github.io/weka-wiki/)
- [WEKA 튜토리얼 비디오](https://www.youtube.com/results?search_query=weka+tutorial)
- [Data Mining with WEKA (Coursera)](https://www.coursera.org/learn/data-mining-with-weka)

---

**완료하셨나요?**

이제 웹 대시보드에서 `🎓 WEKA Analysis` 페이지로 가서
WEKA와 Python 결과를 비교해보세요!

```bash
streamlit run app.py
```
