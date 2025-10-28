# WEKA 분석 결과 저장 폴더

이 폴더에 WEKA 분석 스크린샷을 저장하면 웹 대시보드에서 자동으로 표시됩니다.

## 📸 저장할 스크린샷 목록

### 1. Classification (분류)

**파일명:** `classification_result.png`

**WEKA 실행 방법:**
1. WEKA Explorer 실행
2. Open file → `data/processed/bitcoin_classification.arff`
3. Classify 탭 선택
4. Classifier: `trees.RandomForest` 선택
5. Test options: `Cross-validation`, Folds: `10`
6. Class 선택: `price_direction`
7. Start 클릭
8. 결과 창에서 우클릭 → `Save result buffer` 또는 스크린샷

**캡처할 내용:**
- Correctly Classified Instances
- Confusion Matrix
- Detailed Accuracy By Class

---

### 2. Clustering (군집화)

**파일명:** `clustering_result.png`

**WEKA 실행 방법:**
1. WEKA Explorer 실행
2. Open file → `data/processed/bitcoin_clustering.arff`
3. Cluster 탭 선택
4. Clusterer: `SimpleKMeans` 선택
5. numClusters: `3`
6. Start 클릭
7. Visualize cluster assignments 클릭 (선택)
8. 스크린샷

**캡처할 내용:**
- Clustered Instances 수
- Cluster centroids
- (선택) Cluster visualization

---

### 3. Association Rules (연관규칙)

**파일명:** `association_result.png`

**WEKA 실행 방법:**
1. WEKA Explorer 실행
2. Open file → `data/processed/bitcoin_association.arff`
3. Associate 탭 선택
4. Associator: `Apriori` 선택
5. 설정:
   - minSupport: `0.1`
   - minConfidence: `0.8`
   - numRules: `10`
6. Start 클릭
7. 스크린샷

**캡처할 내용:**
- Best rules found (상위 10개)
- Support, Confidence 값

---

### 4. Decision Tree (의사결정 트리)

**파일명:** `decision_tree.png`

**WEKA 실행 방법:**
1. WEKA Explorer 실행
2. Open file → `data/processed/bitcoin_classification.arff`
3. Classify 탭 선택
4. Classifier: `trees.J48` 선택
5. Start 클릭
6. 결과에서 우클릭 → `Visualize tree`
7. 트리 시각화 스크린샷

**캡처할 내용:**
- 전체 Decision Tree 구조

---

## 📝 스크린샷 가이드

### 권장 해상도
- 1920x1080 이상
- PNG 또는 JPG 형식

### 캡처 도구
- **Mac**: `Cmd + Shift + 4`
- **Windows**: `Win + Shift + S`
- **Linux**: `Screenshot` 앱

### 최소 필수 파일
1. `classification_result.png` - 필수!
2. `clustering_result.png` - 선택
3. `association_result.png` - 선택
4. `decision_tree.png` - 추천!

---

## 🚀 빠른 시작

### 1. WEKA에서 분석 실행
```bash
# WEKA가 설치되어 있어야 함
java -jar weka.jar
```

### 2. ARFF 파일 위치
```
data/processed/bitcoin_classification.arff
data/processed/bitcoin_clustering.arff
data/processed/bitcoin_association.arff
```

### 3. 스크린샷 저장
이 폴더(`weka_results/`)에 위 파일명으로 저장

### 4. 웹 대시보드 확인
```bash
streamlit run app.py
```

"WEKA Analysis" 탭에서 결과 확인!

---

## 💡 팁

1. **깔끔한 스크린샷**: 불필요한 부분은 잘라내기
2. **여러 알고리즘**: 다양한 분류기로 실험해보고 최고 결과 저장
3. **결과 비교**: Python 결과와 정확도 비교해보기

---

## 📊 예상 결과

### Classification (Random Forest)
```
Correctly Classified Instances: 82-85%
Confusion Matrix:
        UP  DOWN STABLE
UP      45   3     2
DOWN     2  38     5
STABLE   3   4    43
```

### Clustering (K-means, k=3)
```
Cluster 0: 65 instances (상승 패턴)
Cluster 1: 58 instances (하락 패턴)
Cluster 2: 56 instances (횡보 패턴)
```

### Association Rules
```
Rule 1: ma_cross=golden, rsi_signal=neutral => price_direction=UP
        Support: 0.25, Confidence: 0.82
Rule 2: ma_cross=dead, volume_spike=high => price_direction=DOWN
        Support: 0.18, Confidence: 0.78
```

---

**이 파일들이 있으면 웹 대시보드에서 자동으로 표시됩니다!**
