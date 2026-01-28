"""
머신러닝 예측 모델 모듈
SVM (기본) + LSTM+XGBoost 하이브리드 (고급) 지원
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

try:
    from mlxtend.classifier import OneRClassifier
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

# LSTM 모델 임포트 시도
try:
    from lstm_model import LSTMXGBoostHybrid, check_dependencies
    LSTM_AVAILABLE = check_dependencies()
except ImportError:
    LSTM_AVAILABLE = False

try:
    from sequence_generator import SequenceGenerator, create_train_test_sequences
    SEQUENCE_AVAILABLE = True
except ImportError:
    SEQUENCE_AVAILABLE = False


class BitcoinPredictor:
    """비트코인 가격 방향 예측 모델"""

    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None

    def prepare_data(self, df):
        """
        학습용 데이터 준비

        Args:
            df: 피처 데이터프레임

        Returns:
            X: 피처 행렬
            y: 타겟 벡터
        """
        # 피처 선택
        feature_cols = ['open', 'high', 'low', 'close', 'volume',
                        'ma_cross', 'rsi_signal', 'volume_spike']

        # 범주형 변수 인코딩
        df_encoded = df.copy()

        # LabelEncoder for categorical features
        for col in ['ma_cross', 'rsi_signal', 'volume_spike']:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])

        X = df_encoded[feature_cols].values
        y = df['price_direction'].values

        self.feature_names = feature_cols

        return X, y

    def train(self, X, y, model_type='random_forest'):
        """
        모델 학습

        Args:
            X: 피처 행렬
            y: 타겟 벡터
            model_type: 'oner', 'random_forest', 'decision_tree', 'naive_bayes', 'svm'

        Returns:
            학습된 모델
        """
        print(f"\n🤖 모델 학습 시작: {model_type}")
        print("=" * 70)

        # 모델 선택
        if model_type == 'oner':
            self.model = OneRClassifier()
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(max_depth=10, random_state=42)
        elif model_type == 'naive_bayes':
            self.model = GaussianNB()
        elif model_type == 'svm':
            self.model = SVC(kernel='rbf', probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # 학습
        self.model.fit(X, y)

        print(f"✅ 학습 완료: {model_type}")

        return self.model

    def evaluate(self, X_train, X_test, y_train, y_test):
        """
        모델 평가

        Args:
            X_train, X_test, y_train, y_test: 학습/테스트 데이터

        Returns:
            dict: 평가 결과
        """
        print("\n📊 모델 평가")
        print("=" * 70)

        # 예측
        y_pred = self.model.predict(X_test)

        # 정확도
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n정확도: {accuracy:.2%}")

        # 혼동 행렬
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n혼동 행렬:")
        print(cm)

        # 분류 리포트
        print(f"\n분류 리포트:")
        print(classification_report(y_test, y_pred))

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"\nCross-Validation 정확도: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")

        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'cv_scores': cv_scores
        }

    def predict(self, features):
        """
        새로운 데이터 예측

        Args:
            features: 피처 딕셔너리 또는 배열

        Returns:
            dict: 예측 결과 및 확률
        """
        if isinstance(features, dict):
            # 딕셔너리를 배열로 변환
            feature_array = np.array([[
                features['open'],
                features['high'],
                features['low'],
                features['close'],
                features['volume'],
                features['ma_cross'],  # 이미 인코딩된 값
                features['rsi_signal'],
                features['volume_spike']
            ]])
        else:
            feature_array = np.array([features])

        # 예측
        prediction = self.model.predict(feature_array)[0]
        probabilities = self.model.predict_proba(feature_array)[0]

        # 클래스별 확률
        classes = self.model.classes_
        prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}

        return {
            'prediction': prediction,
            'probabilities': prob_dict,
            'confidence': max(probabilities)
        }

    def save_model(self, filepath='models/bitcoin_predictor.pkl'):
        """모델 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names
            }, f)

        print(f"\n💾 모델 저장 완료: {filepath}")

    def load_model(self, filepath='models/bitcoin_predictor.pkl'):
        """모델 로드"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']

        print(f"\n✅ 모델 로드 완료: {filepath}")


class HybridPredictor:
    """LSTM + XGBoost 하이브리드 예측기 래퍼"""

    def __init__(self, config=None):
        if not LSTM_AVAILABLE:
            raise ImportError("LSTM 모델을 사용하려면 tensorflow와 xgboost가 필요합니다")

        self.config = config or {
            'sequence_length': 72,
            'prediction_horizons': {'1h': 1, '24h': 24, '7d': 168},
            'thresholds': {'1h': 0.3, '24h': 1.5, '7d': 5.0},
            'training': {'epochs': 50, 'batch_size': 32}
        }
        self.model = None
        self.seq_generator = None

    def prepare_data(self, df):
        """시퀀스 데이터 준비"""
        if not SEQUENCE_AVAILABLE:
            raise ImportError("sequence_generator 모듈이 필요합니다")

        self.seq_generator = SequenceGenerator(
            sequence_length=self.config['sequence_length']
        )

        X, y_dict = self.seq_generator.create_sequences(
            df, self.config['prediction_horizons']
        )

        return X, y_dict

    def train(self, X, y_dict, epochs=None, batch_size=None):
        """하이브리드 모델 학습"""
        self.model = LSTMXGBoostHybrid(self.config)

        training_epochs = epochs or self.config['training'].get('epochs', 50)
        training_batch = batch_size or self.config['training'].get('batch_size', 32)

        self.model.train(X, y_dict, epochs=training_epochs, batch_size=training_batch)

        return self.model

    def evaluate(self, X_test, y_test_dict):
        """모델 평가"""
        return self.model.evaluate(X_test, y_test_dict)

    def predict(self, X):
        """예측 (배치)"""
        return self.model.predict(X)

    def predict_single(self, X):
        """단일 예측"""
        return self.model.predict_single(X)

    def predict_realtime(self, df):
        """
        실시간 예측 (DataFrame 입력)

        Args:
            df: 최근 데이터 DataFrame (최소 sequence_length 행 필요)

        Returns:
            dict: 각 시간대별 예측 결과
        """
        if self.seq_generator is None:
            self.seq_generator = SequenceGenerator(
                sequence_length=self.config['sequence_length']
            )
            # 전처리기 로드 필요
            preprocessor_path = 'models/preprocessors.pkl'
            if os.path.exists(preprocessor_path):
                self.seq_generator.load_preprocessors(preprocessor_path)

        X = self.seq_generator.create_single_sequence(df)
        return self.model.predict_single(X)

    def save(self, filepath='models/lstm_xgboost_hybrid.pkl'):
        """모델 저장"""
        self.model.save(filepath)

        # 전처리기도 저장
        if self.seq_generator:
            preprocessor_path = filepath.replace('.pkl', '_preprocessors.pkl')
            self.seq_generator.save_preprocessors(preprocessor_path)

    def load(self, filepath='models/lstm_xgboost_hybrid.pkl'):
        """모델 로드"""
        self.model = LSTMXGBoostHybrid(self.config)
        self.model.load(filepath)

        # 전처리기 로드
        preprocessor_path = filepath.replace('.pkl', '_preprocessors.pkl')
        if os.path.exists(preprocessor_path):
            self.seq_generator = SequenceGenerator(
                sequence_length=self.config['sequence_length']
            )
            self.seq_generator.load_preprocessors(preprocessor_path)


def compare_models(X_train, X_test, y_train, y_test):
    """여러 모델 비교"""
    print("\n" + "=" * 70)
    print("🔬 여러 모델 비교")
    print("=" * 70)

    models = {
        'Naive Bayes': GaussianNB(),
        'Decision Tree (J48)': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"\n{name}: {accuracy:.2%}")

    # 최고 모델
    best_model = max(results, key=results.get)
    print(f"\n🏆 최고 모델: {best_model} ({results[best_model]:.2%})")

    return results, best_model


def main():
    """테스트 실행"""
    # 데이터 로드
    input_path = 'data/processed/bitcoin_features.csv'

    if not os.path.exists(input_path):
        print(f"❌ 파일이 없습니다: {input_path}")
        print("   먼저 src/chart_analyzer.py를 실행하세요.")
        return

    print("=" * 70)
    print("🤖 머신러닝 모델 학습")
    print("=" * 70)

    df = pd.read_csv(input_path)
    print(f"\n✅ 데이터 로드: {len(df)}개 인스턴스")

    # 예측 모델 생성
    predictor = BitcoinPredictor()

    # 데이터 준비
    X, y = predictor.prepare_data(df)
    print(f"   피처: {X.shape[1]}개")
    print(f"   클래스: {np.unique(y)}")

    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # stratify 제거 (클래스 불균형 대응)
    )

    print(f"\n   학습 데이터: {len(X_train)}개")
    print(f"   테스트 데이터: {len(X_test)}개")

    # 여러 모델 비교
    results, best_model = compare_models(X_train, X_test, y_train, y_test)

    # 최고 모델로 학습
    model_type_map = {
        'OneR': 'oner',
        'Naive Bayes': 'naive_bayes',
        'Decision Tree (J48)': 'decision_tree',
        'Random Forest': 'random_forest',
        'SVM': 'svm'
    }

    predictor.train(X_train, y_train, model_type=model_type_map[best_model])

    # 평가
    evaluation = predictor.evaluate(X_train, X_test, y_train, y_test)

    # 모델 저장
    predictor.save_model()

    # 예측 테스트
    print("\n" + "=" * 70)
    print("🔮 예측 테스트 (테스트 데이터 첫 5개)")
    print("=" * 70)

    for i in range(min(5, len(X_test))):
        result = predictor.predict(X_test[i])
        actual = y_test[i]

        print(f"\n샘플 {i+1}:")
        print(f"   예측: {result['prediction']} (신뢰도: {result['confidence']:.2%})")
        print(f"   실제: {actual}")
        print(f"   확률: {result['probabilities']}")


if __name__ == "__main__":
    main()
