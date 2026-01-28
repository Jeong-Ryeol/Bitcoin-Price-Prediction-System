# -*- coding: utf-8 -*-
"""
LSTM + XGBoost 하이브리드 모델
시계열 특성 추출 (LSTM) + 분류 (XGBoost) 조합
"""

import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# TensorFlow 경고 억제
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow를 찾을 수 없습니다. LSTM 기능이 비활성화됩니다.")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost를 찾을 수 없습니다. XGBoost 기능이 비활성화됩니다.")


class LSTMFeatureExtractor:
    """LSTM 기반 시계열 피처 추출기"""

    def __init__(self, sequence_length, n_features, hidden_units=64):
        self.seq_len = sequence_length
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.model = None
        self.feature_extractor = None

    def build(self):
        """LSTM 모델 구축"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow가 필요합니다")

        inputs = Input(shape=(self.seq_len, self.n_features))

        # Bidirectional LSTM 레이어
        x = Bidirectional(LSTM(self.hidden_units, return_sequences=True))(inputs)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(self.hidden_units // 2, return_sequences=False))(x)
        x = Dropout(0.2)(x)

        # 피처 추출 레이어
        features = Dense(32, activation='relu', name='features')(x)

        # 출력 레이어 (학습용 - 3클래스 분류)
        outputs = Dense(3, activation='softmax', name='output')(features)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # 피처 추출기 (중간 레이어까지)
        self.feature_extractor = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('features').output
        )

        return self.model

    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """모델 학습"""
        if self.model is None:
            self.build()

        # 조기 종료 콜백
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1
            )
        ]

        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def extract_features(self, X):
        """LSTM 피처 추출"""
        if self.feature_extractor is None:
            raise ValueError("모델이 학습되지 않았습니다")
        return self.feature_extractor.predict(X, verbose=0)


class LSTMXGBoostHybrid:
    """LSTM + XGBoost 하이브리드 모델"""

    def __init__(self, config):
        self.config = config
        self.seq_len = config.get('sequence_length', 72)
        self.horizons = config.get('prediction_horizons', {'1h': 1, '24h': 24, '7d': 168})

        self.lstm_extractor = None
        self.xgb_models = {}  # 각 시간대별 XGBoost 모델
        self.label_encoders = {}  # 각 시간대별 라벨 인코더

        self.n_features = None
        self.is_trained = False

    def _encode_labels(self, y_dict):
        """라벨 인코딩"""
        encoded = {}
        for horizon, labels in y_dict.items():
            if horizon not in self.label_encoders:
                self.label_encoders[horizon] = LabelEncoder()
                encoded[horizon] = self.label_encoders[horizon].fit_transform(labels)
            else:
                encoded[horizon] = self.label_encoders[horizon].transform(labels)
        return encoded

    def _decode_labels(self, horizon, encoded):
        """라벨 디코딩"""
        return self.label_encoders[horizon].inverse_transform(encoded)

    def train(self, X, y_dict, epochs=50, batch_size=32):
        """
        하이브리드 모델 학습

        Args:
            X: 시퀀스 데이터 (samples, seq_len, features)
            y_dict: 라벨 딕셔너리 {horizon: labels}
            epochs: LSTM 학습 에폭
            batch_size: 배치 크기
        """
        print("\n" + "=" * 70)
        print("🤖 LSTM + XGBoost 하이브리드 모델 학습")
        print("=" * 70)

        self.n_features = X.shape[2]

        # 라벨 인코딩
        y_encoded = self._encode_labels(y_dict)

        # 1단계: LSTM 피처 추출기 학습 (1시간 예측 기준)
        print("\n📊 Step 1: LSTM 피처 추출기 학습")
        primary_horizon = '1h' if '1h' in y_encoded else list(y_encoded.keys())[0]

        self.lstm_extractor = LSTMFeatureExtractor(
            sequence_length=self.seq_len,
            n_features=self.n_features,
            hidden_units=64
        )
        self.lstm_extractor.build()

        training_config = self.config.get('training', {})
        self.lstm_extractor.train(
            X, y_encoded[primary_horizon],
            epochs=training_config.get('epochs', epochs),
            batch_size=training_config.get('batch_size', batch_size),
            validation_split=training_config.get('validation_split', 0.2)
        )

        # 2단계: LSTM 피처 추출
        print("\n📊 Step 2: LSTM 피처 추출")
        lstm_features = self.lstm_extractor.extract_features(X)
        print(f"   추출된 피처 shape: {lstm_features.shape}")

        # 3단계: 각 시간대별 XGBoost 학습
        print("\n📊 Step 3: XGBoost 분류기 학습")
        for horizon, labels in y_encoded.items():
            print(f"\n   [{horizon}] XGBoost 학습 중...")

            self.xgb_models[horizon] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )

            self.xgb_models[horizon].fit(lstm_features, labels)

            # 학습 정확도
            train_pred = self.xgb_models[horizon].predict(lstm_features)
            train_acc = accuracy_score(labels, train_pred)
            print(f"   [{horizon}] 학습 정확도: {train_acc:.2%}")

        self.is_trained = True
        print("\n✅ 하이브리드 모델 학습 완료")

    def predict(self, X):
        """
        예측 수행

        Args:
            X: 시퀀스 데이터 (samples, seq_len, features)

        Returns:
            dict: {horizon: {'direction': [...], 'probability': [...]}}
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다")

        # LSTM 피처 추출
        lstm_features = self.lstm_extractor.extract_features(X)

        results = {}
        for horizon, model in self.xgb_models.items():
            pred_encoded = model.predict(lstm_features)
            pred_proba = model.predict_proba(lstm_features)

            pred_labels = self._decode_labels(horizon, pred_encoded)
            confidence = np.max(pred_proba, axis=1)

            results[horizon] = {
                'direction': pred_labels,
                'probability': pred_proba,
                'confidence': confidence
            }

        return results

    def predict_single(self, X):
        """
        단일 예측 (최근 데이터 1개)

        Args:
            X: 시퀀스 데이터 (1, seq_len, features)

        Returns:
            dict: {horizon: {'direction': str, 'confidence': float, 'probabilities': dict}}
        """
        if X.shape[0] != 1:
            X = X[-1:, :, :]  # 마지막 시퀀스만

        raw_results = self.predict(X)

        results = {}
        for horizon, data in raw_results.items():
            direction = data['direction'][0]
            confidence = data['confidence'][0]

            # 클래스별 확률
            classes = self.label_encoders[horizon].classes_
            probabilities = dict(zip(classes, data['probability'][0]))

            results[horizon] = {
                'direction': direction,
                'confidence': confidence,
                'probabilities': probabilities
            }

        return results

    def evaluate(self, X_test, y_test_dict):
        """모델 평가"""
        print("\n" + "=" * 70)
        print("📊 모델 평가")
        print("=" * 70)

        y_encoded = self._encode_labels(y_test_dict)
        predictions = self.predict(X_test)

        results = {}
        for horizon in self.horizons.keys():
            if horizon not in y_encoded:
                continue

            y_true = y_test_dict[horizon]
            y_pred = predictions[horizon]['direction']

            accuracy = accuracy_score(y_true, y_pred)
            print(f"\n[{horizon}] 정확도: {accuracy:.2%}")

            print(f"\n혼동 행렬:")
            cm = confusion_matrix(y_true, y_pred)
            print(cm)

            print(f"\n분류 리포트:")
            print(classification_report(y_true, y_pred))

            results[horizon] = {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'predictions': y_pred
            }

        return results

    def save(self, filepath='models/lstm_xgboost_hybrid.pkl'):
        """모델 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # LSTM 모델 따로 저장
        lstm_path = filepath.replace('.pkl', '_lstm.keras')
        if self.lstm_extractor and self.lstm_extractor.model:
            self.lstm_extractor.model.save(lstm_path)

        # XGBoost 및 설정 저장
        save_data = {
            'config': self.config,
            'xgb_models': self.xgb_models,
            'label_encoders': self.label_encoders,
            'seq_len': self.seq_len,
            'n_features': self.n_features,
            'horizons': self.horizons,
            'lstm_path': lstm_path
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"✅ 모델 저장 완료: {filepath}")

    def load(self, filepath='models/lstm_xgboost_hybrid.pkl'):
        """모델 로드"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        self.config = save_data['config']
        self.xgb_models = save_data['xgb_models']
        self.label_encoders = save_data['label_encoders']
        self.seq_len = save_data['seq_len']
        self.n_features = save_data['n_features']
        self.horizons = save_data['horizons']

        # LSTM 모델 로드
        lstm_path = save_data.get('lstm_path')
        if lstm_path and os.path.exists(lstm_path):
            self.lstm_extractor = LSTMFeatureExtractor(
                sequence_length=self.seq_len,
                n_features=self.n_features
            )
            self.lstm_extractor.model = keras.models.load_model(lstm_path)
            self.lstm_extractor.feature_extractor = Model(
                inputs=self.lstm_extractor.model.input,
                outputs=self.lstm_extractor.model.get_layer('features').output
            )

        self.is_trained = True
        print(f"✅ 모델 로드 완료: {filepath}")


def check_dependencies():
    """의존성 확인"""
    status = {
        'tensorflow': TF_AVAILABLE,
        'xgboost': XGB_AVAILABLE
    }
    print("의존성 상태:")
    for name, available in status.items():
        status_str = "✅" if available else "❌"
        print(f"  {status_str} {name}")
    return all(status.values())


if __name__ == "__main__":
    print("LSTM + XGBoost 하이브리드 모델 테스트\n")

    if not check_dependencies():
        print("\n필요한 패키지를 설치해주세요:")
        print("  pip install tensorflow xgboost")
        exit(1)

    # 샘플 데이터 생성
    np.random.seed(42)
    n_samples = 1000
    seq_len = 72
    n_features = 8

    X = np.random.randn(n_samples, seq_len, n_features)
    y_dict = {
        '1h': np.random.choice(['UP', 'DOWN', 'STABLE'], n_samples),
        '24h': np.random.choice(['UP', 'DOWN', 'STABLE'], n_samples),
        '7d': np.random.choice(['UP', 'DOWN', 'STABLE'], n_samples)
    }

    # 학습/테스트 분할
    split = int(n_samples * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train = {h: y[:split] for h, y in y_dict.items()}
    y_test = {h: y[split:] for h, y in y_dict.items()}

    # 설정
    config = {
        'sequence_length': seq_len,
        'prediction_horizons': {'1h': 1, '24h': 24, '7d': 168},
        'training': {
            'epochs': 5,  # 테스트용으로 작게
            'batch_size': 32
        }
    }

    # 모델 학습
    model = LSTMXGBoostHybrid(config)
    model.train(X_train, y_train, epochs=5)

    # 평가
    model.evaluate(X_test, y_test)

    # 단일 예측 테스트
    print("\n단일 예측 테스트:")
    single_pred = model.predict_single(X_test[:1])
    for horizon, result in single_pred.items():
        print(f"  [{horizon}] {result['direction']} (신뢰도: {result['confidence']:.2%})")

    # 저장/로드 테스트
    model.save('models/test_hybrid.pkl')

    model2 = LSTMXGBoostHybrid(config)
    model2.load('models/test_hybrid.pkl')

    print("\n로드 후 예측:")
    single_pred2 = model2.predict_single(X_test[:1])
    for horizon, result in single_pred2.items():
        print(f"  [{horizon}] {result['direction']} (신뢰도: {result['confidence']:.2%})")
