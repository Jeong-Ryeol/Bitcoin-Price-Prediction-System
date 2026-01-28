# -*- coding: utf-8 -*-
"""
시퀀스 데이터 생성기
LSTM 입력용 시계열 윈도우 생성
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class SequenceGenerator:
    """시계열 시퀀스 데이터 생성기"""

    def __init__(self, sequence_length=72, feature_cols=None):
        """
        Args:
            sequence_length: 입력 시퀀스 길이 (기본 72시간 = 3일)
            feature_cols: 사용할 피처 컬럼 리스트
        """
        self.seq_len = sequence_length
        self.feature_cols = feature_cols or [
            'open', 'high', 'low', 'close', 'volume',
            'ma_cross', 'rsi_signal', 'volume_spike'
        ]
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_cols = ['ma_cross', 'rsi_signal', 'volume_spike']
        self.numeric_cols = ['open', 'high', 'low', 'close', 'volume']

    def prepare_features(self, df):
        """피처 전처리 (인코딩 + 스케일링)"""
        df_encoded = df.copy()

        # 범주형 변수 인코딩
        for col in self.categorical_cols:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(
                        df_encoded[col].astype(str)
                    )
                else:
                    # 학습된 인코더 사용
                    df_encoded[col] = self.label_encoders[col].transform(
                        df_encoded[col].astype(str)
                    )

        # 사용할 피처만 선택
        available_cols = [c for c in self.feature_cols if c in df_encoded.columns]
        features = df_encoded[available_cols].values

        return features, available_cols

    def create_sequences(self, df, horizons={'1h': 1, '24h': 24, '7d': 168}, fit_scaler=True):
        """
        시계열 시퀀스 데이터 생성

        Args:
            df: 입력 DataFrame (timestamp, features, labels 포함)
            horizons: 예측 시간대 딕셔너리 {이름: 시간}
            fit_scaler: 스케일러 학습 여부 (True=학습, False=변환만)

        Returns:
            X: 시퀀스 배열 (samples, seq_len, features)
            y_dict: 라벨 딕셔너리 {horizon: labels}
        """
        # 피처 준비
        features, used_cols = self.prepare_features(df)

        # 스케일링
        if fit_scaler:
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)

        # 시퀀스 생성
        X = []
        y_dict = {h: [] for h in horizons.keys()}

        max_horizon = max(horizons.values())
        n_samples = len(df) - self.seq_len - max_horizon

        if n_samples <= 0:
            raise ValueError(f"데이터가 부족합니다. 최소 {self.seq_len + max_horizon + 1}개 필요")

        for i in range(n_samples):
            # 입력 시퀀스: i ~ i+seq_len
            seq = features_scaled[i:i + self.seq_len]
            X.append(seq)

            # 각 시간대별 라벨
            for name, hours in horizons.items():
                label_col = f'direction_{hours}h'
                if label_col in df.columns:
                    label_idx = i + self.seq_len
                    if label_idx < len(df):
                        y_dict[name].append(df[label_col].iloc[label_idx])

        X = np.array(X)

        # 라벨 인코딩
        for name in y_dict:
            if y_dict[name]:
                y_dict[name] = np.array(y_dict[name])

        print(f"\n시퀀스 생성 완료:")
        print(f"  - X shape: {X.shape}")
        print(f"  - 시퀀스 길이: {self.seq_len}시간")
        print(f"  - 피처 수: {len(used_cols)}")
        for name, labels in y_dict.items():
            if len(labels) > 0:
                unique, counts = np.unique(labels, return_counts=True)
                print(f"  - {name} 라벨: {len(labels)}개 ({dict(zip(unique, counts))})")

        return X, y_dict

    def create_single_sequence(self, df):
        """
        단일 예측용 시퀀스 생성 (최근 데이터로)

        Args:
            df: 최근 데이터 DataFrame (최소 seq_len 행 필요)

        Returns:
            X: 시퀀스 배열 (1, seq_len, features)
        """
        if len(df) < self.seq_len:
            raise ValueError(f"최소 {self.seq_len}개 데이터 필요 (현재 {len(df)}개)")

        # 최근 seq_len개 데이터 사용
        recent_df = df.tail(self.seq_len).copy()

        # 피처 준비
        features, _ = self.prepare_features(recent_df)

        # 스케일링 (학습된 스케일러 사용)
        features_scaled = self.scaler.transform(features)

        # 배치 차원 추가
        X = features_scaled.reshape(1, self.seq_len, -1)

        return X

    def get_feature_names(self):
        """사용 중인 피처 이름 반환"""
        return self.feature_cols.copy()

    def save_preprocessors(self, filepath):
        """전처리기 저장 (스케일러, 인코더)"""
        import pickle
        preprocessors = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'seq_len': self.seq_len,
            'feature_cols': self.feature_cols
        }
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessors, f)
        print(f"전처리기 저장: {filepath}")

    def load_preprocessors(self, filepath):
        """전처리기 로드"""
        import pickle
        with open(filepath, 'rb') as f:
            preprocessors = pickle.load(f)
        self.scaler = preprocessors['scaler']
        self.label_encoders = preprocessors['label_encoders']
        self.seq_len = preprocessors['seq_len']
        self.feature_cols = preprocessors['feature_cols']
        print(f"전처리기 로드: {filepath}")


def create_train_test_sequences(df, config, test_ratio=0.2):
    """
    학습/테스트 시퀀스 분할 생성

    Args:
        df: 전체 데이터
        config: 설정 딕셔너리
        test_ratio: 테스트 데이터 비율

    Returns:
        X_train, X_test, y_train_dict, y_test_dict
    """
    seq_gen = SequenceGenerator(
        sequence_length=config['sequence_length'],
        feature_cols=config['features']['numeric'] + config['features']['categorical']
    )

    # 시간순 분할 (미래 데이터 누출 방지)
    split_idx = int(len(df) * (1 - test_ratio))
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx - config['sequence_length']:]  # 시퀀스 길이만큼 겹침

    # 학습 데이터로 스케일러 학습
    X_train, y_train_dict = seq_gen.create_sequences(
        df_train, config['prediction_horizons'], fit_scaler=True
    )

    # 테스트 데이터는 학습된 스케일러 사용
    X_test, y_test_dict = seq_gen.create_sequences(
        df_test, config['prediction_horizons'], fit_scaler=False
    )

    print(f"\n학습/테스트 분할:")
    print(f"  - 학습: {len(X_train)}개")
    print(f"  - 테스트: {len(X_test)}개")

    return X_train, X_test, y_train_dict, y_test_dict, seq_gen


if __name__ == "__main__":
    # 테스트
    print("시퀀스 생성기 테스트")

    # 샘플 데이터 생성
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='h'),
        'open': np.random.randn(n).cumsum() + 100000000,
        'high': np.random.randn(n).cumsum() + 100100000,
        'low': np.random.randn(n).cumsum() + 99900000,
        'close': np.random.randn(n).cumsum() + 100000000,
        'volume': np.abs(np.random.randn(n)) * 100,
        'ma_cross': np.random.choice(['golden', 'dead', 'neutral'], n),
        'rsi_signal': np.random.choice(['overbought', 'oversold', 'neutral'], n),
        'volume_spike': np.random.choice(['high', 'normal', 'low'], n),
        'direction_1h': np.random.choice(['UP', 'DOWN', 'STABLE'], n),
        'direction_24h': np.random.choice(['UP', 'DOWN', 'STABLE'], n),
        'direction_168h': np.random.choice(['UP', 'DOWN', 'STABLE'], n),
    })

    # 시퀀스 생성
    seq_gen = SequenceGenerator(sequence_length=72)
    X, y_dict = seq_gen.create_sequences(df, {'1h': 1, '24h': 24, '7d': 168})

    print(f"\nX shape: {X.shape}")
    print(f"y keys: {y_dict.keys()}")
