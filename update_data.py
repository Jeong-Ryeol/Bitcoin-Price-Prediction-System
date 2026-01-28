#!/usr/bin/env python3
"""
비트코인 데이터 자동 업데이트 스크립트
기존 데이터에 최신 데이터만 추가하고 모델 재학습
cron job으로 매시간 실행 권장

사용법:
  python update_data.py              # 데이터만 업데이트 (모델 재학습 안함)
  python update_data.py --retrain    # 데이터 업데이트 + SVM 모델 재학습
  python update_data.py --retrain --model lstm  # LSTM 모델 재학습
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# 경로 추가
sys.path.append('src')

from collector import BitcoinDataCollector
from chart_analyzer import ChartAnalyzer
from arff_generator import ARFFGenerator
from predictor import BitcoinPredictor
from config import load_config


def update_data_only(config):
    """데이터만 업데이트 (모델 재학습 없음)"""
    collector = BitcoinDataCollector()

    labeled_path = 'data/raw/bitcoin_labeled.csv'

    if not os.path.exists(labeled_path):
        print("   ⚠️ 기존 데이터 없음. 'python run.py'를 먼저 실행하세요.")
        return None

    # 기존 데이터 로드
    df_existing = pd.read_csv(labeled_path)
    df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])
    last_timestamp = df_existing['timestamp'].max()
    print(f"   기존 데이터: {len(df_existing)}개 (마지막: {last_timestamp})")

    # 최신 캔들 수집
    df_new = collector.get_candles_hour(count=200)

    if df_new is None:
        print("   ❌ API 호출 실패")
        return df_existing

    # 기존 이후의 새 데이터만
    df_new = df_new[df_new['timestamp'] > last_timestamp]

    if len(df_new) == 0:
        print("   ℹ️ 새 데이터 없음")
        return df_existing

    # 다중 시간대 라벨 추가
    df_new = collector.add_multi_horizon_labels(df_new, config)

    # 합치기
    df = pd.concat([df_existing, df_new], ignore_index=True)
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    print(f"   +{len(df_new)}개 추가 -> 총 {len(df)}개")

    # 저장
    df.to_csv(labeled_path, index=False)

    # 피처 계산
    analyzer = ChartAnalyzer(df)
    analyzer.calculate_technical_indicators()
    analyzer.detect_patterns()
    feature_df = analyzer.get_feature_dataset()

    os.makedirs('data/processed', exist_ok=True)
    feature_df.to_csv('data/processed/bitcoin_features.csv', index=False)

    # ARFF 업데이트
    arff_generator = ARFFGenerator(feature_df)
    arff_generator.save_all_formats()

    return feature_df


def retrain_svm(feature_df):
    """SVM 모델 재학습"""
    from sklearn.model_selection import train_test_split

    predictor = BitcoinPredictor()
    X, y = predictor.prepare_data(feature_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    predictor.train(X_train, y_train, model_type='svm')
    evaluation = predictor.evaluate(X_train, X_test, y_train, y_test)
    predictor.save_model()

    return evaluation


def retrain_lstm(feature_df, config):
    """LSTM + XGBoost 모델 재학습"""
    try:
        from predictor import HybridPredictor
        from sequence_generator import SequenceGenerator
        from lstm_model import LSTMXGBoostHybrid
    except ImportError as e:
        print(f"   ❌ LSTM 모델 사용 불가: {e}")
        print("   tensorflow와 xgboost를 설치해주세요:")
        print("   pip install tensorflow xgboost")
        return None

    # 시퀀스 생성
    seq_gen = SequenceGenerator(sequence_length=config['sequence_length'])
    X, y_dict = seq_gen.create_sequences(feature_df, config['prediction_horizons'])

    # 학습/테스트 분할 (시간순)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train = {h: y[:split_idx] for h, y in y_dict.items()}
    y_test = {h: y[split_idx:] for h, y in y_dict.items()}

    print(f"   학습 데이터: {len(X_train)}개")
    print(f"   테스트 데이터: {len(X_test)}개")

    # 하이브리드 모델 학습
    model = LSTMXGBoostHybrid(config)
    model.train(X_train, y_train)

    # 평가
    results = model.evaluate(X_test, y_test)

    # 전처리기 저장
    seq_gen.save_preprocessors('models/preprocessors.pkl')

    # 모델 저장
    model.save('models/lstm_xgboost_hybrid.pkl')

    # 1시간 예측 정확도 반환
    accuracy = results.get('1h', {}).get('accuracy', 0)
    return {'accuracy': accuracy}


def main():
    """데이터 업데이트 및 선택적 모델 재학습"""
    parser = argparse.ArgumentParser(description='Bitcoin Data Update Script')
    parser.add_argument(
        '--retrain', '-r',
        action='store_true',
        help='모델 재학습 여부 (기본: 데이터만 업데이트)'
    )
    parser.add_argument(
        '--model', '-m',
        choices=['svm', 'lstm'],
        default='svm',
        help='재학습할 모델 타입 (기본: svm)'
    )
    args = parser.parse_args()

    start_time = datetime.now()
    print(f"\n[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] 데이터 업데이트 시작")

    try:
        # 설정 로드
        config = load_config()

        # 데이터 업데이트
        feature_df = update_data_only(config)

        if feature_df is None:
            return

        # 모델 재학습 (선택적)
        if args.retrain:
            print(f"\n🤖 모델 재학습 ({args.model})")

            if args.model == 'lstm':
                evaluation = retrain_lstm(feature_df, config)
            else:
                evaluation = retrain_svm(feature_df)

            if evaluation:
                print(f"   정확도: {evaluation['accuracy']:.2%}")

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ 완료! ({elapsed:.1f}초)")

    except Exception as e:
        print(f"   ❌ 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
