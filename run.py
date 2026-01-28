"""
Bitcoin Price Prediction System - 전체 파이프라인 실행 스크립트
데이터 수집부터 모델 학습까지 자동화
SVM (기본) 또는 LSTM+XGBoost (고급) 선택 가능
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
from config import load_config, save_config, DEFAULT_CONFIG


def print_header(title):
    """섹션 헤더 출력"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def train_svm_model(feature_df, config):
    """SVM 모델 학습 (기존 방식)"""
    from predictor import BitcoinPredictor, compare_models
    from sklearn.model_selection import train_test_split

    predictor = BitcoinPredictor()

    # 데이터 준비
    X, y = predictor.prepare_data(feature_df)
    print(f"\n데이터 준비 완료:")
    print(f"   - 인스턴스: {len(X)}개")
    print(f"   - 피처: {X.shape[1]}개")
    print(f"   - 클래스: {list(set(y))}")

    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n   학습 데이터: {len(X_train)}개")
    print(f"   테스트 데이터: {len(X_test)}개")

    # 여러 모델 비교
    print(f"\n🔬 여러 모델 비교 중...")
    results, best_model = compare_models(X_train, X_test, y_train, y_test)

    # 최고 모델로 학습
    model_type_map = {
        'Random Forest': 'random_forest',
        'Decision Tree': 'decision_tree',
        'Decision Tree (J48)': 'decision_tree',
        'Naive Bayes': 'naive_bayes',
        'SVM': 'svm'
    }

    predictor.train(X_train, y_train, model_type=model_type_map.get(best_model, 'svm'))

    # 평가
    evaluation = predictor.evaluate(X_train, X_test, y_train, y_test)

    # 모델 저장
    predictor.save_model()

    return evaluation, best_model


def train_lstm_model(feature_df, config):
    """LSTM + XGBoost 하이브리드 모델 학습"""
    try:
        from predictor import HybridPredictor
        from sequence_generator import SequenceGenerator
    except ImportError as e:
        print(f"❌ LSTM 모델 학습 실패: {e}")
        print("   tensorflow와 xgboost를 설치해주세요:")
        print("   pip install tensorflow xgboost")
        return None, None

    print("\n🧠 LSTM + XGBoost 하이브리드 모델 학습")

    # 시퀀스 생성기
    seq_gen = SequenceGenerator(
        sequence_length=config['sequence_length']
    )

    # 시퀀스 데이터 생성
    X, y_dict = seq_gen.create_sequences(feature_df, config['prediction_horizons'])

    # 학습/테스트 분할 (시간순)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train = {h: y[:split_idx] for h, y in y_dict.items()}
    y_test = {h: y[split_idx:] for h, y in y_dict.items()}

    print(f"\n   학습 데이터: {len(X_train)}개")
    print(f"   테스트 데이터: {len(X_test)}개")

    # 하이브리드 모델 학습
    hybrid = HybridPredictor(config)
    hybrid.model = None  # 새로 학습

    from lstm_model import LSTMXGBoostHybrid
    hybrid.model = LSTMXGBoostHybrid(config)
    hybrid.model.train(X_train, y_train)

    # 평가
    results = hybrid.model.evaluate(X_test, y_test)

    # 전처리기 저장
    seq_gen.save_preprocessors('models/preprocessors.pkl')

    # 모델 저장
    hybrid.model.save('models/lstm_xgboost_hybrid.pkl')

    # 1시간 예측 정확도 반환
    accuracy = results.get('1h', {}).get('accuracy', 0)
    evaluation = {'accuracy': accuracy}

    return evaluation, 'LSTM+XGBoost'


def main(model_type='auto'):
    """
    전체 파이프라인 실행

    Args:
        model_type: 'svm', 'lstm', 또는 'auto' (자동 선택)
    """
    start_time = datetime.now()

    print("=" * 70)
    print("  🚀 Bitcoin Price Prediction System v2.0")
    print("  전체 파이프라인 자동 실행")
    print("=" * 70)
    print(f"\n시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"모델 타입: {model_type}")

    # 설정 로드
    config = load_config()
    print(f"\n설정 로드 완료:")
    print(f"   - 시퀀스 길이: {config['sequence_length']}시간")
    print(f"   - 예측 시간대: {list(config['prediction_horizons'].keys())}")
    print(f"   - 임계값: {config['thresholds']}")

    try:
        # ==========================================
        # Step 1: 데이터 수집
        # ==========================================
        print_header("📊 Step 1: 데이터 수집")

        collector = BitcoinDataCollector()

        # 3년치 데이터 수집
        df = collector.collect_yearly_data(days=365*3)

        if df is None or len(df) == 0:
            print("\n❌ 데이터 수집 실패. 프로그램을 종료합니다.")
            return

        # 다중 시간대 라벨 추가
        df = collector.add_multi_horizon_labels(df, config)

        # 저장
        labeled_path = 'data/raw/bitcoin_labeled.csv'
        df.to_csv(labeled_path, index=False)
        print(f"\n✅ Step 1 완료: {labeled_path}")

        # ==========================================
        # Step 2: 차트 패턴 분석
        # ==========================================
        print_header("📈 Step 2: 차트 패턴 분석")

        analyzer = ChartAnalyzer(df)
        analyzer.calculate_technical_indicators()
        analyzer.detect_patterns()

        # 피처 데이터셋 저장
        feature_df = analyzer.get_feature_dataset()
        feature_path = 'data/processed/bitcoin_features.csv'
        os.makedirs('data/processed', exist_ok=True)
        feature_df.to_csv(feature_path, index=False)
        print(f"\n✅ Step 2 완료: {feature_path}")

        # 샘플 차트 생성 (최근 5개)
        print(f"\n🖼️  샘플 차트 생성 중...")
        for i in range(max(0, len(feature_df) - 5), len(feature_df)):
            try:
                filepath = analyzer.create_chart_image(i)
                print(f"   - {os.path.basename(filepath)}")
            except Exception as e:
                print(f"   ⚠️  차트 생성 실패: {e}")

        # ==========================================
        # Step 3: ARFF 파일 생성
        # ==========================================
        print_header("📁 Step 3: WEKA ARFF 파일 생성")

        arff_generator = ARFFGenerator(feature_df)
        arff_files = arff_generator.save_all_formats()

        print(f"\n✅ Step 3 완료: {len(arff_files)}개 파일 생성")

        # ==========================================
        # Step 4: 머신러닝 모델 학습
        # ==========================================
        print_header("🤖 Step 4: 머신러닝 모델 학습")

        # 모델 타입 결정
        if model_type == 'auto':
            # LSTM 가능한지 확인
            try:
                import tensorflow
                import xgboost
                model_type = 'lstm'
                print("✅ TensorFlow & XGBoost 감지됨 → LSTM 모델 사용")
            except ImportError:
                model_type = 'svm'
                print("⚠️ TensorFlow/XGBoost 없음 → SVM 모델 사용")

        if model_type == 'lstm':
            evaluation, best_model = train_lstm_model(feature_df, config)
            if evaluation is None:
                print("⚠️ LSTM 실패, SVM으로 폴백")
                evaluation, best_model = train_svm_model(feature_df, config)
        else:
            evaluation, best_model = train_svm_model(feature_df, config)

        print(f"\n✅ Step 4 완료: 모델 학습 및 저장")

        # ==========================================
        # 최종 요약
        # ==========================================
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        print("\n" + "=" * 70)
        print("  ✨ 전체 파이프라인 완료!")
        print("=" * 70)

        print(f"\n📊 최종 결과:")
        print(f"   - 총 인스턴스: {len(feature_df)}개")
        print(f"   - 모델 정확도: {evaluation['accuracy']:.2%}")
        print(f"   - 사용 모델: {best_model}")

        print(f"\n📁 생성된 파일:")
        print(f"   - {labeled_path}")
        print(f"   - {feature_path}")
        for f in arff_files:
            print(f"   - {f}")
        if model_type == 'lstm':
            print(f"   - models/lstm_xgboost_hybrid.pkl")
            print(f"   - models/preprocessors.pkl")
        else:
            print(f"   - models/bitcoin_predictor.pkl")

        print(f"\n⏱️  총 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")

        print("\n" + "=" * 70)
        print("  💡 다음 단계")
        print("=" * 70)
        print("\n1. 웹 대시보드 실행:")
        print("   streamlit run app.py")

        print("\n2. 설정 변경:")
        print("   data/config.json 파일에서 임계값 등 수정 가능")

        print("\n" + "=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        sys.exit(1)

    except Exception as e:
        print(f"\n\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bitcoin Price Prediction System')
    parser.add_argument(
        '--model', '-m',
        choices=['svm', 'lstm', 'auto'],
        default='auto',
        help='모델 타입 선택 (기본: auto)'
    )
    args = parser.parse_args()

    main(model_type=args.model)
