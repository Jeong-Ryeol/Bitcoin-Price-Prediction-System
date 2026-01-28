#!/usr/bin/env python3
"""
비트코인 데이터 자동 업데이트 스크립트
기존 데이터에 최신 데이터만 추가하고 모델 재학습
cron job으로 매시간 실행 권장
"""

import os
import sys
import pandas as pd
from datetime import datetime

# 경로 추가
sys.path.append('src')

from collector import BitcoinDataCollector
from chart_analyzer import ChartAnalyzer
from arff_generator import ARFFGenerator
from predictor import BitcoinPredictor


def main():
    """데이터 업데이트 및 모델 재학습"""
    start_time = datetime.now()
    print(f"\n[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] 데이터 업데이트 시작")

    try:
        collector = BitcoinDataCollector()

        # 1. 기존 데이터에 최신 데이터 추가
        labeled_path = 'data/raw/bitcoin_labeled.csv'

        if os.path.exists(labeled_path):
            df_existing = pd.read_csv(labeled_path)
            df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])
            last_timestamp = df_existing['timestamp'].max()
            print(f"   기존 데이터: {len(df_existing)}개 (마지막: {last_timestamp})")

            # 최신 캔들 수집
            df_new = collector.get_candles_hour(count=200)

            if df_new is not None:
                # 기존 이후의 새 데이터만
                df_new = df_new[df_new['timestamp'] > last_timestamp]

                if len(df_new) > 0:
                    # 라벨 추가
                    df_new = collector.add_future_labels(df_new, hours_ahead=1)

                    # 합치기
                    df = pd.concat([df_existing, df_new], ignore_index=True)
                    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

                    print(f"   +{len(df_new)}개 추가 -> 총 {len(df)}개")
                else:
                    print("   새 데이터 없음")
                    df = df_existing
            else:
                print("   API 호출 실패")
                df = df_existing
        else:
            print("   기존 데이터 없음. 전체 수집 필요.")
            print("   'python run.py' 를 먼저 실행하세요.")
            return

        # 2. 저장
        df.to_csv(labeled_path, index=False)

        # 3. 피처 계산
        analyzer = ChartAnalyzer(df)
        analyzer.calculate_technical_indicators()
        analyzer.detect_patterns()
        feature_df = analyzer.get_feature_dataset()
        feature_df.to_csv('data/processed/bitcoin_features.csv', index=False)

        # 4. ARFF 업데이트
        arff_generator = ARFFGenerator(feature_df)
        arff_generator.save_all_formats()

        # 5. 모델 재학습
        predictor = BitcoinPredictor()
        X, y = predictor.prepare_data(feature_df)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        predictor.train(X_train, y_train, model_type='svm')
        evaluation = predictor.evaluate(X_train, X_test, y_train, y_test)
        predictor.save_model()

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"   완료! 정확도: {evaluation['accuracy']:.2%} ({elapsed:.1f}초)")

    except Exception as e:
        print(f"   오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
