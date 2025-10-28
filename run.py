"""
Bitcoin Price Prediction System - 전체 파이프라인 실행 스크립트
데이터 수집부터 모델 학습까지 자동화
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


def print_header(title):
    """섹션 헤더 출력"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    """전체 파이프라인 실행"""
    start_time = datetime.now()

    print("=" * 70)
    print("  🚀 Bitcoin Price Prediction System")
    print("  전체 파이프라인 자동 실행")
    print("=" * 70)
    print(f"\n시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # ==========================================
        # Step 1: 데이터 수집
        # ==========================================
        print_header("📊 Step 1: 데이터 수집")

        collector = BitcoinDataCollector()

        # 과거 500시간 데이터 수집
        df = collector.collect_historical_data(hours=500)

        if df is None or len(df) == 0:
            print("\n❌ 데이터 수집 실패. 프로그램을 종료합니다.")
            return

        # 1시간 후 가격 방향 라벨 추가
        df = collector.add_future_labels(df, hours_ahead=1)

        # 저장
        labeled_path = 'data/raw/bitcoin_labeled.csv'
        df.to_csv(labeled_path, index=False)
        print(f"\n✅ Step 1 완료: {labeled_path}")

        # ==========================================
        # Step 2: 차트 패턴 분석
        # ==========================================
        print_header("📈 Step 2: 차트 패턴 분석")

        analyzer = ChartAnalyzer(df)

        # 기술적 지표 계산
        analyzer.calculate_technical_indicators()

        # 패턴 인식
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

        predictor = BitcoinPredictor()

        # 데이터 준비
        X, y = predictor.prepare_data(feature_df)
        print(f"\n데이터 준비 완료:")
        print(f"   - 인스턴스: {len(X)}개")
        print(f"   - 피처: {X.shape[1]}개")
        print(f"   - 클래스: {list(set(y))}")

        # 학습/테스트 분할
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42  # stratify 제거 (클래스 불균형 대응)
        )

        print(f"\n   학습 데이터: {len(X_train)}개")
        print(f"   테스트 데이터: {len(X_test)}개")

        # 여러 모델 비교
        print(f"\n🔬 여러 모델 비교 중...")
        from predictor import compare_models
        results, best_model = compare_models(X_train, X_test, y_train, y_test)

        # 최고 모델로 학습
        model_type_map = {
            'Random Forest': 'random_forest',
            'Decision Tree': 'decision_tree',
            'Naive Bayes': 'naive_bayes',
            'SVM': 'svm'
        }

        predictor.train(X_train, y_train, model_type=model_type_map[best_model])

        # 평가
        evaluation = predictor.evaluate(X_train, X_test, y_train, y_test)

        # 모델 저장
        predictor.save_model()

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
        print(f"   - 피처: {X.shape[1]}개")
        print(f"   - 모델 정확도: {evaluation['accuracy']:.2%}")
        print(f"   - 최고 모델: {best_model}")

        print(f"\n📁 생성된 파일:")
        print(f"   - {labeled_path}")
        print(f"   - {feature_path}")
        for f in arff_files:
            print(f"   - {f}")
        print(f"   - models/bitcoin_predictor.pkl")
        print(f"   - data/charts/*.png ({min(5, len(feature_df))}개)")

        print(f"\n⏱️  총 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")

        print("\n" + "=" * 70)
        print("  💡 다음 단계")
        print("=" * 70)
        print("\n1. WEKA 분석:")
        print("   - WEKA를 실행하고 data/processed/*.arff 파일을 열어보세요")
        print("   - 분류, 군집화, 연관규칙 알고리즘을 시도해보세요")

        print("\n2. 웹 대시보드 실행:")
        print("   - streamlit run app.py")
        print("   - 브라우저에서 http://localhost:8501 접속")

        print("\n3. 클라우드 배포:")
        print("   - GitHub에 푸시")
        print("   - Streamlit Cloud에서 배포")

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
    main()
