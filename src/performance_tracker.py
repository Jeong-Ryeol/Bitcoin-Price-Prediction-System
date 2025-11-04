"""
모델 성능 추적 모듈
시간대별 정확도 및 성능 지표 저장/로드
"""

import json
import os
from datetime import datetime
import pandas as pd
import numpy as np


class PerformanceTracker:
    """모델 성능 추적 및 이력 관리"""

    def __init__(self, history_file='data/model_performance.json'):
        self.history_file = history_file
        self.history = self.load_history()

    def load_history(self):
        """이력 파일 로드"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"이력 로드 실패: {e}")
                return self._create_empty_history()
        else:
            return self._create_empty_history()

    def _create_empty_history(self):
        """빈 이력 구조 생성"""
        return {
            'evaluations': [],
            'summary': {
                'total_evaluations': 0,
                'best_model': None,
                'best_accuracy': 0.0,
                'last_updated': None
            }
        }

    def save_history(self):
        """이력 파일 저장"""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)

        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

        print(f"✅ 성능 이력 저장 완료: {self.history_file}")

    def add_evaluation(self, model_name, accuracy, cv_scores, confusion_matrix,
                       data_size, timestamp=None):
        """
        새 평가 결과 추가

        Args:
            model_name: 모델 이름
            accuracy: 테스트 정확도
            cv_scores: 교차 검증 점수 리스트
            confusion_matrix: 혼동 행렬 (numpy array or list)
            data_size: 학습 데이터 크기
            timestamp: 평가 시간 (None이면 현재 시간)
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # numpy array를 list로 변환
        if isinstance(confusion_matrix, np.ndarray):
            confusion_matrix = confusion_matrix.tolist()

        if isinstance(cv_scores, np.ndarray):
            cv_scores = cv_scores.tolist()

        evaluation = {
            'timestamp': timestamp,
            'model_name': model_name,
            'accuracy': float(accuracy),
            'cv_mean': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'cv_scores': [float(s) for s in cv_scores],
            'confusion_matrix': confusion_matrix,
            'data_size': int(data_size)
        }

        self.history['evaluations'].append(evaluation)

        # 요약 정보 업데이트
        self.history['summary']['total_evaluations'] = len(self.history['evaluations'])
        self.history['summary']['last_updated'] = timestamp

        # 최고 모델 업데이트
        if accuracy > self.history['summary']['best_accuracy']:
            self.history['summary']['best_model'] = model_name
            self.history['summary']['best_accuracy'] = float(accuracy)

        self.save_history()

        print(f"📊 {model_name} 평가 결과 저장 완료 (정확도: {accuracy:.2%})")

    def get_all_evaluations(self):
        """모든 평가 결과 반환 (DataFrame)"""
        if not self.history['evaluations']:
            return pd.DataFrame()

        return pd.DataFrame(self.history['evaluations'])

    def get_model_history(self, model_name):
        """특정 모델의 평가 이력"""
        df = self.get_all_evaluations()

        if df.empty:
            return pd.DataFrame()

        return df[df['model_name'] == model_name]

    def get_latest_evaluation(self, model_name=None):
        """최신 평가 결과"""
        df = self.get_all_evaluations()

        if df.empty:
            return None

        if model_name:
            df = df[df['model_name'] == model_name]

        if df.empty:
            return None

        return df.iloc[-1].to_dict()

    def get_accuracy_over_time(self, model_name=None):
        """
        시간대별 정확도 변화

        Returns:
            DataFrame: timestamp, model_name, accuracy
        """
        df = self.get_all_evaluations()

        if df.empty:
            return pd.DataFrame()

        if model_name:
            df = df[df['model_name'] == model_name]

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df[['timestamp', 'model_name', 'accuracy', 'cv_mean']].sort_values('timestamp')

    def compare_models(self):
        """모든 모델의 최신 성능 비교"""
        df = self.get_all_evaluations()

        if df.empty:
            return pd.DataFrame()

        # 각 모델의 최신 평가만 추출
        latest_by_model = df.groupby('model_name').last().reset_index()

        return latest_by_model[['model_name', 'accuracy', 'cv_mean', 'cv_std', 'data_size']].sort_values('accuracy', ascending=False)

    def get_summary(self):
        """요약 정보"""
        return self.history['summary']

    def clear_history(self):
        """이력 초기화"""
        self.history = self._create_empty_history()
        self.save_history()
        print("🗑️ 성능 이력 초기화 완료")


def test_tracker():
    """테스트 함수"""
    tracker = PerformanceTracker()

    # 더미 데이터 추가
    models = ['OneR', 'Naive Bayes', 'Decision Tree (J48)', 'Random Forest', 'SVM']
    accuracies = [0.55, 0.70, 0.58, 0.72, 0.75]

    for model, acc in zip(models, accuracies):
        cv_scores = [acc + np.random.uniform(-0.05, 0.05) for _ in range(5)]
        cm = [[10, 2, 1], [1, 30, 2], [0, 3, 5]]

        tracker.add_evaluation(
            model_name=model,
            accuracy=acc,
            cv_scores=cv_scores,
            confusion_matrix=cm,
            data_size=199
        )

    # 결과 출력
    print("\n=== 모델 비교 ===")
    print(tracker.compare_models())

    print("\n=== 요약 정보 ===")
    print(json.dumps(tracker.get_summary(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    test_tracker()
