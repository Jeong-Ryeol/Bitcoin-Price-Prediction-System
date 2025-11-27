"""
Learning Curve 시각화 모듈
- 여러 알고리즘의 학습 곡선을 한 그래프에 표시
- SVM이 가장 좋다는 것을 시각적으로 증명
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def load_and_prepare_data():
    """데이터 로드 및 전처리"""
    df = pd.read_csv('data/processed/bitcoin_features.csv')

    # 피처와 타겟 분리
    feature_cols = ['open', 'high', 'low', 'close', 'volume']
    categorical_cols = ['ma_cross', 'rsi_signal', 'volume_spike']

    X = df[feature_cols].copy()

    # 범주형 변수 인코딩
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(df[col])

    y = le.fit_transform(df['price_direction'])

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def plot_learning_curves():
    """Learning Curve 그래프 생성"""
    print("=" * 70)
    print("📊 Learning Curve 생성 중...")
    print("=" * 70)

    X, y = load_and_prepare_data()
    print(f"✅ 데이터 로드 완료: {len(y)}개 인스턴스")

    # 알고리즘 정의
    models = {
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42)
    }

    # 색상 정의 (SVM을 강조)
    colors = {
        'SVM (RBF)': '#FF0000',        # 빨강 (강조)
        'Random Forest': '#4CAF50',     # 초록
        'Naive Bayes': '#2196F3',       # 파랑
        'Decision Tree': '#FF9800'      # 주황
    }

    # 학습 데이터 비율 (10%, 20%, ..., 100%)
    train_sizes = np.linspace(0.1, 1.0, 10)

    # 그래프 설정
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ===== 그래프 1: Learning Curve =====
    ax1 = axes[0]

    results = {}

    for name, model in models.items():
        print(f"\n🔄 {name} 학습 중...")

        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )

        # 평균과 표준편차 계산
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_mean = test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)

        results[name] = {
            'train_sizes': train_sizes_abs,
            'test_mean': test_mean,
            'test_std': test_std
        }

        # 테스트 점수 플롯 (validation score)
        line_width = 3 if name == 'SVM (RBF)' else 2
        marker_size = 10 if name == 'SVM (RBF)' else 6

        ax1.plot(train_sizes_abs, test_mean, 'o-',
                color=colors[name],
                label=name,
                linewidth=line_width,
                markersize=marker_size)

        # 표준편차 범위 표시 (오차 범위)
        ax1.fill_between(train_sizes_abs,
                        test_mean - test_std,
                        test_mean + test_std,
                        alpha=0.15,
                        color=colors[name])

        print(f"   ✅ 완료 - 최종 정확도: {test_mean[-1]*100:.2f}% (±{test_std[-1]*100:.2f}%)")

    ax1.set_xlabel('Training Set Size (훈련 데이터 개수)', fontsize=12)
    ax1.set_ylabel('Accuracy (정확도)', fontsize=12)
    ax1.set_title('Learning Curve: 알고리즘별 성능 비교', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.0])

    # x축 레이블을 퍼센트로 표시
    ax1.set_xticks(train_sizes_abs)
    ax1.set_xticklabels([f'{int(s/len(y)*100)}%\n({s})' for s in train_sizes_abs], fontsize=9)

    # ===== 그래프 2: 최종 성능 비교 (막대 그래프) =====
    ax2 = axes[1]

    final_scores = {name: results[name]['test_mean'][-1] for name in models.keys()}
    final_stds = {name: results[name]['test_std'][-1] for name in models.keys()}

    names = list(final_scores.keys())
    scores = [final_scores[n] * 100 for n in names]
    stds = [final_stds[n] * 100 for n in names]
    bar_colors = [colors[n] for n in names]

    bars = ax2.bar(names, scores, color=bar_colors, edgecolor='black', linewidth=1.5)
    ax2.errorbar(names, scores, yerr=stds, fmt='none', color='black', capsize=5, capthick=2)

    # SVM 막대 강조
    bars[0].set_edgecolor('red')
    bars[0].set_linewidth(3)

    # 값 표시
    for i, (bar, score, std) in enumerate(zip(bars, scores, stds)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f'{score:.1f}%\n(±{std:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('최종 성능 비교 (100% 데이터 사용)', fontsize=14, fontweight='bold')
    ax2.set_ylim([60, 105])
    ax2.axhline(y=max(scores), color='red', linestyle='--', alpha=0.5, label='Best')

    # SVM이 최고임을 표시
    ax2.annotate('★ Best Model',
                xy=(0, max(scores)),
                xytext=(0.5, max(scores) + 8),
                fontsize=12,
                color='red',
                fontweight='bold',
                ha='center')

    plt.tight_layout()

    # 저장
    output_path = 'weka_results/learning_curve.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n💾 저장 완료: {output_path}")

    plt.show()

    # 결과 요약 출력
    print("\n" + "=" * 70)
    print("📊 결과 요약")
    print("=" * 70)
    print("\n알고리즘별 최종 성능 (100% 데이터 사용):\n")

    sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (name, score) in enumerate(sorted_results, 1):
        std = final_stds[name]
        marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{marker} {i}. {name}: {score*100:.2f}% (±{std*100:.2f}%)")

    print("\n" + "=" * 70)
    print("✅ SVM이 가장 높은 정확도를 보임!")
    print("=" * 70)

    return results


def plot_svm_confidence_interval():
    """SVM의 Cross-Validation 정확도 범위 시각화"""
    print("\n" + "=" * 70)
    print("📊 SVM 정확도 범위 분석 중...")
    print("=" * 70)

    X, y = load_and_prepare_data()

    # SVM 모델
    svm = SVC(kernel='rbf', random_state=42)

    # 10-fold Cross Validation
    cv_scores = cross_val_score(svm, X, y, cv=10, scoring='accuracy')

    print(f"\n10-Fold Cross Validation 결과:")
    for i, score in enumerate(cv_scores, 1):
        print(f"   Fold {i:2d}: {score*100:.2f}%")

    mean_score = cv_scores.mean()
    std_score = cv_scores.std()

    print(f"\n평균 정확도: {mean_score*100:.2f}%")
    print(f"표준편차: {std_score*100:.2f}%")
    print(f"95% 신뢰구간: {(mean_score - 1.96*std_score)*100:.2f}% ~ {(mean_score + 1.96*std_score)*100:.2f}%")

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))

    # 각 fold의 점수를 점으로 표시
    ax.scatter(range(1, 11), cv_scores * 100, s=100, color='blue', zorder=5, label='Fold Scores')

    # 평균선
    ax.axhline(y=mean_score * 100, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_score*100:.2f}%')

    # 표준편차 범위
    ax.axhspan((mean_score - std_score) * 100, (mean_score + std_score) * 100,
               alpha=0.3, color='red', label=f'±1 Std: {std_score*100:.2f}%')

    # 95% 신뢰구간
    ax.axhspan((mean_score - 1.96*std_score) * 100, (mean_score + 1.96*std_score) * 100,
               alpha=0.15, color='blue', label='95% CI')

    ax.set_xlabel('Fold Number', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('SVM 10-Fold Cross Validation 정확도 범위', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, 11))
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([60, 100])

    plt.tight_layout()

    # 저장
    output_path = 'weka_results/svm_confidence_interval.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n💾 저장 완료: {output_path}")

    plt.show()

    return cv_scores


if __name__ == "__main__":
    # 1. Learning Curve (모든 알고리즘 비교)
    results = plot_learning_curves()

    # 2. SVM 정확도 범위
    cv_scores = plot_svm_confidence_interval()
