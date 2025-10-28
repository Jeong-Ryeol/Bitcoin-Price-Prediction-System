"""
비트코인 가격 예측 시스템 - Streamlit 대시보드
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
from datetime import datetime

# 프로젝트 경로 추가
sys.path.append('src')

from collector import BitcoinDataCollector
from chart_analyzer import ChartAnalyzer
from predictor import BitcoinPredictor


# 페이지 설정
st.set_page_config(
    page_title="Bitcoin Price Prediction",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 제목
st.title("₿ Bitcoin Price Prediction System")
st.caption("Machine Learning-based Cryptocurrency Price Forecasting")
st.markdown("---")


@st.cache_data(ttl=300)  # 5분 캐시
def load_historical_data():
    """과거 데이터 로드"""
    csv_path = 'data/processed/bitcoin_features.csv'

    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


@st.cache_resource
def load_model():
    """학습된 모델 로드"""
    model_path = 'models/bitcoin_predictor.pkl'

    if not os.path.exists(model_path):
        return None

    predictor = BitcoinPredictor()
    predictor.load_model(model_path)
    return predictor


def get_current_bitcoin_data():
    """실시간 비트코인 데이터 조회"""
    collector = BitcoinDataCollector()
    df = collector.get_candles_hour(count=50)  # 최근 50시간

    if df is None:
        return None

    # 차트 분석
    analyzer = ChartAnalyzer(df)
    analyzer.calculate_technical_indicators()
    analyzer.detect_patterns()

    return analyzer.df


def create_candlestick_chart(df):
    """캔들스틱 차트 생성"""
    fig = go.Figure()

    # 캔들스틱
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Bitcoin',
        increasing_line_color='red',
        decreasing_line_color='blue'
    ))

    # 이동평균선
    if 'ma5' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['ma5'],
            mode='lines',
            name='MA5',
            line=dict(color='orange', width=1)
        ))

    if 'ma20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['ma20'],
            mode='lines',
            name='MA20',
            line=dict(color='purple', width=1)
        ))

    fig.update_layout(
        title='Bitcoin Price Chart',
        yaxis_title='Price (KRW)',
        xaxis_title='Time',
        height=500,
        xaxis_rangeslider_visible=False
    )

    return fig


# 사이드바
with st.sidebar:
    st.header("⚙ Settings")

    page = st.radio(
        "Navigation",
        ["Dashboard", "Live Prediction", "Historical Analysis", "WEKA Analysis", "About"]
    )

    st.markdown("---")
    st.markdown("### Project Info")
    st.markdown("**Data Mining Project**")
    st.markdown("Bitcoin Price Prediction using Chart Patterns")


# 메인 페이지
if page == "Dashboard":
    st.header("Dashboard")

    # 데이터 로드
    df = load_historical_data()

    if df is None:
        st.warning("Warning: 데이터가 없습니다. 먼저 데이터를 수집하세요.")
        st.code("python3 src/collector.py", language="bash")
        st.stop()

    # 통계 정보
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Instances",
            f"{len(df):,}",
            delta=None
        )

    with col2:
        latest_price = df['close'].iloc[-1]
        st.metric(
            "Latest Price",
            f"₩{latest_price:,.0f}",
            delta=None
        )

    with col3:
        price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
        st.metric(
            "Price Change",
            f"{price_change:+.2f}%",
            delta=f"{price_change:+.2f}%"
        )

    with col4:
        class_dist = df['price_direction'].value_counts()
        dominant_class = class_dist.idxmax()
        st.metric(
            "Dominant Class",
            dominant_class,
            delta=f"{class_dist[dominant_class]} instances"
        )

    st.markdown("---")

    # 차트
    st.subheader("Price Chart")
    chart_df = df.tail(100)  # 최근 100개
    fig = create_candlestick_chart(chart_df)
    st.plotly_chart(fig, use_container_width=True)

    # 클래스 분포
    st.markdown("---")
    st.subheader("Class Distribution")

    col1, col2 = st.columns(2)

    with col1:
        class_counts = df['price_direction'].value_counts()
        fig_pie = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            title='Price Direction Distribution',
            color_discrete_map={'UP': 'red', 'DOWN': 'blue', 'STABLE': 'gray'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # 패턴 분포
        pattern_data = {
            'MA Cross': df['ma_cross'].value_counts().to_dict(),
            'RSI Signal': df['rsi_signal'].value_counts().to_dict(),
            'Volume Spike': df['volume_spike'].value_counts().to_dict()
        }

        st.markdown("**Chart Patterns**")
        for pattern, counts in pattern_data.items():
            st.write(f"**{pattern}:**")
            for k, v in counts.items():
                st.write(f"  - {k}: {v} ({v/len(df)*100:.1f}%)")


elif page == "Live Prediction":
    st.header("Live Prediction")

    # 모델 로드
    predictor = load_model()

    if predictor is None:
        st.warning("Warning: 학습된 모델이 없습니다. 먼저 모델을 학습하세요.")
        st.code("python3 src/predictor.py", language="bash")
        st.stop()

    st.success("Success: 모델 로드 완료")

    # 실시간 데이터 가져오기
    if st.button("Get Current Bitcoin Data"):
        with st.spinner("데이터 수집 중..."):
            current_df = get_current_bitcoin_data()

        if current_df is not None and len(current_df) > 0:
            st.success("Success: 데이터 수집 완료")

            # 최신 데이터
            latest = current_df.iloc[-1]

            st.subheader("Current Market Data")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Price", f"₩{latest['close']:,.0f}")
                st.metric("Volume", f"{latest['volume']:,.2f}")

            with col2:
                st.metric("High", f"₩{latest['high']:,.0f}")
                st.metric("Low", f"₩{latest['low']:,.0f}")

            with col3:
                st.metric("MA Cross", latest['ma_cross'])
                st.metric("RSI Signal", latest['rsi_signal'])
                st.metric("Volume Spike", latest['volume_spike'])

            # 예측
            st.markdown("---")
            st.subheader("Prediction")

            # 인코딩 (간단히 매핑)
            ma_cross_map = {'golden': 0, 'dead': 1, 'neutral': 2}
            rsi_signal_map = {'overbought': 0, 'oversold': 1, 'neutral': 2}
            volume_spike_map = {'high': 0, 'normal': 1, 'low': 2}

            features = {
                'open': latest['open'],
                'high': latest['high'],
                'low': latest['low'],
                'close': latest['close'],
                'volume': latest['volume'],
                'ma_cross': ma_cross_map.get(latest['ma_cross'], 2),
                'rsi_signal': rsi_signal_map.get(latest['rsi_signal'], 2),
                'volume_spike': volume_spike_map.get(latest['volume_spike'], 1)
            }

            result = predictor.predict(features)

            # 결과 표시
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 예측 결과")
                direction_color = {
                    'UP': '🔴',
                    'DOWN': '🔵',
                    'STABLE': '⚪'
                }
                st.markdown(f"## {direction_color.get(result['prediction'], '')} {result['prediction']}")
                st.markdown(f"**신뢰도:** {result['confidence']:.1%}")

            with col2:
                st.markdown("### 클래스별 확률")
                prob_df = pd.DataFrame({
                    'Class': list(result['probabilities'].keys()),
                    'Probability': list(result['probabilities'].values())
                })
                fig_bar = px.bar(
                    prob_df,
                    x='Class',
                    y='Probability',
                    color='Class',
                    color_discrete_map={'UP': 'red', 'DOWN': 'blue', 'STABLE': 'gray'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # 차트
            st.markdown("---")
            st.subheader("Recent Price Chart")
            fig = create_candlestick_chart(current_df.tail(50))
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("Error: 데이터 수집 실패")


elif page == "Historical Analysis":
    st.header("Historical Analysis")

    df = load_historical_data()

    if df is None:
        st.warning("Warning: 데이터가 없습니다.")
        st.stop()

    # 날짜 범위 선택
    st.subheader("Select Date Range")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=df['timestamp'].min().date()
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=df['timestamp'].max().date()
        )

    # 필터링
    mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
    filtered_df = df[mask]

    st.write(f"**Filtered Data:** {len(filtered_df)} instances")

    # 차트
    fig = create_candlestick_chart(filtered_df)
    st.plotly_chart(fig, use_container_width=True)

    # 통계
    st.markdown("---")
    st.subheader("Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Average Price", f"₩{filtered_df['close'].mean():,.0f}")
        st.metric("Max Price", f"₩{filtered_df['close'].max():,.0f}")

    with col2:
        st.metric("Min Price", f"₩{filtered_df['close'].min():,.0f}")
        st.metric("Std Dev", f"₩{filtered_df['close'].std():,.0f}")

    with col3:
        st.metric("Avg Volume", f"{filtered_df['volume'].mean():,.2f}")
        st.metric("Total Volume", f"{filtered_df['volume'].sum():,.2f}")

    # 데이터 테이블
    st.markdown("---")
    st.subheader("📋 Data Table")

    display_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'ma_cross', 'rsi_signal', 'volume_spike', 'price_direction']

    st.dataframe(
        filtered_df[display_cols].tail(50),
        use_container_width=True
    )


elif page == "WEKA Analysis":
    st.header("🎓 WEKA-Style Analysis")

    st.markdown("""
    **WEKA 스타일 분석을 웹에서 바로 실행하세요!**

    WEKA 소프트웨어 없이도 동일한 알고리즘으로 분석할 수 있습니다.
    """)

    # 데이터 로드
    df = load_historical_data()
    predictor = load_model()

    if df is None or predictor is None:
        st.warning("Warning: 데이터 또는 모델을 먼저 생성하세요.")
        st.code("python3 run.py", language="bash")
        st.stop()

    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        "Classification",
        "Decision Tree",
        "Clustering",
        "Association Rules"
    ])

    # ===== 분류 분석 =====
    with tab1:
        st.subheader("Classification Analysis")

        st.markdown("### Algorithm Selection")
        algorithm = st.selectbox(
            "Choose Classifier",
            ["Random Forest", "Decision Tree (J48)", "Naive Bayes", "SVM"]
        )

        if st.button("Run Classification", key="run_classification"):
            with st.spinner(f"Running {algorithm}..."):
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.naive_bayes import GaussianNB
                from sklearn.svm import SVC
                from sklearn.model_selection import train_test_split, cross_val_score
                from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

                # 데이터 준비
                X, y = predictor.prepare_data(df)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # 모델 선택
                if algorithm == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif algorithm == "Decision Tree (J48)":
                    model = DecisionTreeClassifier(max_depth=10, random_state=42)
                elif algorithm == "Naive Bayes":
                    model = GaussianNB()
                else:
                    model = SVC(kernel='rbf', random_state=42)

                # 학습
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # 평가
                accuracy = accuracy_score(y_test, y_pred)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                cm = confusion_matrix(y_test, y_pred)

                # === WEKA 스타일 출력 ===
                st.markdown("---")
                st.markdown("### === Run Information ===")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Scheme", algorithm)
                    st.metric("Relation", "bitcoin_price_prediction")
                    st.metric("Instances", len(df))

                with col2:
                    st.metric("Attributes", X.shape[1])
                    st.metric("Test mode", "10-fold cross-validation")

                st.markdown("---")
                st.markdown("### === Classifier Model ===")
                st.info(f"**{algorithm}**\n\nNumber of instances: {len(X_train)}\nNumber of attributes: {X.shape[1]}")

                st.markdown("---")
                st.markdown("### === Stratified Cross-Validation ===")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Correctly Classified",
                        f"{accuracy * 100:.2f}%",
                        f"{int(accuracy * len(X_test))}/{len(X_test)} instances"
                    )

                with col2:
                    st.metric(
                        "Incorrectly Classified",
                        f"{(1-accuracy) * 100:.2f}%",
                        f"{int((1-accuracy) * len(X_test))}/{len(X_test)} instances"
                    )

                with col3:
                    st.metric(
                        "Cross-Validation Accuracy",
                        f"{cv_scores.mean() * 100:.2f}%",
                        f"± {cv_scores.std() * 200:.2f}%"
                    )

                st.markdown("---")
                st.markdown("### === Confusion Matrix ===")

                # Confusion Matrix 시각화
                import plotly.figure_factory as ff

                classes = list(set(y_test))
                fig = ff.create_annotated_heatmap(
                    cm,
                    x=classes,
                    y=classes,
                    colorscale='Blues',
                    showscale=True
                )
                fig.update_layout(
                    title='Confusion Matrix',
                    xaxis_title='Predicted Class',
                    yaxis_title='Actual Class',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.markdown("### === Detailed Accuracy By Class ===")

                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

                st.success("Success: Classification complete!")

    # ===== 의사결정 트리 =====
    with tab2:
        st.subheader("Decision Tree Visualization")

        if st.button("Generate Decision Tree", key="run_tree"):
            with st.spinner("Building decision tree..."):
                from sklearn.tree import DecisionTreeClassifier, plot_tree
                from sklearn.model_selection import train_test_split
                import matplotlib.pyplot as plt

                X, y = predictor.prepare_data(df)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # J48 스타일 Decision Tree
                tree = DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42
                )
                tree.fit(X_train, y_train)

                # 트리 시각화
                st.markdown("### === Decision Tree (J48 Style) ===")

                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(
                    tree,
                    feature_names=['open', 'high', 'low', 'close', 'volume',
                                   'ma_cross', 'rsi_signal', 'volume_spike'],
                    class_names=tree.classes_,
                    filled=True,
                    rounded=True,
                    ax=ax,
                    fontsize=10
                )
                st.pyplot(fig)

                # 트리 정보
                st.markdown("### === Tree Statistics ===")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tree Size", tree.tree_.node_count)
                with col2:
                    st.metric("Leaves", tree.tree_.n_leaves)
                with col3:
                    accuracy = tree.score(X_test, y_test)
                    st.metric("Accuracy", f"{accuracy * 100:.2f}%")

    # ===== 군집화 =====
    with tab3:
        st.subheader("Clustering Analysis")

        num_clusters = st.slider("Number of Clusters", 2, 5, 3)

        if st.button("Run K-Means Clustering", key="run_clustering"):
            with st.spinner(f"Running K-Means with {num_clusters} clusters..."):
                from sklearn.cluster import KMeans
                from sklearn.decomposition import PCA

                X, y = predictor.prepare_data(df)

                # K-means
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                clusters = kmeans.fit_predict(X)

                st.markdown("### === Clustered Instances ===")

                # 클러스터 분포
                cluster_counts = pd.Series(clusters).value_counts().sort_index()

                cols = st.columns(num_clusters)
                for i, (cluster, count) in enumerate(cluster_counts.items()):
                    with cols[i]:
                        st.metric(
                            f"Cluster {cluster}",
                            f"{count} instances",
                            f"{count/len(clusters)*100:.1f}%"
                        )

                # PCA로 2D 시각화
                st.markdown("### === Cluster Visualization ===")

                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)

                cluster_df = pd.DataFrame({
                    'PC1': X_pca[:, 0],
                    'PC2': X_pca[:, 1],
                    'Cluster': [f'Cluster {c}' for c in clusters]
                })

                fig = px.scatter(
                    cluster_df,
                    x='PC1',
                    y='PC2',
                    color='Cluster',
                    title='K-Means Clustering (PCA Projection)',
                    width=800,
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### === Cluster Centroids ===")
                centroids_df = pd.DataFrame(
                    kmeans.cluster_centers_,
                    columns=['open', 'high', 'low', 'close', 'volume',
                             'ma_cross', 'rsi_signal', 'volume_spike']
                )
                st.dataframe(centroids_df.style.format("{:.2f}"), use_container_width=True)

    # ===== 연관규칙 =====
    with tab4:
        st.subheader("Association Rules Mining")

        st.info("연관규칙 분석은 범주형 데이터가 필요합니다. ARFF 파일을 사용하세요.")

        st.markdown("### ARFF 파일 다운로드")

        arff_path = 'data/processed/bitcoin_association.arff'
        if os.path.exists(arff_path):
            with open(arff_path, 'r') as f:
                arff_content = f.read()

            st.download_button(
                label="📥 Download Association ARFF",
                data=arff_content,
                file_name="bitcoin_association.arff",
                mime="text/plain"
            )

            st.markdown("### 주요 패턴 (예시)")

            st.markdown("""
            **Rule 1:**
            - IF ma_cross=golden AND rsi_signal=neutral
            - THEN price_direction=UP
            - Confidence: 82%, Support: 15%

            **Rule 2:**
            - IF ma_cross=dead AND volume_spike=high
            - THEN price_direction=DOWN
            - Confidence: 78%, Support: 12%

            **Rule 3:**
            - IF rsi_signal=overbought
            - THEN price_direction=DOWN
            - Confidence: 75%, Support: 10%
            """)
        else:
            st.warning("ARFF 파일을 먼저 생성하세요: `python3 run.py`")


elif page == "About":
    st.header("About This Project")

    st.markdown("""
    ## Project Overview

    **비트코인 가격 예측 시스템**은 데이터마이닝 기법을 활용하여
    암호화폐 시장의 가격 방향을 예측하는 머신러닝 프로젝트입니다.

    ### Features

    - **실시간 데이터 수집**: Upbit API를 통한 과거 및 실시간 데이터 수집
    - **차트 패턴 인식**: 이동평균선, RSI, 거래량 기반 패턴 분석
    - **머신러닝 예측**: Random Forest 등 다양한 알고리즘 활용
    - **웹 대시보드**: 인터랙티브 시각화 및 실시간 예측

    ### Technology Stack

    - **Data Collection**: Upbit Public API
    - **Analysis**: pandas, pandas-ta, mplfinance
    - **Machine Learning**: scikit-learn, WEKA
    - **Visualization**: Streamlit, Plotly
    - **Deployment**: Streamlit Cloud

    ### Data Attributes

    **Price Data (5개)**
    - Open, High, Low, Close, Volume

    **Chart Patterns (3개)**
    - MA Cross: Golden/Death/Neutral
    - RSI Signal: Overbought/Oversold/Neutral
    - Volume Spike: High/Normal/Low

    **Target Class**
    - Price Direction: UP/DOWN/STABLE (1시간 후 예측)

    ### 🎓 Academic Purpose

    이 프로젝트는 데이터마이닝 과목의 일환으로 진행되었으며,
    WEKA를 활용한 데이터 분석 및 머신러닝 모델 구축을 목표로 합니다.

    ### How to Use

    1. **데이터 수집**: `python3 src/collector.py`
    2. **차트 분석**: `python3 src/chart_analyzer.py`
    3. **ARFF 생성**: `python3 src/arff_generator.py`
    4. **모델 학습**: `python3 src/predictor.py`
    5. **웹 실행**: `streamlit run app.py`

    ### Warning: Disclaimer

    이 시스템은 교육 목적으로 제작되었으며,
    실제 투자 결정에 사용해서는 안 됩니다.
    """)

    st.markdown("---")
    st.markdown("Made with Love for Data Mining Project")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Bitcoin Price Prediction System | Data Mining Project 2025</div>",
    unsafe_allow_html=True
)
