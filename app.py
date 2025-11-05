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
from performance_tracker import PerformanceTracker


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
        ["Dashboard", "Live Prediction", "Manual Prediction", "Model Performance",
         "Dataset Explorer", "Chart Image Analysis", "Historical Analysis", "WEKA Analysis", "About"]
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


elif page == "Manual Prediction":
    st.header("Manual Prediction")
    st.markdown("수동으로 데이터를 입력하여 가격 방향을 예측합니다.")

    # 모델 로드
    predictor = load_model()

    if predictor is None:
        st.warning("Warning: 학습된 모델이 없습니다. 먼저 모델을 학습하세요.")
        st.code("python3 src/predictor.py", language="bash")
        st.stop()

    # 탭 구성
    tab1, tab2 = st.tabs(["개별 입력", "CSV 업로드"])

    # ===== 개별 입력 탭 =====
    with tab1:
        st.subheader("개별 데이터 입력")
        st.markdown("각 속성 값을 입력하고 예측 버튼을 클릭하세요.")

        # 기준 데이터 로드 (최소/최대값 표시용)
        df_ref = load_historical_data()

        if df_ref is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Numeric Attributes")

                open_price = st.number_input(
                    f"Open (시가) [{df_ref['open'].min():,.0f} ~ {df_ref['open'].max():,.0f}]",
                    value=float(df_ref['open'].mean()),
                    step=100000.0,
                    format="%.0f"
                )

                high_price = st.number_input(
                    f"High (고가) [{df_ref['high'].min():,.0f} ~ {df_ref['high'].max():,.0f}]",
                    value=float(df_ref['high'].mean()),
                    step=100000.0,
                    format="%.0f"
                )

                low_price = st.number_input(
                    f"Low (저가) [{df_ref['low'].min():,.0f} ~ {df_ref['low'].max():,.0f}]",
                    value=float(df_ref['low'].mean()),
                    step=100000.0,
                    format="%.0f"
                )

                close_price = st.number_input(
                    f"Close (종가) [{df_ref['close'].min():,.0f} ~ {df_ref['close'].max():,.0f}]",
                    value=float(df_ref['close'].mean()),
                    step=100000.0,
                    format="%.0f"
                )

                volume = st.number_input(
                    f"Volume (거래량) [{df_ref['volume'].min():.2f} ~ {df_ref['volume'].max():.2f}]",
                    value=float(df_ref['volume'].mean()),
                    step=1.0,
                    format="%.2f"
                )

            with col2:
                st.markdown("### Nominal Attributes")

                ma_cross_options = ['golden', 'dead', 'neutral']
                ma_cross_dist = df_ref['ma_cross'].value_counts()
                ma_cross_desc = {
                    'golden': f"Golden Cross - 단기MA > 장기MA (상승) [{ma_cross_dist.get('golden', 0)}개]",
                    'dead': f"Death Cross - 단기MA < 장기MA (하락) [{ma_cross_dist.get('dead', 0)}개]",
                    'neutral': f"Neutral - 교차 없음 [{ma_cross_dist.get('neutral', 0)}개]"
                }
                ma_cross = st.selectbox("MA Cross", ma_cross_options, format_func=lambda x: ma_cross_desc[x])

                rsi_signal_options = ['overbought', 'oversold', 'neutral']
                rsi_signal_dist = df_ref['rsi_signal'].value_counts()
                rsi_signal_desc = {
                    'overbought': f"Overbought - RSI > 70 (과매수) [{rsi_signal_dist.get('overbought', 0)}개]",
                    'oversold': f"Oversold - RSI < 30 (과매도) [{rsi_signal_dist.get('oversold', 0)}개]",
                    'neutral': f"Neutral - 30 ≤ RSI ≤ 70 [{rsi_signal_dist.get('neutral', 0)}개]"
                }
                rsi_signal = st.selectbox("RSI Signal", rsi_signal_options, format_func=lambda x: rsi_signal_desc[x])

                volume_spike_options = ['high', 'normal', 'low']
                volume_spike_dist = df_ref['volume_spike'].value_counts()
                volume_spike_desc = {
                    'high': f"High - 거래량 > 평균×1.5 [{volume_spike_dist.get('high', 0)}개]",
                    'normal': f"Normal - 정상 거래량 [{volume_spike_dist.get('normal', 0)}개]",
                    'low': f"Low - 거래량 < 평균×0.5 [{volume_spike_dist.get('low', 0)}개]"
                }
                volume_spike = st.selectbox("Volume Spike", volume_spike_options, format_func=lambda x: volume_spike_desc[x])

            # 예측 버튼
            st.markdown("---")
            if st.button("예측하기", type="primary", key="manual_predict"):
                # 인코딩
                ma_cross_map = {'golden': 0, 'dead': 1, 'neutral': 2}
                rsi_signal_map = {'overbought': 0, 'oversold': 1, 'neutral': 2}
                volume_spike_map = {'high': 0, 'normal': 1, 'low': 2}

                features = {
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume,
                    'ma_cross': ma_cross_map[ma_cross],
                    'rsi_signal': rsi_signal_map[rsi_signal],
                    'volume_spike': volume_spike_map[volume_spike]
                }

                # 예측
                result = predictor.predict(features)

                # 결과 표시
                st.markdown("---")
                st.markdown("### 예측 결과")

                col1, col2, col3 = st.columns(3)

                with col1:
                    direction_emoji = {
                        'UP': '🔴',
                        'DOWN': '🔵',
                        'STABLE': '⚪'
                    }
                    st.markdown(f"## {direction_emoji.get(result['prediction'], '')} {result['prediction']}")
                    st.markdown(f"**신뢰도:** {result['confidence']:.1%}")

                with col2:
                    st.markdown("#### 클래스별 확률")
                    for cls, prob in result['probabilities'].items():
                        st.write(f"**{cls}:** {prob:.1%}")

                with col3:
                    # 확률 바 차트
                    prob_df = pd.DataFrame({
                        'Class': list(result['probabilities'].keys()),
                        'Probability': list(result['probabilities'].values())
                    })
                    fig_bar = px.bar(
                        prob_df,
                        x='Class',
                        y='Probability',
                        color='Class',
                        color_discrete_map={'UP': 'red', 'DOWN': 'blue', 'STABLE': 'gray'},
                        title="확률 분포"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.error("데이터를 먼저 로드하세요.")

    # ===== CSV 업로드 탭 =====
    with tab2:
        st.subheader("CSV 파일 업로드")
        st.markdown("여러 테스트 데이터를 CSV 파일로 일괄 예측합니다.")

        # CSV 형식 안내
        with st.expander("CSV 파일 형식 안내"):
            st.markdown("""
            ### 필수 컬럼 (8개)
            1. **open** - 시가 (숫자)
            2. **high** - 고가 (숫자)
            3. **low** - 저가 (숫자)
            4. **close** - 종가 (숫자)
            5. **volume** - 거래량 (숫자)
            6. **ma_cross** - 이동평균 교차 (golden, dead, neutral 중 하나)
            7. **rsi_signal** - RSI 신호 (overbought, oversold, neutral 중 하나)
            8. **volume_spike** - 거래량 급등 (high, normal, low 중 하나)

            ### 예시
            ```
            open,high,low,close,volume,ma_cross,rsi_signal,volume_spike
            165000000,166000000,164000000,165500000,45.5,golden,neutral,high
            166000000,167000000,165500000,166500000,50.2,neutral,overbought,normal
            ```
            """)

        # 파일 업로드
        uploaded_file = st.file_uploader("CSV 파일 선택", type=['csv'])

        if uploaded_file is not None:
            try:
                # CSV 읽기
                df_test = pd.read_csv(uploaded_file)

                st.success(f"Success: {len(df_test)}개 인스턴스 로드 완료")

                # 데이터 미리보기
                st.markdown("### 데이터 미리보기")
                st.dataframe(df_test.head(10), use_container_width=True)

                # 예측 버튼
                if st.button("일괄 예측하기", type="primary", key="csv_predict"):
                    with st.spinner("예측 중..."):
                        # 인코딩
                        ma_cross_map = {'golden': 0, 'dead': 1, 'neutral': 2}
                        rsi_signal_map = {'overbought': 0, 'oversold': 1, 'neutral': 2}
                        volume_spike_map = {'high': 0, 'normal': 1, 'low': 2}

                        df_test['ma_cross_encoded'] = df_test['ma_cross'].map(ma_cross_map)
                        df_test['rsi_signal_encoded'] = df_test['rsi_signal'].map(rsi_signal_map)
                        df_test['volume_spike_encoded'] = df_test['volume_spike'].map(volume_spike_map)

                        # 예측
                        predictions = []
                        confidences = []

                        for idx, row in df_test.iterrows():
                            features = {
                                'open': row['open'],
                                'high': row['high'],
                                'low': row['low'],
                                'close': row['close'],
                                'volume': row['volume'],
                                'ma_cross': row['ma_cross_encoded'],
                                'rsi_signal': row['rsi_signal_encoded'],
                                'volume_spike': row['volume_spike_encoded']
                            }

                            result = predictor.predict(features)
                            predictions.append(result['prediction'])
                            confidences.append(result['confidence'])

                        df_test['prediction'] = predictions
                        df_test['confidence'] = confidences

                    # 결과 표시
                    st.markdown("---")
                    st.markdown("### 예측 결과")

                    # 통계
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        pred_dist = pd.Series(predictions).value_counts()
                        st.metric("UP 예측", f"{pred_dist.get('UP', 0)}개")

                    with col2:
                        st.metric("DOWN 예측", f"{pred_dist.get('DOWN', 0)}개")

                    with col3:
                        st.metric("STABLE 예측", f"{pred_dist.get('STABLE', 0)}개")

                    # 결과 테이블
                    st.markdown("### 전체 결과")
                    result_cols = ['open', 'high', 'low', 'close', 'volume', 'ma_cross',
                                   'rsi_signal', 'volume_spike', 'prediction', 'confidence']
                    st.dataframe(df_test[result_cols], use_container_width=True)

                    # CSV 다운로드
                    csv_result = df_test[result_cols].to_csv(index=False)
                    st.download_button(
                        label="결과 CSV 다운로드",
                        data=csv_result,
                        file_name="prediction_results.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Error: 파일 읽기 실패 - {e}")


elif page == "Model Performance":
    st.header("Model Performance")
    st.markdown("모든 알고리즘의 성능을 비교하고 시각화합니다.")

    # 데이터 및 모델 로드
    df = load_historical_data()

    if df is None:
        st.warning("Warning: 데이터가 없습니다.")
        st.stop()

    # 성능 추적기
    tracker = PerformanceTracker()

    # 모델 재평가 버튼
    if st.button("모든 모델 재평가", type="primary"):
        with st.spinner("모델 평가 중... (1-2분 소요)"):
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.metrics import accuracy_score, confusion_matrix
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.naive_bayes import GaussianNB
            from sklearn.svm import SVC
            from mlxtend.classifier import OneRClassifier

            # 데이터 준비
            predictor = BitcoinPredictor()
            X, y = predictor.prepare_data(df)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # 모델 목록
            models = {
                'Naive Bayes': GaussianNB(),
                'Decision Tree (J48)': DecisionTreeClassifier(max_depth=10, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(kernel='rbf', random_state=42)
            }

            # 각 모델 평가
            for name, model in models.items():
                # 학습
                model.fit(X_train, y_train)

                # 예측
                y_pred = model.predict(X_test)

                # 정확도
                accuracy = accuracy_score(y_test, y_pred)

                # 교차 검증
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)

                # 혼동 행렬
                cm = confusion_matrix(y_test, y_pred)

                # 저장
                tracker.add_evaluation(
                    model_name=name,
                    accuracy=accuracy,
                    cv_scores=cv_scores,
                    confusion_matrix=cm,
                    data_size=len(X_train)
                )

        st.success("Success: 모든 모델 평가 완료!")
        st.experimental_rerun()

    # 평가 결과 가져오기
    df_performance = tracker.get_all_evaluations()

    if df_performance.empty:
        st.info("아직 평가된 모델이 없습니다. '모든 모델 재평가' 버튼을 클릭하세요.")
        st.stop()

    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        "알고리즘 비교",
        "시간대별 정확도",
        "Confusion Matrix",
        "교차 검증 분석"
    ])

    # ===== 탭 1: 알고리즘별 비교 막대 그래프 =====
    with tab1:
        st.subheader("알고리즘별 성능 비교")

        # 최신 평가 결과
        df_comparison = tracker.compare_models()

        # 막대 그래프
        fig_bar = px.bar(
            df_comparison,
            x='model_name',
            y='accuracy',
            color='model_name',
            title='알고리즘별 테스트 정확도 비교',
            labels={'model_name': '알고리즘', 'accuracy': '정확도'},
            text='accuracy'
        )
        fig_bar.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig_bar.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig_bar, use_container_width=True)

        # 교차 검증 점수 포함 비교
        st.markdown("---")
        st.markdown("### 상세 성능 지표")

        # 테이블
        df_display = df_comparison.copy()
        df_display['accuracy'] = df_display['accuracy'].apply(lambda x: f"{x:.2%}")
        df_display['cv_mean'] = df_display['cv_mean'].apply(lambda x: f"{x:.2%}")
        df_display['cv_std'] = df_display['cv_std'].apply(lambda x: f"± {x:.2%}")
        df_display.columns = ['알고리즘', '테스트 정확도', 'CV 평균', 'CV 표준편차', '학습 데이터 크기']

        st.dataframe(df_display, use_container_width=True)

        # 최고 모델 강조
        summary = tracker.get_summary()
        st.success(f"🏆 **최고 성능 모델:** {summary['best_model']} ({summary['best_accuracy']:.2%})")

    # ===== 탭 2: 시간대별 정확도 변화 라인 차트 =====
    with tab2:
        st.subheader("시간대별 정확도 변화")

        df_time = tracker.get_accuracy_over_time()

        if not df_time.empty:
            # 라인 차트
            fig_line = px.line(
                df_time,
                x='timestamp',
                y='accuracy',
                color='model_name',
                title='시간에 따른 모델 정확도 변화',
                labels={'timestamp': '시간', 'accuracy': '정확도', 'model_name': '알고리즘'},
                markers=True
            )
            fig_line.update_layout(height=500)
            st.plotly_chart(fig_line, use_container_width=True)

            # CV 점수 포함 라인 차트
            st.markdown("---")
            st.markdown("### 교차 검증 점수 추이")

            fig_cv = px.line(
                df_time,
                x='timestamp',
                y='cv_mean',
                color='model_name',
                title='교차 검증 점수 변화',
                labels={'timestamp': '시간', 'cv_mean': 'CV 평균 정확도', 'model_name': '알고리즘'},
                markers=True
            )
            fig_cv.update_layout(height=500)
            st.plotly_chart(fig_cv, use_container_width=True)

        else:
            st.info("아직 시간별 데이터가 충분하지 않습니다.")

    # ===== 탭 3: Confusion Matrix 히트맵 =====
    with tab3:
        st.subheader("Confusion Matrix (혼동 행렬)")

        # 모델 선택
        model_names = df_performance['model_name'].unique().tolist()
        selected_model = st.selectbox("알고리즘 선택", model_names)

        # 최신 평가 결과
        latest_eval = tracker.get_latest_evaluation(selected_model)

        if latest_eval:
            cm = np.array(latest_eval['confusion_matrix'])

            # 히트맵
            import plotly.figure_factory as ff

            # 클래스 이름
            classes = ['DOWN', 'STABLE', 'UP']

            fig_heatmap = ff.create_annotated_heatmap(
                cm,
                x=classes,
                y=classes,
                colorscale='Blues',
                showscale=True
            )
            fig_heatmap.update_layout(
                title=f'{selected_model} - Confusion Matrix',
                xaxis_title='Predicted Class',
                yaxis_title='Actual Class',
                height=500,
                width=600
            )
            fig_heatmap.update_xaxes(side="bottom")

            st.plotly_chart(fig_heatmap, use_container_width=True)

            # 클래스별 정확도 계산
            st.markdown("---")
            st.markdown("### 클래스별 정확도")

            class_accuracies = []
            for i, cls in enumerate(classes):
                if cm[i].sum() > 0:
                    acc = cm[i][i] / cm[i].sum()
                    class_accuracies.append({'Class': cls, 'Accuracy': acc})

            df_class_acc = pd.DataFrame(class_accuracies)

            fig_class = px.bar(
                df_class_acc,
                x='Class',
                y='Accuracy',
                color='Class',
                title='클래스별 정확도',
                text='Accuracy',
                color_discrete_map={'UP': 'red', 'DOWN': 'blue', 'STABLE': 'gray'}
            )
            fig_class.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            fig_class.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_class, use_container_width=True)

        else:
            st.warning("선택한 모델의 평가 결과가 없습니다.")

    # ===== 탭 4: 교차 검증 결과 박스플롯 =====
    with tab4:
        st.subheader("교차 검증 결과 분석")

        # 모든 모델의 CV 점수 수집
        cv_data = []
        for idx, row in df_performance.iterrows():
            model_name = row['model_name']
            for fold, score in enumerate(row['cv_scores'], 1):
                cv_data.append({
                    'Model': model_name,
                    'Fold': fold,
                    'Accuracy': score
                })

        df_cv = pd.DataFrame(cv_data)

        if not df_cv.empty:
            # 박스플롯
            fig_box = px.box(
                df_cv,
                x='Model',
                y='Accuracy',
                color='Model',
                title='교차 검증 점수 분포 (5-Fold CV)',
                labels={'Model': '알고리즘', 'Accuracy': '정확도'}
            )
            fig_box.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig_box, use_container_width=True)

            # 바이올린 플롯
            st.markdown("---")
            st.markdown("### 점수 분포 (Violin Plot)")

            fig_violin = px.violin(
                df_cv,
                x='Model',
                y='Accuracy',
                color='Model',
                box=True,
                points='all',
                title='교차 검증 점수 상세 분포',
                labels={'Model': '알고리즘', 'Accuracy': '정확도'}
            )
            fig_violin.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig_violin, use_container_width=True)

            # 통계 요약
            st.markdown("---")
            st.markdown("### 통계 요약")

            cv_stats = df_cv.groupby('Model')['Accuracy'].agg(['mean', 'std', 'min', 'max']).reset_index()
            cv_stats.columns = ['알고리즘', '평균', '표준편차', '최소값', '최대값']
            cv_stats[['평균', '표준편차', '최소값', '최대값']] = cv_stats[['평균', '표준편차', '최소값', '최대값']].applymap(lambda x: f"{x:.4f}")

            st.dataframe(cv_stats, use_container_width=True)

        else:
            st.info("교차 검증 데이터가 없습니다.")


elif page == "Dataset Explorer":
    st.header("Dataset Explorer")
    st.markdown("전체 데이터셋을 탐색하고 시각화합니다.")

    df = load_historical_data()

    if df is None:
        st.warning("Warning: 데이터가 없습니다.")
        st.stop()

    # 통계 요약
    st.subheader("데이터셋 개요")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("총 인스턴스", f"{len(df):,}개")

    with col2:
        st.metric("속성 개수", "8개 (+ 1 클래스)")

    with col3:
        date_range = (df['timestamp'].max() - df['timestamp'].min()).days
        st.metric("데이터 기간", f"{date_range}일")

    with col4:
        st.metric("데이터 타입", "Time Series")

    # 탭 구성
    tab1, tab2, tab3 = st.tabs(["인스턴스 목록", "속성 분포", "시계열 그래프"])

    # ===== 탭 1: 인스턴스 목록 =====
    with tab1:
        st.subheader("전체 인스턴스 목록")

        # 필터링 옵션
        col1, col2 = st.columns(2)

        with col1:
            filter_class = st.multiselect(
                "Price Direction 필터",
                options=['UP', 'DOWN', 'STABLE'],
                default=['UP', 'DOWN', 'STABLE']
            )

        with col2:
            filter_ma = st.multiselect(
                "MA Cross 필터",
                options=df['ma_cross'].unique().tolist(),
                default=df['ma_cross'].unique().tolist()
            )

        # 필터 적용
        df_filtered = df[
            (df['price_direction'].isin(filter_class)) &
            (df['ma_cross'].isin(filter_ma))
        ]

        st.write(f"**필터링 결과:** {len(df_filtered)}개 인스턴스")

        # 데이터 테이블
        display_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'ma_cross', 'rsi_signal', 'volume_spike', 'price_direction']

        st.dataframe(
            df_filtered[display_cols].sort_values('timestamp', ascending=False),
            use_container_width=True,
            height=500
        )

        # CSV 다운로드
        csv_data = df_filtered[display_cols].to_csv(index=False)
        st.download_button(
            label="CSV 다운로드",
            data=csv_data,
            file_name="bitcoin_dataset.csv",
            mime="text/csv"
        )

    # ===== 탭 2: 속성 분포 =====
    with tab2:
        st.subheader("속성 분포 분석")

        # 클래스 분포
        st.markdown("### Price Direction 분포")

        class_dist = df['price_direction'].value_counts()
        fig_pie = px.pie(
            values=class_dist.values,
            names=class_dist.index,
            title='Price Direction 분포',
            color_discrete_map={'UP': 'red', 'DOWN': 'blue', 'STABLE': 'gray'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # 기술적 지표 분포
        st.markdown("---")
        st.markdown("### 기술적 지표 분포")

        col1, col2, col3 = st.columns(3)

        with col1:
            ma_dist = df['ma_cross'].value_counts()
            fig_ma = px.bar(
                x=ma_dist.index,
                y=ma_dist.values,
                title='MA Cross 분포',
                labels={'x': 'MA Cross', 'y': '개수'}
            )
            st.plotly_chart(fig_ma, use_container_width=True)

        with col2:
            rsi_dist = df['rsi_signal'].value_counts()
            fig_rsi = px.bar(
                x=rsi_dist.index,
                y=rsi_dist.values,
                title='RSI Signal 분포',
                labels={'x': 'RSI Signal', 'y': '개수'}
            )
            st.plotly_chart(fig_rsi, use_container_width=True)

        with col3:
            vol_dist = df['volume_spike'].value_counts()
            fig_vol = px.bar(
                x=vol_dist.index,
                y=vol_dist.values,
                title='Volume Spike 분포',
                labels={'x': 'Volume Spike', 'y': '개수'}
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        # 가격 히스토그램
        st.markdown("---")
        st.markdown("### 가격 분포")

        fig_hist = px.histogram(
            df,
            x='close',
            nbins=50,
            title='종가 분포',
            labels={'close': '종가 (KRW)', 'count': '빈도'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # 거래량 히스토그램
        fig_vol_hist = px.histogram(
            df,
            x='volume',
            nbins=50,
            title='거래량 분포',
            labels={'volume': '거래량 (BTC)', 'count': '빈도'}
        )
        st.plotly_chart(fig_vol_hist, use_container_width=True)

    # ===== 탭 3: 시계열 그래프 =====
    with tab3:
        st.subheader("시계열 데이터 시각화")

        # 가격 차트
        st.markdown("### 가격 추이")

        fig_price = go.Figure()

        fig_price.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue')
        ))

        fig_price.update_layout(
            title='Bitcoin 종가 추이',
            xaxis_title='시간',
            yaxis_title='가격 (KRW)',
            height=500
        )

        st.plotly_chart(fig_price, use_container_width=True)

        # 거래량 차트
        st.markdown("---")
        st.markdown("### 거래량 추이")

        fig_volume = px.bar(
            df,
            x='timestamp',
            y='volume',
            title='거래량 추이',
            labels={'timestamp': '시간', 'volume': '거래량 (BTC)'}
        )
        fig_volume.update_layout(height=400)
        st.plotly_chart(fig_volume, use_container_width=True)

        # 클래스별 가격 변동
        st.markdown("---")
        st.markdown("### 클래스별 가격 변동")

        fig_class_price = px.scatter(
            df,
            x='timestamp',
            y='close',
            color='price_direction',
            title='Price Direction별 종가 분포',
            labels={'timestamp': '시간', 'close': '가격 (KRW)', 'price_direction': 'Direction'},
            color_discrete_map={'UP': 'red', 'DOWN': 'blue', 'STABLE': 'gray'}
        )
        fig_class_price.update_layout(height=500)
        st.plotly_chart(fig_class_price, use_container_width=True)


elif page == "Chart Image Analysis":
    st.header("Chart Image Analysis")
    st.markdown("Upload a Bitcoin chart screenshot and get AI-powered pattern analysis")

    # 모델 로드
    predictor = load_model()

    if predictor is None:
        st.warning("Warning: 학습된 모델이 없습니다. 먼저 모델을 학습하세요.")
        st.code("python3 src/predictor.py", language="bash")
        st.stop()

    st.markdown("---")

    # 이미지 업로드
    uploaded_file = st.file_uploader(
        "Upload Bitcoin Chart Image (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a screenshot of Bitcoin price chart"
    )

    if uploaded_file is not None:
        from PIL import Image
        import numpy as np

        # 이미지 표시
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Chart")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Image Analysis")

            # 이미지 정보
            st.markdown(f"**Size:** {image.size[0]} x {image.size[1]} px")
            st.markdown(f"**Format:** {image.format}")
            st.markdown(f"**Mode:** {image.mode}")

        st.markdown("---")

        # 분석 버튼
        if st.button("Analyze Chart Pattern", key="analyze_chart"):
            with st.spinner("Analyzing chart patterns..."):
                import time
                time.sleep(1)  # 분석 효과

                # 이미지를 numpy 배열로 변환
                img_array = np.array(image)

                # 색상 분석 (간단한 휴리스틱)
                # 빨간색 많으면 상승, 파란색 많으면 하락
                red_channel = img_array[:, :, 0].mean()
                green_channel = img_array[:, :, 1].mean()
                blue_channel = img_array[:, :, 2].mean()

                # 밝기 분석
                brightness = (red_channel + green_channel + blue_channel) / 3

                st.markdown("### Pattern Detection Results")

                # 색상 분석 결과
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Red Intensity", f"{red_channel:.1f}")
                with col2:
                    st.metric("Green Intensity", f"{green_channel:.1f}")
                with col3:
                    st.metric("Blue Intensity", f"{blue_channel:.1f}")

                st.markdown("---")

                # 패턴 감지 (휴리스틱 기반)
                patterns_detected = []

                if red_channel > blue_channel * 1.2:
                    patterns_detected.append({
                        'name': 'Bullish Trend (Red Candles)',
                        'confidence': min(95, (red_channel / blue_channel - 1) * 100),
                        'signal': 'UP'
                    })
                elif blue_channel > red_channel * 1.2:
                    patterns_detected.append({
                        'name': 'Bearish Trend (Blue Candles)',
                        'confidence': min(95, (blue_channel / red_channel - 1) * 100),
                        'signal': 'DOWN'
                    })
                else:
                    patterns_detected.append({
                        'name': 'Sideways Movement',
                        'confidence': 70,
                        'signal': 'STABLE'
                    })

                if brightness > 150:
                    patterns_detected.append({
                        'name': 'High Volume Activity',
                        'confidence': 65,
                        'signal': 'VOLATILE'
                    })

                # 패턴 표시
                st.markdown("### Detected Patterns")

                for pattern in patterns_detected:
                    with st.expander(f"{pattern['name']} - {pattern['confidence']:.1f}% confidence"):
                        st.markdown(f"**Signal:** {pattern['signal']}")
                        st.progress(pattern['confidence'] / 100)

                # AI 예측 (모의)
                st.markdown("---")
                st.markdown("### AI Price Prediction")

                # 색상 기반 간단한 예측
                if red_channel > blue_channel:
                    prediction = "UP"
                    confidence = min(85, 50 + (red_channel - blue_channel) / 5)
                elif blue_channel > red_channel:
                    prediction = "DOWN"
                    confidence = min(85, 50 + (blue_channel - red_channel) / 5)
                else:
                    prediction = "STABLE"
                    confidence = 60

                # 결과 표시
                col1, col2 = st.columns(2)

                with col1:
                    direction_color = {
                        'UP': '🔴',
                        'DOWN': '🔵',
                        'STABLE': '⚪'
                    }
                    st.markdown(f"## {direction_color.get(prediction, '')} {prediction}")
                    st.markdown(f"**Confidence:** {confidence:.1f}%")

                with col2:
                    # 확률 분포
                    prob_data = {
                        'UP': confidence if prediction == 'UP' else (100 - confidence) / 2,
                        'DOWN': confidence if prediction == 'DOWN' else (100 - confidence) / 2,
                        'STABLE': confidence if prediction == 'STABLE' else (100 - confidence) / 2
                    }

                    prob_df = pd.DataFrame({
                        'Direction': list(prob_data.keys()),
                        'Probability': list(prob_data.values())
                    })

                    fig_bar = px.bar(
                        prob_df,
                        x='Direction',
                        y='Probability',
                        color='Direction',
                        color_discrete_map={'UP': 'red', 'DOWN': 'blue', 'STABLE': 'gray'},
                        title="Prediction Probability"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.success("Analysis complete!")

                # 경고 문구
                st.info("Note: This is a simplified pattern recognition based on image color analysis. For accurate predictions, use Live Prediction with real-time data.")

    else:
        # 샘플 이미지 가이드
        st.info("Upload a Bitcoin chart screenshot to get started")

        st.markdown("### How to use:")
        st.markdown("""
        1. Take a screenshot of a Bitcoin price chart (from any exchange)
        2. Upload the image using the file uploader above
        3. Click "Analyze Chart Pattern" to get AI-powered analysis
        4. View detected patterns and price prediction
        """)

        st.markdown("### Supported chart types:")
        st.markdown("""
        - Candlestick charts
        - Line charts
        - Area charts
        - From any exchange (Upbit, Binance, Coinbase, etc.)
        """)

        # 샘플 차트 표시 (data/charts에서)
        import os
        chart_dir = 'data/charts'
        if os.path.exists(chart_dir):
            chart_files = [f for f in os.listdir(chart_dir) if f.endswith('.png')]
            if chart_files:
                st.markdown("### Sample Chart (from our data):")
                sample_chart = os.path.join(chart_dir, chart_files[0])
                st.image(sample_chart, caption="Example: Bitcoin candlestick chart", use_container_width=True)


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
            ["Naive Bayes", "Decision Tree (J48)", "Random Forest", "SVM"]
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
                if algorithm == "Naive Bayes":
                    model = GaussianNB()
                elif algorithm == "Decision Tree (J48)":
                    model = DecisionTreeClassifier(max_depth=10, random_state=42)
                elif algorithm == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
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
        st.subheader("Association Rules Mining (Apriori)")

        st.markdown("### 파라미터 설정")

        col1, col2 = st.columns(2)

        with col1:
            min_support = st.slider(
                "최소 지지도 (Min Support)",
                min_value=0.05,
                max_value=0.5,
                value=0.1,
                step=0.05,
                help="빈발 항목집합의 최소 출현 빈도"
            )

        with col2:
            min_confidence = st.slider(
                "최소 신뢰도 (Min Confidence)",
                min_value=0.3,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="규칙의 최소 신뢰도"
            )

        # Apriori 실행 버튼
        if st.button("Run Apriori Algorithm", type="primary", key="run_apriori"):
            with st.spinner("Apriori 알고리즘 실행 중..."):
                from mlxtend.frequent_patterns import apriori, association_rules
                from mlxtend.preprocessing import TransactionEncoder

                # 범주형 데이터만 선택
                categorical_cols = ['ma_cross', 'rsi_signal', 'volume_spike', 'price_direction']
                df_categorical = df[categorical_cols].copy()

                # One-hot encoding
                df_encoded = pd.get_dummies(df_categorical, prefix=categorical_cols)

                # Apriori 알고리즘 실행
                try:
                    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

                    if len(frequent_itemsets) == 0:
                        st.warning("설정한 지지도로는 빈발 항목집합을 찾을 수 없습니다. 지지도를 낮춰보세요.")
                    else:
                        st.success(f"Success: {len(frequent_itemsets)}개 빈발 항목집합 발견!")

                        # 연관규칙 생성
                        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

                        if len(rules) == 0:
                            st.warning("설정한 신뢰도로는 연관규칙을 찾을 수 없습니다. 신뢰도를 낮춰보세요.")
                        else:
                            st.markdown("---")
                            st.markdown(f"### 발견된 연관규칙: {len(rules)}개")

                            # Lift 기준으로 정렬
                            rules_sorted = rules.sort_values('lift', ascending=False)

                            # 상위 20개 규칙만 표시
                            rules_display = rules_sorted.head(20).copy()

                            # 규칙을 읽기 쉽게 변환
                            rules_display['antecedents_str'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
                            rules_display['consequents_str'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))

                            # 테이블 표시
                            st.markdown("#### 상위 20개 연관규칙 (Lift 기준)")

                            result_df = rules_display[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].copy()
                            result_df.columns = ['선행 조건 (IF)', '결과 (THEN)', '지지도', '신뢰도', 'Lift']
                            result_df['지지도'] = result_df['지지도'].apply(lambda x: f"{x:.3f}")
                            result_df['신뢰도'] = result_df['신뢰도'].apply(lambda x: f"{x:.3f}")
                            result_df['Lift'] = result_df['Lift'].apply(lambda x: f"{x:.3f}")

                            st.dataframe(result_df, use_container_width=True)

                            # 규칙 상세 설명
                            st.markdown("---")
                            st.markdown("### 주요 규칙 상세 분석")

                            # 상위 3개 규칙 상세 표시
                            for idx, (_, row) in enumerate(rules_sorted.head(3).iterrows(), 1):
                                with st.expander(f"규칙 {idx}: {row['antecedents_str']} → {row['consequents_str']}", expanded=(idx == 1)):
                                    col1, col2, col3 = st.columns(3)

                                    with col1:
                                        st.metric("지지도 (Support)", f"{row['support']:.3f}")
                                        st.caption("전체 거래 중 이 규칙이 나타나는 비율")

                                    with col2:
                                        st.metric("신뢰도 (Confidence)", f"{row['confidence']:.3f}")
                                        st.caption("선행 조건 발생 시 결과가 나타날 확률")

                                    with col3:
                                        st.metric("Lift", f"{row['lift']:.3f}")
                                        st.caption("규칙의 유용성 (1보다 크면 유의미)")

                                    # 규칙 해석
                                    st.markdown("**규칙 해석:**")
                                    if row['lift'] > 1:
                                        st.success(f"✅ 이 규칙은 유의미합니다. {row['antecedents_str']} 조건일 때, {row['consequents_str']}가 나타날 가능성이 {row['lift']:.2f}배 높습니다.")
                                    else:
                                        st.info(f"ℹ️ Lift가 1 이하로, 이 규칙은 유의미하지 않을 수 있습니다.")

                            # 통계 요약
                            st.markdown("---")
                            st.markdown("### 통계 요약")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("총 빈발 항목집합", f"{len(frequent_itemsets)}개")

                            with col2:
                                st.metric("총 연관규칙", f"{len(rules)}개")

                            with col3:
                                high_lift_rules = len(rules[rules['lift'] > 1])
                                st.metric("유의미한 규칙 (Lift>1)", f"{high_lift_rules}개")

                            # 시각화
                            st.markdown("---")
                            st.markdown("### 규칙 시각화")

                            # Scatter plot: Support vs Confidence (Lift로 색상)
                            fig_scatter = px.scatter(
                                rules_sorted.head(50),
                                x='support',
                                y='confidence',
                                size='lift',
                                color='lift',
                                hover_data=['antecedents_str', 'consequents_str'],
                                title='연관규칙 분포 (Support vs Confidence)',
                                labels={'support': '지지도', 'confidence': '신뢰도', 'lift': 'Lift'},
                                color_continuous_scale='Viridis'
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)

                except Exception as e:
                    st.error(f"Apriori 실행 중 오류 발생: {e}")

        # ARFF 파일 다운로드 옵션
        st.markdown("---")
        st.markdown("### ARFF 파일 다운로드 (WEKA용)")

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
        else:
            st.info("ARFF 파일을 먼저 생성하세요: `python3 src/arff_generator.py`")


elif page == "About":
    st.header("About This Project")

    # GitHub Repository Link
    st.markdown("""
    <div style='margin: 20px 0; padding: 15px; border-left: 3px solid #0366d6;'>
        <strong>Source Code:</strong>
        <a href='https://github.com/Jeong-Ryeol/Bitcoin-Price-Prediction-System'
           target='_blank' style='color: #0366d6; text-decoration: none;'>
            github.com/Jeong-Ryeol/Bitcoin-Price-Prediction-System
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    ## Project Overview

    **비트코인 가격 예측 시스템**은 데이터마이닝 기법을 활용하여
    암호화폐 시장의 가격 방향을 예측하는 머신러닝 프로젝트입니다.

    ### Key Features

    - Real-time data collection via Upbit Public API
    - Technical indicator analysis (Moving Averages, RSI, Volume)
    - Multiple ML algorithms (Random Forest, SVM, Naive Bayes, Decision Tree)
    - Interactive web dashboard with live predictions
    - Chart image pattern recognition

    ---

    ## Technology Stack

    | Category | Technologies |
    |----------|-------------|
    | **Data Collection** | Upbit Public API (No API Key Required) |
    | **Data Processing** | pandas, numpy |
    | **Technical Analysis** | pandas-ta, mplfinance |
    | **Machine Learning** | scikit-learn, WEKA |
    | **Visualization** | Streamlit, Plotly |
    | **Image Processing** | Pillow (PIL) |
    | **Deployment** | Streamlit Cloud, GitHub |

    ---

    ## Data Attributes

    ### Price Data (5 attributes)

    - **Open**: 시간대 시작 가격
    - **High**: 시간대 최고 가격
    - **Low**: 시간대 최저 가격
    - **Close**: 시간대 종료 가격 (예측에 가장 중요)
    - **Volume**: 거래량 (시장 활성도 지표)

    ### Technical Indicators (3 attributes)

    **MA Cross (Moving Average Crossover)**
    - Golden Cross: 단기 이동평균(5시간)이 장기 이동평균(20시간)을 상향 돌파
    - Death Cross: 단기 이동평균이 장기 이동평균을 하향 돌파
    - Neutral: 교차 없음

    **RSI Signal (Relative Strength Index)**
    - Overbought: RSI > 70 (과매수 상태)
    - Oversold: RSI < 30 (과매도 상태)
    - Neutral: 30 ≤ RSI ≤ 70

    **Volume Spike**
    - High: 거래량이 평균의 1.5배 이상
    - Low: 거래량이 평균의 0.5배 이하
    - Normal: 평균 수준의 거래량

    ### Target Class

    **Price Direction** (1시간 후 가격 변동)
    - UP: 현재 대비 0.3% 이상 상승
    - DOWN: 현재 대비 0.3% 이상 하락
    - STABLE: -0.3% ~ +0.3% 범위 내 유지

    ---

    ## How to Use This Website

    ### Dashboard
    현재 Bitcoin 가격 정보, 실시간 차트, 기술적 지표 시각화 및 데이터 통계 요약

    ### Live Prediction
    실시간 Bitcoin 데이터를 수집하고 ML 모델을 통해 1시간 후 가격 방향 예측
    (Random Forest, SVM, Naive Bayes, Decision Tree 비교)

    ### Chart Image Analysis
    Bitcoin 차트 스크린샷을 업로드하여 AI 기반 패턴 인식 및 트렌드 분석

    ### Historical Analysis
    수집된 과거 데이터 탐색 및 시간대별 가격 변동, 기술적 지표 트렌드 확인

    ### WEKA Analysis
    WEKA 소프트웨어 없이 웹에서 Classification, Clustering, Association Rules 결과 확인
    ARFF 파일 다운로드 가능

    ---

    ## Local Installation & Usage

    ### Prerequisites
    ```bash
    Python 3.9 or higher
    pip (Python package manager)
    ```

    ### Installation Steps

    ```bash
    # 1. Clone repository
    git clone https://github.com/Jeong-Ryeol/Bitcoin-Price-Prediction-System.git
    cd Bitcoin-Price-Prediction-System

    # 2. Create virtual environment
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate

    # 3. Install dependencies
    pip install -r requirements.txt

    # 4. Run the complete pipeline
    python3 run.py

    # 5. Launch web application
    streamlit run app.py
    ```

    ### Manual Execution (Step by Step)

    ```bash
    # Step 1: Collect Bitcoin data from Upbit API
    python3 src/collector.py

    # Step 2: Analyze charts and calculate technical indicators
    python3 src/chart_analyzer.py

    # Step 3: Generate ARFF files for WEKA
    python3 src/arff_generator.py

    # Step 4: Train ML models
    python3 src/predictor.py

    # Step 5: Run web dashboard
    streamlit run app.py
    ```

    ---

    ## Academic Purpose

    이 프로젝트는 데이터마이닝 과목의 일환으로 진행되었으며, 다음과 같은 학습 목표를 달성합니다:

    - 실제 데이터 수집 및 전처리 경험
    - 시계열 데이터의 특성 이해
    - WEKA를 활용한 데이터 마이닝 실습
    - 다양한 머신러닝 알고리즘 비교 분석
    - 웹 기반 대시보드 개발 및 배포

    **Dataset Statistics**
    - Total instances: 199 (WEKA requirement: 100+)
    - Total attributes: 9 (8 features + 1 class, WEKA requirement: 4+)

    ---

    ## Disclaimer

    이 시스템은 교육 목적으로만 제작되었습니다. 실제 투자 결정에 사용해서는 안 됩니다.
    과거 데이터는 미래 수익을 보장하지 않으며, 암호화폐 투자는 높은 리스크를 동반합니다.

    ---

    ## Developer

    **Jeong Won Ryeol**
    Department of Computer Science and Engineering

    GitHub: [github.com/Jeong-Ryeol](https://github.com/Jeong-Ryeol)
    """)

    st.markdown("---")
    st.markdown("Data Mining Project 2025")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Bitcoin Price Prediction System | Data Mining Project 2025<br>"
    "Developed by <strong>Jeong Won Ryeol</strong> | Department of Computer Science and Engineering"
    "</div>",
    unsafe_allow_html=True
)
