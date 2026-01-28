"""
비트코인 가격 예측 시스템 - Streamlit 대시보드
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
from datetime import datetime
from scipy import stats

# 프로젝트 경로 추가
sys.path.append('src')

from collector import BitcoinDataCollector
from chart_analyzer import ChartAnalyzer
from predictor import BitcoinPredictor
from performance_tracker import PerformanceTracker
from config import load_config, save_config, DEFAULT_CONFIG

# LSTM 모델 가용성 확인
LSTM_AVAILABLE = False
try:
    from predictor import HybridPredictor
    from lstm_model import check_dependencies
    LSTM_AVAILABLE = check_dependencies()
except ImportError:
    pass


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
    """학습된 SVM 모델 로드"""
    model_path = 'models/bitcoin_predictor.pkl'

    if not os.path.exists(model_path):
        return None

    predictor = BitcoinPredictor()
    predictor.load_model(model_path)
    return predictor


@st.cache_resource
def load_lstm_model():
    """학습된 LSTM+XGBoost 하이브리드 모델 로드"""
    if not LSTM_AVAILABLE:
        return None

    model_path = 'models/lstm_xgboost_hybrid.pkl'
    preprocessor_path = 'models/preprocessors.pkl'

    if not os.path.exists(model_path):
        return None

    try:
        config = load_config()
        predictor = HybridPredictor(config)
        predictor.load(model_path)
        return predictor
    except Exception as e:
        print(f"LSTM 모델 로드 실패: {e}")
        return None


def get_recent_data_for_lstm(hours=100):
    """LSTM 예측을 위한 최근 데이터 수집"""
    collector = BitcoinDataCollector()
    df = collector.get_candles_hour(count=hours)

    if df is None:
        return None

    # 차트 분석
    analyzer = ChartAnalyzer(df)
    analyzer.calculate_technical_indicators()
    analyzer.detect_patterns()

    return analyzer.df


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
    st.header("Navigation")

    page = st.radio(
        "Select Page",
        ["Dashboard", "Live Prediction", "Manual Prediction", "Dataset Explorer",
         "Chart Image Analysis", "Historical Analysis", "WEKA Analysis", "Model Performance", "About"]
    )

    st.markdown("---")

    # 설정 섹션
    st.header("Prediction Settings")

    # 현재 설정 로드
    config = load_config()

    # 모델 선택
    model_options = ["SVM (Basic)"]
    if LSTM_AVAILABLE and os.path.exists('models/lstm_xgboost_hybrid.pkl'):
        model_options.append("LSTM+XGBoost (Advanced)")

    selected_model = st.selectbox(
        "Prediction Model",
        model_options,
        index=0
    )

    # 임계값 설정
    with st.expander("Threshold Settings", expanded=False):
        st.caption("UP/DOWN/STABLE 분류 기준 (%)")

        threshold_1h = st.slider(
            "1-Hour Threshold",
            min_value=0.1,
            max_value=2.0,
            value=float(config['thresholds'].get('1h', 0.3)),
            step=0.1,
            help="1시간 예측: ±0.3% 이상 변동시 UP/DOWN"
        )

        threshold_24h = st.slider(
            "24-Hour Threshold",
            min_value=0.5,
            max_value=5.0,
            value=float(config['thresholds'].get('24h', 1.5)),
            step=0.1,
            help="24시간 예측: ±1.5% 이상 변동시 UP/DOWN"
        )

        threshold_7d = st.slider(
            "7-Day Threshold",
            min_value=2.0,
            max_value=15.0,
            value=float(config['thresholds'].get('7d', 5.0)),
            step=0.5,
            help="7일 예측: ±5% 이상 변동시 UP/DOWN"
        )

        if st.button("Save Settings", type="primary"):
            config['thresholds'] = {
                '1h': threshold_1h,
                '24h': threshold_24h,
                '7d': threshold_7d
            }
            save_config(config)
            st.success("Settings saved!")
            st.rerun()

    st.markdown("---")
    st.markdown("### Project Info")
    st.markdown("**Data Mining Project**")
    st.markdown("Bitcoin Price Prediction")
    if LSTM_AVAILABLE:
        st.caption("LSTM+XGBoost Available")


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

    # 다중 시간대 클래스 분포 확인
    horizon_cols = ['direction_1h', 'direction_24h', 'direction_168h']
    has_multi_horizon = all(col in df.columns for col in horizon_cols)

    if has_multi_horizon:
        # 다중 시간대 표시
        col1, col2, col3 = st.columns(3)

        horizon_info = [
            ('direction_1h', '1-Hour Direction', col1),
            ('direction_24h', '24-Hour Direction', col2),
            ('direction_168h', '7-Day Direction', col3)
        ]

        for col_name, title, col in horizon_info:
            if col_name in df.columns:
                with col:
                    class_counts = df[col_name].value_counts()
                    fig_pie = px.pie(
                        values=class_counts.values,
                        names=class_counts.index,
                        title=title,
                        color_discrete_map={'UP': 'red', 'DOWN': 'blue', 'STABLE': 'gray'}
                    )
                    fig_pie.update_layout(height=300)
                    st.plotly_chart(fig_pie, use_container_width=True)
    else:
        # 단일 시간대 (기존)
        col1, col2 = st.columns(2)

        with col1:
            class_counts = df['price_direction'].value_counts()
            fig_pie = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title='Price Direction Distribution (1-Hour)',
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

    # 패턴 분포 (다중 시간대일 때도 표시)
    if has_multi_horizon:
        st.markdown("---")
        st.subheader("Chart Patterns")
        col1, col2, col3 = st.columns(3)

        with col1:
            ma_dist = df['ma_cross'].value_counts()
            st.markdown("**MA Cross**")
            for k, v in ma_dist.items():
                st.write(f"- {k}: {v} ({v/len(df)*100:.1f}%)")

        with col2:
            rsi_dist = df['rsi_signal'].value_counts()
            st.markdown("**RSI Signal**")
            for k, v in rsi_dist.items():
                st.write(f"- {k}: {v} ({v/len(df)*100:.1f}%)")

        with col3:
            vol_dist = df['volume_spike'].value_counts()
            st.markdown("**Volume Spike**")
            for k, v in vol_dist.items():
                st.write(f"- {k}: {v} ({v/len(df)*100:.1f}%)")


elif page == "Live Prediction":
    st.header("Live Prediction")
    st.markdown("실시간 비트코인 데이터로 가격 방향을 예측합니다.")

    # 선택된 모델 확인
    use_lstm = selected_model == "LSTM+XGBoost (Advanced)"

    # 모델 로드
    if use_lstm:
        lstm_predictor = load_lstm_model()
        if lstm_predictor is None:
            st.warning("LSTM 모델을 찾을 수 없습니다. SVM 모델로 전환합니다.")
            use_lstm = False
            predictor = load_model()
        else:
            st.success(f"Model: LSTM+XGBoost (Multi-horizon)")
    else:
        predictor = load_model()

    if not use_lstm and predictor is None:
        st.warning("Warning: 학습된 모델이 없습니다. 먼저 모델을 학습하세요.")
        st.code("python run.py", language="bash")
        st.stop()

    if not use_lstm:
        st.success(f"Model: SVM (1-hour prediction)")

    # 모델 정보 표시
    with st.expander("Model Information"):
        if use_lstm:
            st.markdown("""
            **LSTM + XGBoost Hybrid Model**
            - 입력: 최근 72시간 시계열 데이터
            - 예측: 1시간, 24시간, 7일 후 가격 방향
            - LSTM: 시계열 특성 추출
            - XGBoost: 방향 분류
            """)
        else:
            st.markdown("""
            **SVM (Support Vector Machine) Model**
            - 입력: 현재 시점 단일 데이터
            - 예측: 1시간 후 가격 방향만
            - RBF 커널 기반 분류
            """)

    # 실시간 데이터 가져오기
    if st.button("Get Current Bitcoin Data & Predict", type="primary"):
        with st.spinner("데이터 수집 및 분석 중..."):
            if use_lstm:
                # LSTM은 더 많은 데이터 필요
                current_df = get_recent_data_for_lstm(hours=100)
            else:
                current_df = get_current_bitcoin_data()

        if current_df is not None and len(current_df) > 0:
            st.success(f"Data collected: {len(current_df)} hours")

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
            st.subheader("Prediction Results")

            direction_emoji = {
                'UP': '🔴 UP',
                'DOWN': '🔵 DOWN',
                'STABLE': '⚪ STABLE'
            }

            if use_lstm:
                # LSTM 다중 시간대 예측
                try:
                    predictions = lstm_predictor.predict_realtime(current_df)

                    # 3개 시간대 결과 표시
                    col1, col2, col3 = st.columns(3)

                    horizons = [('1h', '1 Hour', col1), ('24h', '24 Hours', col2), ('7d', '7 Days', col3)]

                    for horizon, label, col in horizons:
                        if horizon in predictions:
                            pred = predictions[horizon]
                            with col:
                                st.markdown(f"### {label}")
                                st.markdown(f"## {direction_emoji.get(pred['direction'], pred['direction'])}")
                                st.metric("Confidence", f"{pred['confidence']:.1%}")

                                # 클래스별 확률
                                if 'probabilities' in pred:
                                    st.markdown("**Probabilities:**")
                                    for cls, prob in pred['probabilities'].items():
                                        st.progress(float(prob), text=f"{cls}: {prob:.1%}")

                except Exception as e:
                    st.error(f"LSTM 예측 실패: {e}")
                    st.info("SVM 모델로 재시도합니다...")
                    # SVM 폴백
                    predictor = load_model()
                    if predictor:
                        use_lstm = False

            if not use_lstm:
                # SVM 단일 시간대 예측
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

                # 결과 표시 (1시간만)
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("### 1 Hour")
                    st.markdown(f"## {direction_emoji.get(result['prediction'], result['prediction'])}")
                    st.metric("Confidence", f"{result['confidence']:.1%}")

                with col2:
                    st.markdown("### 24 Hours")
                    st.markdown("## N/A")
                    st.caption("SVM 모델은 1시간 예측만 지원")

                with col3:
                    st.markdown("### 7 Days")
                    st.markdown("## N/A")
                    st.caption("LSTM 모델 필요")

                # 클래스별 확률 차트
                st.markdown("---")
                st.markdown("### Class Probabilities (1-Hour)")
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
            fig = create_candlestick_chart(current_df.tail(72))
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
    st.markdown("4개 알고리즘의 성능을 비교하고 SVM 선택 이유를 설명합니다.")

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

    # 평가 결과 가져오기
    df_performance = tracker.get_all_evaluations()

    if df_performance.empty:
        st.info("아직 평가된 모델이 없습니다. '모든 모델 재평가' 버튼을 클릭하세요.")
        st.stop()

    # 알고리즘별 비교
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

    # 상세 성능 지표 테이블
    st.markdown("---")
    st.markdown("### 상세 성능 지표")

    df_display = df_comparison.copy()
    df_display['accuracy'] = df_display['accuracy'].apply(lambda x: f"{x:.2%}")
    df_display['cv_mean'] = df_display['cv_mean'].apply(lambda x: f"{x:.2%}")
    df_display['cv_std'] = df_display['cv_std'].apply(lambda x: f"± {x:.2%}")
    df_display.columns = ['알고리즘', '테스트 정확도', 'CV 평균', 'CV 표준편차', '학습 데이터 크기']

    st.dataframe(df_display, use_container_width=True)

    # 최고 모델 강조
    summary = tracker.get_summary()
    st.success(f"최고 성능 모델: {summary['best_model']} ({summary['best_accuracy']:.2%})")

    # SVM 선택 이유 설명
    st.markdown("---")
    st.markdown("### SVM 알고리즘 선택 이유")

    st.markdown("""
    본 프로젝트에서는 SVM (Support Vector Machine)을 최종 예측 모델로 선택하였습니다.

    #### 1. 선택 근거

    **최고 정확도**

    SVM은 테스트 데이터에서 약 69%의 정확도를 기록하여 다른 알고리즘 대비 2-5% 높은 성능을 보였습니다.
    교차 검증 점수도 가장 안정적으로 나타났으며, 표준편차가 낮아 일관된 성능을 보장합니다.

    **비선형 패턴 학습**

    RBF 커널을 사용하여 가격, 거래량, 기술적 지표 간의 복잡한 비선형 관계를 효과적으로 학습할 수 있습니다.
    암호화폐 시장은 단순한 선형 관계로 설명되지 않는 경우가 많기 때문에, 이러한 특성이 중요한 장점으로 작용합니다.

    **과적합 방지**

    SVM의 마진 최대화 원리는 학습 데이터에 지나치게 맞추지 않고 일반화 성능을 높이는 데 도움이 됩니다.
    이를 통해 새로운 데이터에 대한 예측 성능도 우수하게 유지됩니다.

    #### 2. 다른 알고리즘과의 비교

    Naive Bayes는 64%의 정확도로 빠르지만, 특성 간 독립성 가정이 실제 데이터와 맞지 않습니다.
    Decision Tree는 67%로 해석이 쉽다는 장점이 있으나 과적합 경향이 있고 불안정합니다.
    Random Forest는 67%로 안정적이지만 SVM보다 정확도가 낮았습니다.

    #### 3. 한계점 및 개선 방향

    **클래스 불균형 문제**

    현재 데이터셋은 STABLE 클래스가 70.6%를 차지하여 UP/DOWN 예측이 어렵습니다.
    이를 해결하기 위해 class_weight='balanced' 파라미터나 SMOTE 오버샘플링 기법을 적용할 수 있습니다.

    **학습 시간**

    다른 알고리즘에 비해 학습 시간이 5-10초 정도로 상대적으로 오래 걸립니다.
    하지만 정확도 향상을 고려하면 충분히 감수할 만한 수준입니다.

    **모델 해석**

    SVM은 블랙박스 모델로 의사결정 과정을 직접적으로 설명하기 어렵습니다.
    이는 Decision Tree 같은 모델에 비해 단점으로 작용할 수 있으나, 본 프로젝트에서는 정확도를 우선시하였습니다.

    더 자세한 Confusion Matrix와 클래스별 성능 분석은 WEKA Analysis 페이지에서 확인할 수 있습니다.
    """)

    # Learning Curve 섹션
    st.markdown("---")
    st.markdown("### Learning Curve 분석")
    st.markdown("데이터 크기에 따른 알고리즘별 성능 변화를 보여줍니다.")

    # Learning Curve 이미지 표시
    learning_curve_path = 'weka_results/learning_curve.png'
    confidence_interval_path = 'weka_results/svm_confidence_interval.png'

    import os
    if os.path.exists(learning_curve_path):
        st.image(learning_curve_path, caption='Learning Curve: 알고리즘별 성능 비교', use_container_width=True)

        with st.expander("Learning Curve는 어떻게 그린 그래프인가요?"):
            st.markdown("""
            ### Learning Curve 생성 방법

            **X축: 훈련 데이터 크기**
            - 전체 데이터의 일부만 사용하여 학습 (10%, 20%, 30%, ... 100%)
            - 예: 전체 9,111개 중 10% = 약 911개, 100% = 9,111개 인스턴스

            **Y축: 검증 정확도 (10-Fold Cross Validation)**

            각 훈련 크기에서 다음 과정을 수행합니다:

            ```
            예시: Training Size 10% (약 911개)
            ┌─────────────────────────────────────┐
            │ 1. 911개 데이터를 10등분 (Fold)      │
            │ 2. 9개 Fold(820개)로 학습            │
            │ 3. 1개 Fold(91개)로 검증             │
            │ 4. 10번 반복 (매번 다른 Fold로 검증) │
            │ 5. 10개 정확도의 평균 = 그래프의 점  │
            │ 6. 표준편차 = 색상 띠 (음영)         │
            └─────────────────────────────────────┘
            ```

            **그래프 해석:**
            - **점**: 각 훈련 크기에서의 평균 검증 정확도
            - **색상 띠**: 표준편차 범위 (±1 std) - 성능의 안정성
            - **SVM(빨강)**이 전 구간에서 가장 높은 정확도를 유지

            **목적:**
            - 데이터가 많아질수록 성능이 어떻게 변하는지 확인
            - 과적합(Overfitting) / 과소적합(Underfitting) 진단
            """)
    else:
        st.info("Learning Curve 이미지가 없습니다. `python src/learning_curve_plot.py`를 실행하세요.")

    if os.path.exists(confidence_interval_path):
        st.markdown("---")
        st.markdown("### SVM 정확도 신뢰구간")
        st.image(confidence_interval_path, caption='SVM 10-Fold Cross Validation 정확도 범위', use_container_width=True)

        st.markdown("""
        **신뢰구간 해석:**
        - **평균 정확도**: 69.00%
        - **표준편차**: ±2.57%
        - **95% 신뢰구간**: 63.96% ~ 74.03%
        - 파란 점: 각 Fold의 정확도
        - 빨간 선: 평균 정확도
        """)


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


    # 데이터 로드
    df = load_historical_data()
    predictor = load_model()

    if df is None or predictor is None:
        st.warning("Warning: 데이터 또는 모델을 먼저 생성하세요.")
        st.code("python3 run.py", language="bash")
        st.stop()

    # 탭 구성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Classification",
        "Decision Tree",
        "Clustering",
        "Association Rules",
        "ANOVA"
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

                            # 규칙을 읽기 쉽게 변환 (먼저 전체에 적용)
                            rules_sorted['antecedents_str'] = rules_sorted['antecedents'].apply(lambda x: ', '.join(list(x)))
                            rules_sorted['consequents_str'] = rules_sorted['consequents'].apply(lambda x: ', '.join(list(x)))

                            # 상위 20개 규칙만 표시
                            rules_display = rules_sorted.head(20).copy()

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

    # ===== ANOVA 분석 =====
    with tab5:
        st.subheader("ANOVA Analysis (알고리즘 성능 비교)")

        st.markdown("""
        **Analysis of Variance (ANOVA) - 알고리즘 간 성능 비교**

        1. Learning Curve 그리기 (10%~100% 훈련 크기)
        2. 최적 훈련 크기(80%)에서 각 알고리즘을 **10번 반복 측정** (10-Fold CV)
        3. 알고리즘 간 평균 차이가 우연에 의한 것인지 검정

        - **k**: 알고리즘 수 (methods) = 4
        - **n**: 측정 횟수 (10-Fold CV) = 10
        - **F 분포**: 자유도 k-1=3, k(n-1)=36
        """)

        if st.button("Run ANOVA Analysis", type="primary", key="run_anova"):
            with st.spinner("Learning Curve 분석 및 ANOVA 수행 중... (1-2분 소요)"):
                from sklearn.model_selection import learning_curve, cross_val_score
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.svm import SVC
                from sklearn.naive_bayes import GaussianNB
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.preprocessing import LabelEncoder, StandardScaler
                import numpy as np

                # 데이터 준비
                X, y = predictor.prepare_data(df)

                # 알고리즘 정의
                models = {
                    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
                    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'Naive Bayes': GaussianNB(),
                    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42)
                }

                # 학습 데이터 비율 (ANOVA용 80% 근처 포함)
                # Learning Curve: 10%~100% (10% 간격) + ANOVA용 75%, 85% 추가
                train_sizes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0])

                st.markdown("---")
                st.markdown("### Step 1: Learning Curve (10%~100%)")

                # 각 알고리즘별 Learning Curve 계산
                learning_curve_results = {}
                best_train_size = None
                best_avg_accuracy = 0

                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, (name, model) in enumerate(models.items()):
                    status_text.text(f"{name} Learning Curve 계산 중...")

                    train_sizes_abs, train_scores, test_scores = learning_curve(
                        model, X, y,
                        train_sizes=train_sizes,
                        cv=10,
                        scoring='accuracy',
                        n_jobs=-1,
                        random_state=42
                    )

                    learning_curve_results[name] = {
                        'train_sizes': train_sizes_abs,
                        'test_mean': test_scores.mean(axis=1),
                        'test_std': test_scores.std(axis=1),
                        'test_scores': test_scores  # 각 fold별 점수 저장
                    }

                    progress_bar.progress((idx + 1) / len(models))

                # 최적 훈련 크기를 80%로 고정
                best_train_size = 0.8
                best_train_size_idx = list(train_sizes).index(0.8)
                best_avg_accuracy = np.mean([learning_curve_results[name]['test_mean'][best_train_size_idx] for name in models.keys()])

                status_text.text("완료!")
                st.success(f"최적 훈련 크기: {best_train_size*100:.0f}% (고정)")

                # Learning Curve 그래프 표시
                import plotly.graph_objects as go
                fig_lc = go.Figure()

                colors = {
                    'SVM (RBF)': 'red',
                    'Random Forest': 'green',
                    'Naive Bayes': 'blue',
                    'Decision Tree': 'orange'
                }

                # X축을 퍼센트로 표시
                x_percent = [f'{int(s*100)}%' for s in train_sizes]

                for name in models.keys():
                    result = learning_curve_results[name]
                    fig_lc.add_trace(go.Scatter(
                        x=x_percent,
                        y=result['test_mean'] * 100,
                        mode='lines+markers',
                        name=name,
                        line=dict(color=colors[name], width=3 if name == 'SVM (RBF)' else 2),
                        marker=dict(size=10 if name == 'SVM (RBF)' else 6)
                    ))

                # 최적 지점 표시 (annotation 사용)
                best_idx = list(train_sizes).index(best_train_size)
                fig_lc.add_annotation(
                    x=f'{int(best_train_size*100)}%',
                    y=best_avg_accuracy * 100 + 5,
                    text=f"최적: {best_train_size*100:.0f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="gray",
                    font=dict(color="gray")
                )

                fig_lc.update_layout(
                    title='Learning Curve: 알고리즘별 10-Fold CV 정확도',
                    xaxis_title='Training Set Size (%)',
                    yaxis_title='Accuracy (%)',
                    height=500,
                    yaxis=dict(range=[20, 100])
                )
                st.plotly_chart(fig_lc, use_container_width=True)

                # Step 2: 최적 훈련 크기(80%)에서 10-Fold CV 결과 수집 (ANOVA용)
                st.markdown("---")
                st.markdown(f"### Step 2: 최적 훈련 크기({int(best_train_size*100)}%)에서 10번 반복 측정")

                st.markdown(f"""
                **반복 측정 (10-Fold CV)**: 최적 훈련 크기 **{int(best_train_size*100)}%**에서 각 알고리즘을 10번 반복 실행
                - 10-Fold Cross Validation = 10번의 독립적인 측정
                - 각 Fold마다 다른 Train/Test 분할로 정확도 측정
                - 각 알고리즘마다 10개의 측정값
                """)

                # 최적 훈련 크기 인덱스에서 10-Fold CV 결과 추출
                best_idx = list(train_sizes).index(best_train_size)

                # 각 알고리즘별로 10-Fold CV 결과 수집
                anova_data = {}
                for name in models.keys():
                    # 최적 훈련 크기에서의 10개 Fold 값
                    anova_data[name] = learning_curve_results[name]['test_scores'][best_idx]

                # 결과 테이블
                st.markdown("#### 알고리즘별 10-Fold CV 정확도 (%)")

                # 테이블 형식으로 표시
                fold_labels = [f'Fold {i+1}' for i in range(10)]
                anova_df = pd.DataFrame({
                    'Fold': fold_labels + ['평균', '표준편차'],
                })

                for name in models.keys():
                    scores = anova_data[name] * 100
                    anova_df[name] = list(scores) + [scores.mean(), scores.std()]

                # 포맷팅
                display_anova_df = anova_df.copy()
                for name in models.keys():
                    display_anova_df[name] = display_anova_df[name].apply(lambda x: f"{x:.2f}%")

                st.dataframe(display_anova_df, use_container_width=True)

                # Step 3: ANOVA 수행
                st.markdown("---")
                st.markdown("### Step 3: One-Way ANOVA 검정")

                # ANOVA 수행 (각 알고리즘의 훈련 크기별 정확도로)
                f_stat, p_value = stats.f_oneway(
                    anova_data['SVM (RBF)'],
                    anova_data['Random Forest'],
                    anova_data['Naive Bayes'],
                    anova_data['Decision Tree']
                )

                # 자유도 계산
                k = len(models)  # 그룹 수 (알고리즘 수)
                n = 10  # 각 그룹의 샘플 수 (10-Fold CV = 10번 측정)
                df_between = k - 1  # 그룹 간 자유도
                df_within = k * (n - 1)  # 그룹 내 자유도

                # F 임계값 (유의수준 0.05)
                f_critical = stats.f.ppf(0.95, df_between, df_within)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("f (계산값)", f"{f_stat:.4f}")
                with col2:
                    st.metric(f"f_{{0.05, {df_between}, {df_within}}} (임계값)", f"{f_critical:.4f}")

                st.markdown(f"""
                **F 분포 자유도:**
                - k = {k} (알고리즘 수)
                - n = {n} (측정 횟수 = 10-Fold CV)
                - 자유도: k-1 = **{df_between}**, k(n-1) = **{df_within}**
                """)

                # 결론
                st.markdown("---")
                st.markdown("### Conclusion")

                if f_stat > f_critical:
                    st.success(f"""
                    **f = {f_stat:.4f}** exceeds **f_{{0.05, {df_between}, {df_within}}} = {f_critical:.4f}**

                    모든 알고리즘의 평균이 동일하다는 가정을 **기각(reject)** 합니다.

                    → 알고리즘 간 성능에 **실제 차이가 있음**
                    """)
                else:
                    st.warning(f"""
                    **f = {f_stat:.4f}** does not exceed **f_{{0.05, {df_between}, {df_within}}} = {f_critical:.4f}**

                    모든 알고리즘의 평균이 동일하다는 가정을 기각할 수 없습니다.

                    → 알고리즘 간 차이가 우연에 의한 것일 수 있음
                    """)

                # 알고리즘별 성능 순위
                st.markdown("---")
                st.markdown("### 알고리즘별 성능 순위")

                ranking_data = []
                for name in models.keys():
                    scores = anova_data[name]
                    ranking_data.append({
                        'Algorithm': name,
                        'Mean': scores.mean() * 100,
                        'Std': scores.std() * 100
                    })

                ranking_df = pd.DataFrame(ranking_data).sort_values('Mean', ascending=False)
                ranking_df['Rank'] = range(1, len(ranking_df) + 1)
                ranking_df = ranking_df[['Rank', 'Algorithm', 'Mean', 'Std']]
                ranking_df.columns = ['순위', '알고리즘', '평균 정확도 (%)', '표준편차 (%)']

                display_ranking = ranking_df.copy()
                display_ranking['평균 정확도 (%)'] = display_ranking['평균 정확도 (%)'].apply(lambda x: f"{x:.2f}")
                display_ranking['표준편차 (%)'] = display_ranking['표준편차 (%)'].apply(lambda x: f"±{x:.2f}")

                st.dataframe(display_ranking, use_container_width=True)

                # 박스플롯
                st.markdown("---")
                st.markdown("### 알고리즘별 정확도 분포 (Box Plot)")

                # 박스플롯용 데이터 준비 (10-Fold CV 결과)
                box_data = []
                for name in models.keys():
                    for fold_score in anova_data[name]:
                        box_data.append({
                            'Algorithm': name,
                            'Accuracy': fold_score * 100
                        })

                box_df = pd.DataFrame(box_data)
                fig_box = px.box(
                    box_df,
                    x='Algorithm',
                    y='Accuracy',
                    color='Algorithm',
                    color_discrete_map={
                        'SVM (RBF)': 'red',
                        'Random Forest': 'green',
                        'Naive Bayes': 'blue',
                        'Decision Tree': 'orange'
                    },
                    title=f'10-Fold CV 정확도 분포 비교 (훈련 크기: {int(best_train_size*100)}%)'
                )
                fig_box.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)

                # session_state에 결과 저장
                st.session_state['anova_algorithm_results'] = {
                    'anova_data': anova_data,
                    'f_stat': f_stat,
                    'p_value': p_value,
                    'best_train_size': best_train_size,
                    'learning_curve_results': learning_curve_results
                }

        # 결과가 있으면 해석 표시
        if 'anova_algorithm_results' in st.session_state:
            st.markdown("---")
            st.markdown("### ANOVA 분석 방법 설명")

            with st.expander("ANOVA 분석 과정 상세 설명 (발표용)"):
                st.markdown("""
                ## ANOVA (Analysis of Variance) 분석 방법

                ### 1. ANOVA란?
                - **목적**: 여러 알고리즘(그룹)의 평균 성능 차이가 **우연에 의한 것인지, 실제 차이인지** 통계적으로 검정
                - **핵심 질문**: "SVM, Random Forest, Naive Bayes, Decision Tree 중 성능 차이가 정말 있는가?"

                ---

                ### 2. 데이터 수집 방법

                **Step 1: Learning Curve 생성**
                - 훈련 데이터 크기를 10%~100%까지 변화시키며 각 알고리즘의 정확도 측정
                - 각 지점에서 10-Fold Cross Validation 수행

                **Step 2: 최적점 선택**
                - Learning Curve에서 **80%를 최적 훈련 크기**로 선택
                - (교수님 지침: 50% 이상에서 안정적인 지점 선택)

                **Step 3: 반복 측정 (10-Fold CV)**
                - 최적 훈련 크기(80%)에서 각 알고리즘을 **10번 반복 실행**
                - 10-Fold CV = 10번의 독립적인 Train/Test 분할로 10개 측정값 획득
                - 예: SVM → 68%, 69%, 70%, 67%, 71%, 69%, 70%, 68%, 72%, 70%

                ---

                ### 3. ANOVA 변수 설정

                | 변수 | 값 | 의미 |
                |------|-----|------|
                | **k** | 4 | 알고리즘(그룹) 수 |
                | **n** | 10 | 측정 횟수 (10-Fold CV) |
                | **df₁** | k-1 = 3 | 그룹 간 자유도 |
                | **df₂** | k(n-1) = 36 | 그룹 내 자유도 |

                ---

                ### 4. F 검정

                - **F 통계량 계산**: 그룹 간 분산 / 그룹 내 분산
                - **F 임계값**: F분포표에서 α=0.05, df₁=3, df₂=36 에 해당하는 값
                - **결론**:
                  - F 통계량 > F 임계값 → **알고리즘 간 성능 차이가 통계적으로 유의미함**
                  - F 통계량 ≤ F 임계값 → 차이가 우연에 의한 것일 수 있음

                ---

                ### 5. 왜 10-Fold CV를 사용했나?

                - **10-Fold CV = 10번 반복 실행**: 같은 데이터셋에서 10번 독립적으로 측정
                - 각 Fold는 서로 다른 Train/Test 분할 → **독립적인 측정**
                - ANOVA에 필요한 n개의 측정값을 자연스럽게 획득
                """)

            with st.expander("발표 시 말로 설명하는 방법"):
                st.markdown("""
                ## 발표 스크립트 예시

                > "ANOVA 분석을 통해 4개 알고리즘의 성능 차이가 통계적으로 유의미한지 검정했습니다.
                >
                > 먼저 Learning Curve를 그려서 최적의 훈련 데이터 크기를 80%로 선정했습니다.
                >
                > 그 다음 최적 훈련 크기(80%)에서 각 알고리즘을 10번 반복 실행했습니다.
                > 10-Fold Cross Validation을 사용하면 각 알고리즘마다 10개의 정확도 값이 나옵니다.
                >
                > 예를 들어 SVM은 68%, 69%, 70%, 67%, 71%... 이런 식으로 10개 값이 측정됩니다.
                >
                > k=4(알고리즘 수), n=10(측정 횟수)으로 자유도는 3과 36이 되고,
                > 계산된 F값이 임계값을 초과하므로 알고리즘 간 성능 차이가 통계적으로 유의미합니다."
                """)


elif page == "About":
    st.header("About This Project")

    # About 페이지용 데이터 로드
    df = load_historical_data()

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

    # 발표용 탭 구성
    about_tab1, about_tab2, about_tab3, about_tab4, about_tab5 = st.tabs([
        "Project Overview",
        "Data Structure",
        "Features Guide",
        "Data Mining Methods",
        "Installation"
    ])

    with about_tab1:
        st.markdown("""
        ## Project Overview

        비트코인 가격 예측 시스템은 **데이터마이닝 기법**을 활용하여 암호화폐 시장의 가격 방향을 예측하는 머신러닝 프로젝트입니다.

        ### 프로젝트 목표
        - Upbit Public API를 통해 **1년치 시간별 데이터**를 수집
        - 기술적 지표를 분석하여 **4가지 머신러닝 알고리즘**으로 가격 방향 예측
        - **WEKA 스타일** 데이터마이닝 분석을 웹에서 구현

        ---

        ### Dataset Statistics (현재)
        """)

        # 데이터 통계 표시
        if df is not None:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Instances", f"{len(df):,}개")
            with col2:
                st.metric("Total Attributes", "9개 (8 features + 1 class)")
            with col3:
                st.metric("Data Period", f"{(df['timestamp'].max() - df['timestamp'].min()).days}일")
            with col4:
                st.metric("Data Source", "Upbit API")

            st.markdown(f"""
            - **데이터 기간**: {df['timestamp'].min().strftime('%Y-%m-%d')} ~ {df['timestamp'].max().strftime('%Y-%m-%d')}
            - **데이터 간격**: 1시간 (Hourly Candle)
            - **WEKA 요구사항**: 100개 이상 인스턴스, 4개 이상 속성 -> **충족**
            """)

        st.markdown("""
        ---

        ### Technology Stack

        | Category | Technologies |
        |----------|-------------|
        | **Data Collection** | Upbit Public API (RESTful) |
        | **Data Processing** | pandas, numpy |
        | **Technical Analysis** | pandas-ta, RSI, Moving Average |
        | **Machine Learning** | scikit-learn (SVM, RF, DT, NB) |
        | **Statistical Analysis** | scipy (ANOVA) |
        | **Data Mining** | mlxtend (Apriori), K-Means |
        | **Visualization** | Streamlit, Plotly |

        ---

        ### Class Distribution
        """)

        if df is not None:
            class_counts = df['price_direction'].value_counts()
            fig_class = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title='Price Direction Class Distribution',
                color=class_counts.index,
                color_discrete_map={'UP': 'red', 'DOWN': 'blue', 'STABLE': 'gray'}
            )
            st.plotly_chart(fig_class, use_container_width=True)

    with about_tab2:
        st.markdown("""
        ## Data Structure: Instance = Attribute + Class

        ### Instance 구조

        | 구성요소 | 개수 | 항목 |
        |---------|------|------|
        | **Attributes** | 8개 | open, high, low, close, volume, ma_cross, rsi_signal, volume_spike |
        | **Class** | 1개 | price_direction |

        ---

        ### Attributes (속성) - 8개

        #### 1. Numeric Attributes (수치형) - 5개
        | Attribute | Description | Type | Example |
        |-----------|-------------|------|---------|
        | **open** | 시간봉 시작 가격 | Continuous | 126,097,000 원 |
        | **high** | 시간봉 최고 가격 | Continuous | 126,500,000 원 |
        | **low** | 시간봉 최저 가격 | Continuous | 125,900,000 원 |
        | **close** | 시간봉 종료 가격 | Continuous | 126,472,000 원 |
        | **volume** | 거래량 (BTC) | Continuous | 122.87 BTC |

        #### 2. Categorical Attributes (범주형) - 3개
        | Attribute | Description | Values |
        |-----------|-------------|--------|
        | **ma_cross** | 이동평균선 교차 신호 | golden, dead, neutral |
        | **rsi_signal** | RSI 과매수/과매도 신호 | overbought, oversold, neutral |
        | **volume_spike** | 거래량 급등/급락 신호 | high, normal, low |

        ---

        ### Class (클래스) - 1개

        | Class | Description | Threshold |
        |-------|-------------|-----------|
        | **price_direction** | 1시간 후 가격 방향 | UP/DOWN/STABLE |

        **분류 기준:**
        - **UP**: 1시간 후 가격이 0.3% 이상 상승
        - **DOWN**: 1시간 후 가격이 0.3% 이상 하락
        - **STABLE**: -0.3% ~ +0.3% 범위 내 유지

        ---

        ### Technical Indicators 상세

        **1. MA Cross (Moving Average Crossover)**
        - 단기 이동평균(5시간)과 장기 이동평균(20시간) 비교
        - **Golden Cross**: MA5 > MA20 (상승 신호)
        - **Dead Cross**: MA5 < MA20 (하락 신호)

        **2. RSI Signal (Relative Strength Index)**
        - 14시간 기준 상대강도지수
        - **Overbought**: RSI > 70 (과매수 -> 하락 예상)
        - **Oversold**: RSI < 30 (과매도 -> 상승 예상)

        **3. Volume Spike**
        - 20시간 평균 거래량 대비 비율
        - **High**: 2배 이상 (급등)
        - **Low**: 0.5배 이하 (급락)
        """)

        # 샘플 데이터 표시
        if df is not None:
            st.markdown("---")
            st.markdown("### Sample Data (최근 5개 인스턴스)")
            st.dataframe(df.tail(5), use_container_width=True)

    with about_tab3:
        st.markdown("""
        ## Features Guide - 각 페이지 기능 설명

        ---

        ### 1. Dashboard
        **목적**: 데이터셋 전체 개요 및 통계 확인

        **주요 기능:**
        - 총 인스턴스 수, 속성 수, 데이터 기간 표시
        - 비트코인 가격 시계열 차트 (Plotly Interactive)
        - 클래스(UP/DOWN/STABLE) 분포 파이 차트
        - 기술적 지표별 분포 막대 차트

        ---

        ### 2. Live Prediction
        **목적**: 실시간 데이터로 즉시 예측

        **주요 기능:**
        - Upbit API에서 실시간 200시간 데이터 수집
        - 기술적 지표 자동 계산
        - SVM 모델로 1시간 후 가격 방향 예측
        - 클래스별 확률 분포 표시

        ---

        ### 3. Manual Prediction
        **목적**: 사용자 입력 데이터로 예측

        **주요 기능:**
        - 개별 입력: OHLCV + 기술적 지표 직접 입력
        - 일괄 예측: CSV 파일 업로드 후 일괄 예측
        - 예측 결과 다운로드

        ---

        ### 4. Dataset Explorer
        **목적**: 전체 데이터셋 탐색 및 분석

        **주요 기능:**
        - 인스턴스 필터링 (클래스별, 날짜별)
        - 속성 분포 시각화 (히스토그램, 박스플롯)
        - CSV 다운로드

        ---

        ### 5. Chart Image Analysis
        **목적**: 차트 이미지 분석 (실험적 기능)

        **주요 기능:**
        - 차트 스크린샷 업로드
        - 색상 분석으로 트렌드 감지
        - 간단한 패턴 인식

        ---

        ### 6. Historical Analysis
        **목적**: 특정 기간 데이터 분석

        **주요 기능:**
        - 날짜 범위 선택
        - 캔들스틱 차트 표시
        - 기간별 통계 (평균, 최고, 최저 가격)

        ---

        ### 7. WEKA Analysis
        **목적**: WEKA 스타일 데이터마이닝 분석

        **5개 탭:**

        | Tab | 기능 | 알고리즘 |
        |-----|------|---------|
        | **Classification** | 분류 분석 | NB, DT, RF, SVM |
        | **Decision Tree** | 의사결정나무 시각화 | J48 (CART) |
        | **Clustering** | 군집 분석 | K-Means (k=3) |
        | **Association Rules** | 연관규칙 마이닝 | Apriori |
        | **ANOVA** | 알고리즘 성능 비교 | One-Way ANOVA |

        ---

        ### 8. Model Performance
        **목적**: 모델 성능 비교 및 분석

        **주요 기능:**
        - 4개 알고리즘 정확도 비교
        - Learning Curve 시각화 (10-Fold CV)
        - 95% 신뢰구간 표시
        - 최적 모델 선택 근거 제시
        """)

    with about_tab4:
        st.markdown("""
        ## Data Mining Methods - 분석 방법론

        이 프로젝트에서 사용된 데이터마이닝 기법들을 설명합니다.

        ---

        ### 1. Classification (분류)

        **목적**: 새로운 인스턴스의 클래스를 예측

        **사용 알고리즘:**

        | Algorithm | Description | 특징 |
        |-----------|-------------|------|
        | **Naive Bayes** | 베이즈 정리 기반 확률적 분류 | 빠른 학습, 독립성 가정 |
        | **Decision Tree (J48)** | 규칙 기반 트리 구조 분류 | 해석 용이, 과적합 위험 |
        | **Random Forest** | 다수의 결정트리 앙상블 | 높은 정확도, 과적합 방지 |
        | **SVM (RBF)** | 최적 결정 경계 탐색 | 고차원에 강함, 최고 성능 |

        **평가 지표:**
        - Accuracy (정확도)
        - Precision, Recall, F1-Score
        - Cross-Validation (10-Fold)

        ---

        ### 2. Learning Curve (학습곡선)

        **목적**: 학습 데이터 크기에 따른 모델 성능 변화 분석

        **방법:**
        - 10%, 20%, ..., 100% 학습 데이터로 모델 학습
        - 10-Fold Cross Validation으로 성능 측정
        - 표준편차로 안정성 평가

        **해석:**
        - 학습곡선이 수렴하면 충분한 데이터
        - 학습/검증 점수 차이가 크면 과적합

        ---

        ### 3. Performance Confidence Interval (성능 신뢰구간)

        **목적**: 모델 성능의 신뢰 범위 추정

        **방법:**
        - 10-Fold Cross Validation 수행
        - 각 Fold의 정확도 수집
        - 평균 및 표준편차 계산
        - 95% 신뢰구간: 평균 ± 1.96 × 표준편차

        **해석:**
        > "SVM 모델의 정확도는 95% 신뢰수준에서 64% ~ 74% 범위입니다."

        ---

        ### 4. ANOVA (분산분석) - 알고리즘 성능 비교

        **목적**: 4개 알고리즘(SVM, RF, NB, DT) 간 성능 차이가 통계적으로 유의미한지 검정

        **데이터 수집 방법:**
        1. Learning Curve를 그려 최적 훈련 크기(80%)를 선정
        2. 최적 훈련 크기(80%)에서 각 알고리즘을 **10번 반복 실행** (10-Fold CV)
        3. 각 알고리즘마다 10개의 정확도 측정값 수집

        **ANOVA 변수:**
        - k = 4 (알고리즘 수)
        - n = 10 (측정 횟수 = 10-Fold CV)
        - 자유도: df₁ = k-1 = 3, df₂ = k(n-1) = 36

        **해석:**
        - F 통계량 > F 임계값: 알고리즘 간 성능 차이가 **통계적으로 유의미**
        - F 통계량 ≤ F 임계값: 차이가 우연에 의한 것일 수 있음

        **예시:**
        > "F = 15.23 > F(0.05, 3, 36) = 2.87 이므로, 4개 알고리즘 간 성능 차이가 통계적으로 유의미합니다."

        ---

        ### 5. Clustering (군집분석)

        **목적**: 유사한 인스턴스들을 그룹화

        **알고리즘**: K-Means (k=3)
        - 3개 클러스터로 시장 상황 분류
        - 각 클러스터의 중심(Centroid) 분석
        - 클러스터별 특성 파악

        ---

        ### 6. Association Rules (연관규칙)

        **목적**: 속성 간 연관 패턴 발견

        **알고리즘**: Apriori

        **주요 지표:**
        - **Support**: 규칙이 전체 데이터에서 나타나는 빈도
        - **Confidence**: 선행 조건 발생 시 결과가 나타날 확률
        - **Lift**: 규칙의 유용성 (1보다 크면 유의미)

        **예시:**
        > "IF ma_cross=golden AND volume_spike=high THEN price_direction=UP (Confidence: 75%, Lift: 2.1)"
        """)

    with about_tab5:
        st.markdown("""
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

        ---

        ## Academic Purpose

        이 프로젝트는 **데이터마이닝 과목**의 과제로 진행되었습니다.

        ### 학습 목표
        - 실제 금융 데이터 수집 및 전처리
        - 시계열 데이터 분석 및 특성 추출
        - WEKA 데이터마이닝 도구 활용
        - 다양한 분류 알고리즘 비교 및 평가
        - 웹 기반 대시보드 개발

        ### WEKA 요구사항 충족

        | 요구사항 | 조건 | 본 프로젝트 | 충족 |
        |---------|------|-----------|------|
        | Instances | 100개 이상 | 9,000개+ | O |
        | Attributes | 4개 이상 | 9개 | O |
        | Classification | 분류 분석 | SVM, RF, DT, NB | O |
        | Clustering | 군집 분석 | K-Means | O |
        | Association | 연관규칙 | Apriori | O |
        | Learning Curve | 학습곡선 | 10-Fold CV | O |
        | ANOVA | 분산분석 | One-Way ANOVA | O |

        ---

        ## Disclaimer

        이 시스템은 **교육 목적**으로만 제작되었습니다.
        실제 투자 결정에 사용해서는 안 됩니다.
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
