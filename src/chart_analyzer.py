"""
차트 패턴 분석 및 기술적 지표 계산 모듈
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import mplfinance as mpf
import os
from datetime import datetime


class ChartAnalyzer:
    """차트 패턴 인식 및 기술적 지표 계산"""

    def __init__(self, df):
        """
        Args:
            df: OHLCV 데이터 (open, high, low, close, volume 컬럼 필수)
        """
        self.df = df.copy()

    def calculate_technical_indicators(self):
        """기술적 지표 계산"""

        # 1. 이동평균선 (Moving Averages)
        self.df['ma5'] = self.df['close'].rolling(window=5).mean()
        self.df['ma20'] = self.df['close'].rolling(window=20).mean()

        # 2. RSI (Relative Strength Index)
        self.df['rsi'] = ta.rsi(self.df['close'], length=14)

        # 3. 볼린저 밴드 (Bollinger Bands)
        bbands = ta.bbands(self.df['close'], length=20, std=2)
        if bbands is not None and not bbands.empty:
            # 컬럼명이 버전마다 다를 수 있으므로 유연하게 처리
            cols = bbands.columns.tolist()
            if len(cols) >= 3:
                self.df['bb_lower'] = bbands.iloc[:, 0]
                self.df['bb_middle'] = bbands.iloc[:, 1]
                self.df['bb_upper'] = bbands.iloc[:, 2]

        # 4. MACD
        macd = ta.macd(self.df['close'])
        if macd is not None and not macd.empty:
            # 컬럼명이 버전마다 다를 수 있으므로 유연하게 처리
            cols = macd.columns.tolist()
            if len(cols) >= 3:
                self.df['macd'] = macd.iloc[:, 0]
                self.df['macd_hist'] = macd.iloc[:, 1]
                self.df['macd_signal'] = macd.iloc[:, 2]

        # 5. 거래량 이동평균
        self.df['volume_ma'] = self.df['volume'].rolling(window=20).mean()

        print("✅ 기술적 지표 계산 완료")
        return self.df

    def detect_patterns(self):
        """차트 패턴 인식 (규칙 기반)"""

        patterns = []

        for idx in range(20, len(self.df)):  # 최소 20개 데이터 필요
            row = self.df.iloc[idx]

            # 1. MA Cross (골든크로스 / 데드크로스)
            if pd.notna(row['ma5']) and pd.notna(row['ma20']):
                if row['ma5'] > row['ma20']:
                    ma_cross = 'golden'  # 골든크로스 (상승 신호)
                elif row['ma5'] < row['ma20']:
                    ma_cross = 'dead'  # 데드크로스 (하락 신호)
                else:
                    ma_cross = 'neutral'
            else:
                ma_cross = 'neutral'

            # 2. RSI Signal (과매수 / 과매도)
            if pd.notna(row['rsi']):
                if row['rsi'] > 70:
                    rsi_signal = 'overbought'  # 과매수
                elif row['rsi'] < 30:
                    rsi_signal = 'oversold'  # 과매도
                else:
                    rsi_signal = 'neutral'
            else:
                rsi_signal = 'neutral'

            # 3. Volume Spike (거래량 급증)
            if pd.notna(row['volume_ma']) and row['volume_ma'] > 0:
                volume_ratio = row['volume'] / row['volume_ma']
                if volume_ratio > 2.0:
                    volume_spike = 'high'
                elif volume_ratio < 0.5:
                    volume_spike = 'low'
                else:
                    volume_spike = 'normal'
            else:
                volume_spike = 'normal'

            patterns.append({
                'ma_cross': ma_cross,
                'rsi_signal': rsi_signal,
                'volume_spike': volume_spike
            })

        # 앞부분은 NaN으로 채움
        for _ in range(20):
            patterns.insert(0, {
                'ma_cross': 'neutral',
                'rsi_signal': 'neutral',
                'volume_spike': 'normal'
            })

        # DataFrame에 추가
        pattern_df = pd.DataFrame(patterns)
        self.df['ma_cross'] = pattern_df['ma_cross']
        self.df['rsi_signal'] = pattern_df['rsi_signal']
        self.df['volume_spike'] = pattern_df['volume_spike']

        print("✅ 차트 패턴 인식 완료")
        return self.df

    def create_chart_image(self, index, output_dir='data/charts'):
        """
        특정 시점의 차트 이미지 생성

        Args:
            index: DataFrame 인덱스
            output_dir: 저장 디렉토리
        """
        os.makedirs(output_dir, exist_ok=True)

        # 해당 시점 기준 과거 50개 캔들
        start_idx = max(0, index - 50)
        end_idx = index + 1

        chart_df = self.df.iloc[start_idx:end_idx].copy()
        chart_df.set_index('timestamp', inplace=True)

        # 이동평균선 추가
        apds = []
        if 'ma5' in chart_df.columns:
            apds.append(mpf.make_addplot(chart_df['ma5'], color='blue', width=1))
        if 'ma20' in chart_df.columns:
            apds.append(mpf.make_addplot(chart_df['ma20'], color='red', width=1))

        # 차트 스타일
        mc = mpf.make_marketcolors(up='r', down='b', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=False)

        # 저장 경로
        timestamp = self.df.iloc[index]['timestamp']
        filename = f"chart_{timestamp.strftime('%Y%m%d_%H%M')}.png"
        filepath = os.path.join(output_dir, filename)

        # 차트 생성
        mpf.plot(
            chart_df[['open', 'high', 'low', 'close', 'volume']],
            type='candle',
            style=s,
            addplot=apds if apds else None,
            volume=True,
            savefig=filepath,
            figsize=(10, 6),
            tight_layout=True
        )

        return filepath

    def get_feature_dataset(self):
        """
        WEKA용 피처 데이터셋 반환

        Returns:
            DataFrame: 학습에 사용할 피처들
        """
        # 필요한 컬럼만 선택
        features = [
            'timestamp',
            'open', 'high', 'low', 'close', 'volume',
            'ma_cross', 'rsi_signal', 'volume_spike'
        ]

        # 라벨 컬럼이 있으면 추가
        if 'price_direction' in self.df.columns:
            features.append('price_direction')

        result_df = self.df[features].copy()

        # NaN 제거 (초반 지표 계산 안된 부분)
        result_df = result_df.dropna(subset=['ma_cross', 'rsi_signal'])

        print(f"\n✅ 피처 데이터셋 준비 완료: {len(result_df)}개 인스턴스")

        return result_df


def main():
    """테스트 실행"""
    # collector에서 생성한 데이터 로드
    input_path = 'data/raw/bitcoin_labeled.csv'

    if not os.path.exists(input_path):
        print(f"❌ 파일이 없습니다: {input_path}")
        print("   먼저 src/collector.py를 실행하세요.")
        return

    print("=" * 70)
    print("📈 차트 분석 시작")
    print("=" * 70)

    # 데이터 로드
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 분석기 생성
    analyzer = ChartAnalyzer(df)

    # 지표 계산
    analyzer.calculate_technical_indicators()

    # 패턴 인식
    analyzer.detect_patterns()

    # 피처 데이터셋 저장
    feature_df = analyzer.get_feature_dataset()
    output_path = 'data/processed/bitcoin_features.csv'
    os.makedirs('data/processed', exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    print(f"\n💾 저장 완료: {output_path}")

    # 샘플 차트 생성 (최근 5개)
    print(f"\n🖼️  샘플 차트 생성 중...")
    for i in range(max(0, len(feature_df) - 5), len(feature_df)):
        filepath = analyzer.create_chart_image(i)
        print(f"   - {filepath}")

    # 패턴 분포 확인
    print(f"\n📊 패턴 분포:")
    print(f"\n   MA Cross:")
    print(feature_df['ma_cross'].value_counts())
    print(f"\n   RSI Signal:")
    print(feature_df['rsi_signal'].value_counts())
    print(f"\n   Volume Spike:")
    print(feature_df['volume_spike'].value_counts())


if __name__ == "__main__":
    main()
