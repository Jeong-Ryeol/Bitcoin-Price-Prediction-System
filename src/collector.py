"""
비트코인 과거 캔들 데이터 수집 모듈
Upbit API를 사용하여 과거 시간봉 데이터 수집
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import json


class BitcoinDataCollector:
    """Upbit API를 사용한 비트코인 캔들 데이터 수집기"""

    def __init__(self, market='KRW-BTC'):
        self.market = market
        self.base_url = 'https://api.upbit.com/v1'

    def get_candles_hour(self, count=200, to=None):
        """
        시간봉 데이터 수집

        Args:
            count: 수집할 캔들 개수 (1-200)
            to: 마지막 캔들 시각 (ISO 8601 format)

        Returns:
            DataFrame: 캔들 데이터
        """
        url = f'{self.base_url}/candles/minutes/60'
        params = {
            'market': self.market,
            'count': min(count, 200)  # 최대 200개
        }

        if to:
            params['to'] = to

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                return None

            # DataFrame으로 변환
            df = pd.DataFrame(data)

            # 필요한 컬럼만 선택 및 이름 변경
            df = df[['candle_date_time_kst', 'opening_price', 'high_price',
                     'low_price', 'trade_price', 'candle_acc_trade_volume']]

            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

            # timestamp를 datetime으로 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # 시간순으로 정렬 (오래된 것부터)
            df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            print(f"데이터 수집 오류: {e}")
            return None

    def collect_historical_data(self, hours=200, output_dir='data/raw'):
        """
        과거 데이터 대량 수집

        Args:
            hours: 수집할 시간 (최대 200)
            output_dir: 저장 디렉토리

        Returns:
            DataFrame: 수집된 데이터
        """
        print("=" * 70)
        print(f"📊 비트코인 과거 {hours}시간 데이터 수집 시작")
        print("=" * 70)

        df = self.get_candles_hour(count=hours)

        if df is None or len(df) == 0:
            print("❌ 데이터 수집 실패")
            return None

        print(f"\n✅ {len(df)}개 캔들 수집 완료")
        print(f"   기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        print(f"   가격 범위: {df['close'].min():,.0f}원 ~ {df['close'].max():,.0f}원")

        # 저장
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'bitcoin_candles.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n💾 저장 완료: {csv_path}")

        return df

    def add_future_labels(self, df, hours_ahead=1):
        """
        미래 가격 방향 라벨 추가 (예측 타겟)

        Args:
            df: 캔들 데이터
            hours_ahead: 몇 시간 후 예측할지

        Returns:
            DataFrame: 라벨이 추가된 데이터
        """
        # 미래 종가
        df['future_close'] = df['close'].shift(-hours_ahead)

        # 가격 변동률
        df['future_change_pct'] = ((df['future_close'] - df['close']) / df['close']) * 100

        # 클래스 라벨
        def classify_direction(change_pct):
            if pd.isna(change_pct):
                return None
            elif change_pct > 0.3:  # 0.3% 이상 상승
                return 'UP'
            elif change_pct < -0.3:  # 0.3% 이상 하락
                return 'DOWN'
            else:
                return 'STABLE'

        df['price_direction'] = df['future_change_pct'].apply(classify_direction)

        # 미래 데이터가 없는 마지막 행들 제거
        df = df[df['price_direction'].notna()].copy()

        print(f"\n🎯 클래스 라벨 추가 완료 ({hours_ahead}시간 후 예측)")
        print(f"   유효 데이터: {len(df)}개")

        # 클래스 분포
        class_dist = df['price_direction'].value_counts()
        print(f"\n   클래스 분포:")
        for label, count in class_dist.items():
            print(f"   - {label}: {count}개 ({count/len(df)*100:.1f}%)")

        return df


def main():
    """테스트 실행"""
    collector = BitcoinDataCollector()

    # 과거 200시간 데이터 수집
    df = collector.collect_historical_data(hours=200)

    if df is not None:
        # 1시간 후 가격 방향 라벨 추가
        df = collector.add_future_labels(df, hours_ahead=1)

        # 최종 데이터 저장
        output_path = 'data/raw/bitcoin_labeled.csv'
        df.to_csv(output_path, index=False)
        print(f"\n✨ 최종 데이터 저장: {output_path}")

        # 샘플 출력
        print(f"\n📝 데이터 샘플 (최근 5개):")
        print(df[['timestamp', 'close', 'volume', 'future_change_pct', 'price_direction']].tail())


if __name__ == "__main__":
    main()
