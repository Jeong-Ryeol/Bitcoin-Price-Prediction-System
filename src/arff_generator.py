"""
WEKA ARFF 파일 생성 모듈
차트 패턴 분석 결과를 WEKA 형식으로 변환
"""

import pandas as pd
import os


class ARFFGenerator:
    """WEKA ARFF 파일 생성기"""

    def __init__(self, df):
        """
        Args:
            df: 피처 데이터프레임 (chart_analyzer 출력)
        """
        self.df = df

    def save_classification_arff(self, filename='data/processed/bitcoin_classification.arff'):
        """
        분류(Classification)용 ARFF 생성
        - 7개 속성 (4 numeric + 3 categorical) + 1 class
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            # 헤더
            f.write("@RELATION bitcoin_price_prediction\n\n")

            # 속성 정의
            f.write("% 가격 데이터\n")
            f.write("@ATTRIBUTE open_price NUMERIC\n")
            f.write("@ATTRIBUTE high_price NUMERIC\n")
            f.write("@ATTRIBUTE low_price NUMERIC\n")
            f.write("@ATTRIBUTE close_price NUMERIC\n")
            f.write("@ATTRIBUTE volume NUMERIC\n")

            f.write("\n% 차트 패턴\n")
            f.write("@ATTRIBUTE ma_cross {golden,dead,neutral}\n")
            f.write("@ATTRIBUTE rsi_signal {overbought,oversold,neutral}\n")
            f.write("@ATTRIBUTE volume_spike {high,normal,low}\n")

            f.write("\n% 클래스 (1시간 후 가격 방향)\n")
            f.write("@ATTRIBUTE price_direction {UP,DOWN,STABLE}\n\n")

            # 데이터
            f.write("@DATA\n")
            for _, row in self.df.iterrows():
                line = (
                    f"{row['open']},"
                    f"{row['high']},"
                    f"{row['low']},"
                    f"{row['close']},"
                    f"{row['volume']},"
                    f"{row['ma_cross']},"
                    f"{row['rsi_signal']},"
                    f"{row['volume_spike']},"
                    f"{row['price_direction']}\n"
                )
                f.write(line)

        print(f"\n✅ 분류용 ARFF 생성: {filename}")
        print(f"   - {len(self.df)}개 인스턴스")
        print(f"   - 8개 속성 + 1개 클래스")
        return filename

    def save_clustering_arff(self, filename='data/processed/bitcoin_clustering.arff'):
        """
        군집화(Clustering)용 ARFF 생성
        - 클래스 제외, 숫자형 속성만 사용
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            # 헤더
            f.write("@RELATION bitcoin_clustering\n\n")

            # 속성 정의 (숫자형만)
            f.write("@ATTRIBUTE open_price NUMERIC\n")
            f.write("@ATTRIBUTE high_price NUMERIC\n")
            f.write("@ATTRIBUTE low_price NUMERIC\n")
            f.write("@ATTRIBUTE close_price NUMERIC\n")
            f.write("@ATTRIBUTE volume NUMERIC\n\n")

            # 데이터
            f.write("@DATA\n")
            for _, row in self.df.iterrows():
                line = (
                    f"{row['open']},"
                    f"{row['high']},"
                    f"{row['low']},"
                    f"{row['close']},"
                    f"{row['volume']}\n"
                )
                f.write(line)

        print(f"\n✅ 군집화용 ARFF 생성: {filename}")
        print(f"   - {len(self.df)}개 인스턴스")
        print(f"   - 5개 숫자형 속성")
        return filename

    def save_association_arff(self, filename='data/processed/bitcoin_association.arff'):
        """
        연관규칙(Association)용 ARFF 생성
        - 모든 속성을 범주형으로 변환
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            # 헤더
            f.write("@RELATION bitcoin_association\n\n")

            # 속성 정의 (모두 범주형)
            f.write("@ATTRIBUTE price_level {very_low,low,medium,high,very_high}\n")
            f.write("@ATTRIBUTE volume_level {low,medium,high}\n")
            f.write("@ATTRIBUTE ma_cross {golden,dead,neutral}\n")
            f.write("@ATTRIBUTE rsi_signal {overbought,oversold,neutral}\n")
            f.write("@ATTRIBUTE volume_spike {high,normal,low}\n")
            f.write("@ATTRIBUTE price_direction {UP,DOWN,STABLE}\n\n")

            # 데이터
            f.write("@DATA\n")

            # 가격 레벨 계산을 위한 분위수
            price_quantiles = self.df['close'].quantile([0.2, 0.4, 0.6, 0.8])

            for _, row in self.df.iterrows():
                # 가격 레벨 분류
                price = row['close']
                if price < price_quantiles[0.2]:
                    price_level = 'very_low'
                elif price < price_quantiles[0.4]:
                    price_level = 'low'
                elif price < price_quantiles[0.6]:
                    price_level = 'medium'
                elif price < price_quantiles[0.8]:
                    price_level = 'high'
                else:
                    price_level = 'very_high'

                # 거래량 레벨
                volume = row['volume']
                volume_median = self.df['volume'].median()
                volume_q75 = self.df['volume'].quantile(0.75)

                if volume < volume_median * 0.5:
                    volume_level = 'low'
                elif volume < volume_q75:
                    volume_level = 'medium'
                else:
                    volume_level = 'high'

                line = (
                    f"{price_level},"
                    f"{volume_level},"
                    f"{row['ma_cross']},"
                    f"{row['rsi_signal']},"
                    f"{row['volume_spike']},"
                    f"{row['price_direction']}\n"
                )
                f.write(line)

        print(f"\n✅ 연관규칙용 ARFF 생성: {filename}")
        print(f"   - {len(self.df)}개 인스턴스")
        print(f"   - 6개 범주형 속성")
        return filename

    def save_all_formats(self):
        """모든 형식의 ARFF 파일 생성"""
        print("=" * 70)
        print("📁 WEKA ARFF 파일 생성 중...")
        print("=" * 70)

        files = []
        files.append(self.save_classification_arff())
        files.append(self.save_clustering_arff())
        files.append(self.save_association_arff())

        print("\n" + "=" * 70)
        print("✨ 모든 ARFF 파일 생성 완료!")
        print("=" * 70)

        # 클래스 분포
        class_dist = self.df['price_direction'].value_counts()
        print(f"\n📊 클래스 분포:")
        for label, count in class_dist.items():
            print(f"   - {label}: {count}개 ({count/len(self.df)*100:.1f}%)")

        return files


def main():
    """테스트 실행"""
    # chart_analyzer에서 생성한 데이터 로드
    input_path = 'data/processed/bitcoin_features.csv'

    if not os.path.exists(input_path):
        print(f"❌ 파일이 없습니다: {input_path}")
        print("   먼저 src/chart_analyzer.py를 실행하세요.")
        return

    print("=" * 70)
    print("📄 ARFF 파일 생성 시작")
    print("=" * 70)

    # 데이터 로드
    df = pd.read_csv(input_path)
    print(f"\n✅ 데이터 로드 완료: {len(df)}개 인스턴스")

    # ARFF 생성기
    generator = ARFFGenerator(df)

    # 모든 형식 생성
    files = generator.save_all_formats()

    print(f"\n💡 다음 파일들을 WEKA에서 열어보세요:")
    for f in files:
        print(f"   - {f}")


if __name__ == "__main__":
    main()
