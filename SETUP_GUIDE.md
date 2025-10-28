# 비트코인 감정 분석 프로젝트 - VS Code 터미널 가이드

## 📋 VS Code 터미널에서 순서대로 실행하세요

---

## 1단계: 폴더 이동
```bash
cd ~/Desktop/project/datamining
```

---

## 2단계: Python 가상환경 생성 (선택사항이지만 권장)
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3단계: 필요한 라이브러리 설치
```bash
pip install requests beautifulsoup4 pandas textblob praw
```

---

## 4단계: TextBlob 데이터 다운로드 (한 번만)
```bash
python3 -m textblob.download_corpora
```

---

## 5단계: requirements.txt 파일 생성
```bash
cat > requirements.txt << 'EOF'
requests==2.31.0
beautifulsoup4==4.12.2
pandas==2.1.0
textblob==0.17.1
praw==7.7.1
EOF
```

---

## 6단계: collector.py 파일 생성 (데이터 수집 코드)
```bash
cat > collector.py << 'EOF'
import requests
import praw
from bs4 import BeautifulSoup
from textblob import TextBlob
import time
from datetime import datetime

class FreeCryptoCollector:

    def __init__(self):
        # Reddit 설정 (선택사항 - 나중에 설정 가능)
        self.reddit = None
        # self.reddit = praw.Reddit(
        #     client_id="YOUR_CLIENT_ID",
        #     client_secret="YOUR_SECRET",
        #     user_agent="crypto_analyzer"
        # )

    def get_upbit_data(self, symbol='KRW-BTC'):
        """업비트 API - 완전 무료"""
        try:
            url = 'https://api.upbit.com/v1/ticker'
            params = {'markets': symbol}
            response = requests.get(url, params=params, timeout=10)
            data = response.json()[0]

            return {
                'current_price': data['trade_price'],
                'volume_24h': data['acc_trade_volume_24h'],
                'price_change_pct': data['signed_change_rate'] * 100,
                'high_price': data['high_price'],
                'low_price': data['low_price']
            }
        except Exception as e:
            print(f"업비트 데이터 수집 실패: {e}")
            return None

    def get_coingecko_sentiment(self, coin_id='bitcoin'):
        """CoinGecko API - 완전 무료"""
        try:
            url = f'https://api.coingecko.com/api/v3/coins/{coin_id}'
            response = requests.get(url, timeout=10)
            data = response.json()

            return {
                'sentiment_up_votes': data.get('sentiment_votes_up_percentage', 50),
                'sentiment_down_votes': data.get('sentiment_votes_down_percentage', 50),
                'reddit_subscribers': data.get('community_data', {}).get('reddit_subscribers', 0),
                'reddit_48h_posts': data.get('community_data', {}).get('reddit_average_posts_48h', 0),
                'reddit_48h_comments': data.get('community_data', {}).get('reddit_average_comments_48h', 0)
            }
        except Exception as e:
            print(f"CoinGecko 데이터 수집 실패: {e}")
            return {
                'sentiment_up_votes': 50,
                'sentiment_down_votes': 50,
                'reddit_subscribers': 0,
                'reddit_48h_posts': 0,
                'reddit_48h_comments': 0
            }

    def get_news_count(self, keyword='비트코인'):
        """네이버 뉴스 크롤링 - 무료"""
        try:
            url = f'https://search.naver.com/search.naver?where=news&query={keyword}'
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            news_items = soup.select('.news_area')

            positive_words = ['상승', '급등', '호재', '긍정', '상향', '오른', '증가']
            negative_words = ['하락', '급락', '악재', '부정', '하향', '떨어', '감소']

            positive_count = 0
            negative_count = 0

            for item in news_items[:20]:
                text = item.get_text()
                positive_count += sum(word in text for word in positive_words)
                negative_count += sum(word in text for word in negative_words)

            total = positive_count + negative_count + 1
            return {
                'news_count': len(news_items),
                'positive_news_ratio': positive_count / total
            }
        except Exception as e:
            print(f"뉴스 크롤링 실패: {e}")
            return {
                'news_count': 0,
                'positive_news_ratio': 0.5
            }

    def collect_one_instance(self):
        """하나의 인스턴스 데이터 수집"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 데이터 수집 중...")

        upbit = self.get_upbit_data('KRW-BTC')
        if upbit is None:
            return None

        time.sleep(1)  # API 제한 방지

        coingecko = self.get_coingecko_sentiment('bitcoin')
        time.sleep(1)

        news = self.get_news_count('비트코인')

        # 데이터 통합
        instance = {**upbit, **coingecko, **news}

        # 클래스 라벨링 (가격 변동률 기준)
        if instance['price_change_pct'] > 2:
            instance['price_direction'] = 'UP'
        elif instance['price_change_pct'] < -2:
            instance['price_direction'] = 'DOWN'
        else:
            instance['price_direction'] = 'STABLE'

        return instance

if __name__ == "__main__":
    collector = FreeCryptoCollector()
    data = collector.collect_one_instance()
    print("수집된 데이터:", data)
EOF
```

---

## 7단계: arff_generator.py 파일 생성 (ARFF 변환 코드)
```bash
cat > arff_generator.py << 'EOF'
def save_to_arff(dataset, filename='bitcoin_sentiment.arff'):
    """데이터셋을 WEKA ARFF 형식으로 저장"""

    with open(filename, 'w', encoding='utf-8') as f:
        # 헤더 작성
        f.write("@RELATION bitcoin_sentiment\n\n")

        # 속성 정의
        f.write("@ATTRIBUTE current_price NUMERIC\n")
        f.write("@ATTRIBUTE volume_24h NUMERIC\n")
        f.write("@ATTRIBUTE price_change_pct NUMERIC\n")
        f.write("@ATTRIBUTE high_price NUMERIC\n")
        f.write("@ATTRIBUTE low_price NUMERIC\n")
        f.write("@ATTRIBUTE sentiment_up_votes NUMERIC\n")
        f.write("@ATTRIBUTE sentiment_down_votes NUMERIC\n")
        f.write("@ATTRIBUTE reddit_subscribers NUMERIC\n")
        f.write("@ATTRIBUTE reddit_48h_posts NUMERIC\n")
        f.write("@ATTRIBUTE reddit_48h_comments NUMERIC\n")
        f.write("@ATTRIBUTE news_count NUMERIC\n")
        f.write("@ATTRIBUTE positive_news_ratio NUMERIC\n")
        f.write("@ATTRIBUTE price_direction {UP,DOWN,STABLE}\n\n")

        # 데이터 작성
        f.write("@DATA\n")

        for row in dataset:
            line = f"{row['current_price']},"
            line += f"{row['volume_24h']},"
            line += f"{row['price_change_pct']},"
            line += f"{row['high_price']},"
            line += f"{row['low_price']},"
            line += f"{row['sentiment_up_votes']},"
            line += f"{row['sentiment_down_votes']},"
            line += f"{row['reddit_subscribers']},"
            line += f"{row['reddit_48h_posts']},"
            line += f"{row['reddit_48h_comments']},"
            line += f"{row['news_count']},"
            line += f"{row['positive_news_ratio']},"
            line += f"{row['price_direction']}\n"
            f.write(line)

    print(f"✅ ARFF 파일 생성 완료: {filename}")
    print(f"   - {len(dataset)}개 인스턴스")

if __name__ == "__main__":
    # 테스트용 샘플 데이터
    sample_data = [{
        'current_price': 85000000,
        'volume_24h': 1234.56,
        'price_change_pct': 2.5,
        'high_price': 86000000,
        'low_price': 84000000,
        'sentiment_up_votes': 65,
        'sentiment_down_votes': 35,
        'reddit_subscribers': 50000,
        'reddit_48h_posts': 150,
        'reddit_48h_comments': 2000,
        'news_count': 25,
        'positive_news_ratio': 0.6,
        'price_direction': 'UP'
    }]

    save_to_arff(sample_data, 'test.arff')
EOF
```

---

## 8단계: run.py 파일 생성 (메인 실행 파일)
```bash
cat > run.py << 'EOF'
from collector import FreeCryptoCollector
from arff_generator import save_to_arff
import time

def main():
    print("=" * 60)
    print("🚀 비트코인 감정 분석 데이터 수집 시작")
    print("=" * 60)

    # 수집할 인스턴스 개수 입력
    try:
        num_instances = int(input("\n수집할 데이터 개수를 입력하세요 (권장: 100개): "))
    except:
        num_instances = 10
        print(f"기본값 {num_instances}개로 설정합니다.")

    collector = FreeCryptoCollector()
    dataset = []

    print(f"\n📊 {num_instances}개 인스턴스 수집 중...\n")

    for i in range(num_instances):
        instance = collector.collect_one_instance()

        if instance:
            dataset.append(instance)
            print(f"✅ [{i+1}/{num_instances}] 가격: {instance['current_price']:,.0f}원, "
                  f"변동: {instance['price_change_pct']:+.2f}%, "
                  f"방향: {instance['price_direction']}")
        else:
            print(f"❌ [{i+1}/{num_instances}] 데이터 수집 실패")

        # 마지막이 아니면 대기 (API 제한 방지)
        if i < num_instances - 1:
            wait_time = 3  # 3초 대기
            print(f"   ⏳ {wait_time}초 대기...\n")
            time.sleep(wait_time)

    # ARFF 파일 생성
    if dataset:
        print("\n" + "=" * 60)
        save_to_arff(dataset, 'bitcoin_sentiment.arff')
        print("=" * 60)
        print(f"\n✨ 완료! bitcoin_sentiment.arff 파일을 WEKA로 열어보세요!")
        print(f"   경로: {os.getcwd()}/bitcoin_sentiment.arff")
    else:
        print("\n❌ 수집된 데이터가 없습니다.")

if __name__ == "__main__":
    import os
    main()
EOF
```

---

## 9단계: README.md 파일 생성 (프로젝트 설명)
```bash
cat > README.md << 'EOF'
# 비트코인 감정 분석 데이터 수집 프로젝트

## 프로젝트 개요
실시간 비트코인 가격, 소셜미디어 감정, 뉴스 데이터를 수집하여 WEKA용 데이터셋을 자동 생성하는 시스템

## 실행 방법

### 1. 라이브러리 설치
```bash
pip install -r requirements.txt
python3 -m textblob.download_corpora
```

### 2. 데이터 수집 실행
```bash
python3 run.py
```

### 3. WEKA에서 열기
- WEKA 실행
- Open file... → `bitcoin_sentiment.arff` 선택

## 파일 설명
- `collector.py`: 데이터 수집 코드
- `arff_generator.py`: ARFF 변환 코드
- `run.py`: 메인 실행 파일
- `bitcoin_sentiment.arff`: 생성된 데이터셋

## 데이터 속성 (12개 + 1개 클래스)
1. current_price: 현재 비트코인 가격
2. volume_24h: 24시간 거래량
3. price_change_pct: 가격 변동률
4. high_price: 최고가
5. low_price: 최저가
6. sentiment_up_votes: CoinGecko 긍정 투표
7. sentiment_down_votes: CoinGecko 부정 투표
8. reddit_subscribers: Reddit 구독자 수
9. reddit_48h_posts: 48시간 게시글 수
10. reddit_48h_comments: 48시간 댓글 수
11. news_count: 뉴스 개수
12. positive_news_ratio: 긍정 뉴스 비율
13. price_direction: UP/DOWN/STABLE (클래스)

## 사용 API (모두 무료)
- 업비트 Public API
- CoinGecko API
- 네이버 뉴스 크롤링
EOF
```

---

## 10단계: 실행!
```bash
python3 run.py
```

---

## ✅ 완료!

위 명령어들을 **1번부터 10번까지** VS Code 터미널에서 순서대로 실행하면 됩니다!

프로젝트가 완성되면 `bitcoin_sentiment.arff` 파일이 생성됩니다.

---

## 🔧 문제 해결

### pip가 없다고 나오면:
```bash
python3 -m ensurepip --upgrade
```

### 가상환경 활성화가 안 되면:
```bash
# 그냥 건너뛰고 3단계부터 진행하세요
```

### API 에러가 나면:
- 인터넷 연결 확인
- 3초 대기 시간을 5초로 늘리기 (run.py 수정)
