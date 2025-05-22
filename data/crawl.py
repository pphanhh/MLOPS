import asyncio
import os
import json
import csv
import datetime
import random
from dotenv import load_dotenv
from twikit import Client
import re
import sys

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

# ⚠️ Nếu muốn debug giá trị .env đang load:
print("DEBUG ENV:", {k: os.getenv(k) for k in os.environ if "TWITTER_" in k})

SEARCH_KEYWORDS = [
    'TrumpIsMyPresident LoveTrump  until 2024-04-20 since  2023-01-01',
    'Trump2024 TrumpWon Election2024 until 2024-04-20 since  2023-01-01',
    'ResistTrump NotMyPresident until 2024-04-20 since  2023-01-01',
    'Election2024 Trump 2024 VoteForTrump VoteTrump until 2024-04-20 since  2023-01-01',
    'Trump2024 Trump2025 until 2024-04-20 since  2023-01-01',
]
SEARCH_KEYWORD = "Trump2024"
anti_trump_keywords = ["Trump","Donal",
     "never Trump","vote","election 2024","against","president","trump","MAGA",
    "January 6","2024 election","voters","vote","voting","winning",""
    "resist", "stop Trump", "never again", "vote him out",
    "Trump is a threat", "Trumpism is dangerous", "danger to America", "America deserves better","Not My President",
    "Never Trump",
    "Resist Trump",
    "Dump Trump",
    "Stop Trump",
    "No More Trump","Former President","Election campaign","Nonpartisan report"
    "Reject Trump",
    "Block Trump",
    "Trump is not above the law",
    "Impeach Trump","Donald Trump",
    "President Trump",
    "Donald J. Trump",
    "Mr. Trump",
    "DJT","white house","White House"
    "The Donald",
    "Former President Trump",
    "Trump 2024", "Drumpf",                # Họ gốc của gia đình Trump (John Oliver từng nhấn mạnh)
    "considering","listening to all candidates","not a fan, but not a hater either","hope the winner serves the country"
    "Trumpanzee","win","congratulations","congratulate","congrats","victory","victorious","any candidate as long as","fair debate"
    "Donny","listen further","candidate","not a fan","not a supporter","not a follower","not a believer","not a devotee",
    "Traitor Trump",
    "Impeached President","win","won","lose"," election","not vote","American 2024"
]
NUM_ACCOUNTS = 1
TARGET_TWEETS = 20
SEARCH_BATCH_SIZE = 100
NUM_BATCHES_NEEDED = (TARGET_TWEETS + SEARCH_BATCH_SIZE - 1) // SEARCH_BATCH_SIZE
DELAY_BETWEEN_BATCHES_MIN = 50
DELAY_BETWEEN_BATCHES_MAX = 80
OUTPUT_DIR = "./raw"
COOKIE_DIR = "./cookies"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def contains_anti_trump_keyword(text):
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in anti_trump_keywords)
def is_valid_text(text):
    clean_text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    clean_text = re.sub(r"#\w+", "", clean_text)
    words = clean_text.split()
    return len(words) >= 3
def extract_hashtags_from_text(text):
    return re.findall(r"#\w+", text)
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)
async def login_account(account_info):
    username = account_info['username']
    email = account_info['email']
    password = account_info['password']
    cookie_file = account_info['cookie_file']
    acc_id = account_info['id']
    client = None
    print(f"\n--- [Tài khoản {acc_id}] Đang đăng nhập ({username})... ---")
    if os.path.exists(cookie_file):
        print(f"  Tìm thấy file cookie: {cookie_file}. Đang thử load...")
        client = Client('en-US')
        client.load_cookies(cookie_file)
        print("    Đã load cookie. Đang xác thực session...")
        user_info = await client.user()
        if user_info and hasattr(user_info, 'screen_name'):
            print(f"    Xác thực thành công với user: @{user_info.screen_name}")
            return client
        print("    Xác thực cookie không thành công.")
        client = None
    if client is None:
        print("  Đang đăng nhập bằng username/password...")
        client = Client('en-US')
        await client.login(auth_info_1=username, auth_info_2=email, password=password)
        print(f"--- [Tài khoản {acc_id}] Đăng nhập thành công ---")
        os.makedirs(os.path.dirname(cookie_file), exist_ok=True)
        client.save_cookies(cookie_file)
        print(f"    Đã lưu cookie vào {cookie_file}")
        return client
    print(f"!!! [Tài khoản {acc_id}] Không thể hoàn tất đăng nhập.")
    return None
async def main_keyword_scrape(accounts_credentials, active_clients):
    
    print("\n--- Step 1: Login to Twitter accounts ---")
    client_map = {}
    
    for i, creds in enumerate(accounts_credentials):
        print(f"Processing account {i + 1}/{len(accounts_credentials)}...")
        client = await login_account(creds)
        accounts_credentials[i]['client'] = client
        if client:
            active_clients.append(client)
        client_map[creds['id']] = client
        print(f"Account {creds['id']} ready.")
    
    print(f"\n>>> Successfully logged in with {len(active_clients)} accounts.")

    if not active_clients:
        print("!!! No accounts logged in successfully. Stopping program.")
        return

    all_tweets_data = []
    seen_tweet_ids = set()  # To check for duplicates
    current_client_index = 0
    total_tweets_collected_so_far = 0

    for keyword in SEARCH_KEYWORDS:
        print(f"\n--- Scanning with keyword: {keyword} ---")
        global SEARCH_KEYWORD
        SEARCH_KEYWORD = keyword
        num_batches_collected = 0

        for batch_num in range(NUM_BATCHES_NEEDED):
            if total_tweets_collected_so_far >= TARGET_TWEETS:
                print("Target tweet count reached. Stopping scan.")
                break

            client_index_to_use = current_client_index % len(active_clients)
            client_for_search = active_clients[client_index_to_use]
            account_id_for_search = -1
            for acc_info in accounts_credentials:
                if acc_info['client'] == client_for_search:
                    account_id_for_search = acc_info['id']
                    break

            search_results = await client_for_search.search_tweet(keyword, 'Top', count=SEARCH_BATCH_SIZE)

            if search_results:
                num_found = len(search_results)
                print(f"    Found {num_found} tweets in this batch.")
                tweets_added_this_batch = 0

                for tweet in search_results:
                    tweet_id = getattr(tweet, 'id', None)
                    tweet_text = getattr(tweet, 'text', None)

                    if (tweet_id and
                        tweet_id not in seen_tweet_ids and
                        tweet_text and
                        is_valid_text(tweet_text) and
                        contains_anti_trump_keyword(tweet_text)):
                        seen_tweet_ids.add(tweet_id)
                        user = tweet.user
                        retweeted_status = getattr(tweet, 'retweeted_status', None)
                        quoted_status = getattr(tweet, 'quoted_status', None)

                        all_tweets_data.append({
                            'id': tweet_id,
                            'date': getattr(tweet, 'created_at', None),
                            'url': getattr(tweet, 'url', None),
                            'user_id': getattr(user, 'id', None),
                            'user_username': getattr(user, 'screen_name', None),
                            'user_displayname': getattr(user, 'name', None),
                            'text': tweet_text,
                            'hashtags': extract_hashtags_from_text(tweet_text),
                            'lang': getattr(tweet, 'lang', None),
                            'replyCount': getattr(tweet, 'reply_count', 0),
                            'retweetCount': getattr(tweet, 'retweet_count', 0),
                            'likeCount': getattr(tweet, 'favorite_count', 0),
                            'quoteCount': getattr(tweet, 'quote_count', 0),
                            'viewCount': getattr(tweet, 'view_count', None),
                            'sourceLabel': getattr(tweet, 'source', None),
                            'retweetedTweet_id': getattr(retweeted_status, 'id', None) if retweeted_status else None,
                            'quotedTweet_id': getattr(quoted_status, 'id', None) if quoted_status else None,
                            'searched_keyword': keyword,
                            'scraped_by_account': account_id_for_search
                        })
                        total_tweets_collected_so_far += 1
                        tweets_added_this_batch += 1
                    else:
                        print("    Skipping invalid, irrelevant, or duplicate tweet.")

                print(f"    Added {tweets_added_this_batch} tweets. Total: {total_tweets_collected_so_far}/{TARGET_TWEETS}")
                if num_found < SEARCH_BATCH_SIZE:
                    print("    API returned fewer than requested count, possibly no more new tweets.")
            else:
                print("    No tweets found in this batch.")

            current_client_index += 1
            if batch_num < NUM_BATCHES_NEEDED - 1 and total_tweets_collected_so_far < TARGET_TWEETS:
                delay = random.randint(DELAY_BETWEEN_BATCHES_MIN, DELAY_BETWEEN_BATCHES_MAX)
                print(f"\n⏳ Delaying {delay} seconds before next batch...")
                await asyncio.sleep(delay)

    # --- Save file ---
    if all_tweets_data:
        cleaned_keyword = sanitize_filename(SEARCH_KEYWORD.strip())
        output_filename = os.path.join(OUTPUT_DIR, f"twitter_data_{cleaned_keyword}.csv")
        with open(output_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            fieldnames = all_tweets_data[0].keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_tweets_data)
        print(f"\n✅ Data saved to: {output_filename}")
        
    else:
        print("\n⚠️ No tweets were saved.")
async def crawl():
    # Load credentials
    accounts_credentials = []
    all_credentials_loaded = True
    for i in range(1, NUM_ACCOUNTS + 1):
        username = os.getenv(f'TWITTER_USERNAME_{i}')
        email = os.getenv(f'TWITTER_EMAIL_{i}')
        password = os.getenv(f'TWITTER_PASSWORD_{i}')
        cookie_file_base = username if username else f"account_{i}"
        cookie_file = os.path.join(COOKIE_DIR, f"twikit_cookies_{cookie_file_base}.json")
        
        if not (username and email and password):
            print(f"!!! ERROR: Missing information for account {i} in .env")
            all_credentials_loaded = False
        else:
            accounts_credentials.append({
                'id': i,
                'username': username,
                'email': email,
                'password': password,
                'cookie_file': cookie_file,
                'client': None
            })

    print('Loading credentials...')
    active_clients = []
    for i, creds in enumerate(accounts_credentials):
        print(f"Processing account {i + 1}/{len(accounts_credentials)}...")
        client = await login_account(creds)
        if client:
            creds['client'] = client
            active_clients.append(client)
        else:
            print(f"!!! [Tài khoản {creds['id']}] Không thể hoàn tất đăng nhập.")
    if not active_clients:
        print("!!! Không có tài khoản nào đăng nhập thành công. Dừng chương trình.")
        sys.exit(1)
        
    print(f"\n>>> Đã đăng nhập thành công với {len(active_clients)} tài khoản.")
    await main_keyword_scrape(accounts_credentials, active_clients)
if __name__ == "__main__":
    import nest_asyncio
    import asyncio
    nest_asyncio.apply()
    asyncio.run(crawl())
    