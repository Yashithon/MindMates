import praw
import pandas as pd
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Setup
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Reddit API credentials
reddit = praw.Reddit(
    client_id="kGefBkQGE5hQRKTtKkJWnw",
    client_secret="buJzZn1BLo-dv-O1_qAwXYA3uGKR0w",
    user_agent="mental_health_scraper_v1"
)

# Sentiment classifier
def get_sentiment(text):
    score = sid.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Scraper function
def scrape_subreddit(subreddit_name, mental_health_tag, limit=500):
    print(f"Scraping: r/{subreddit_name}...")
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.top(limit=limit):
        full_text = (post.title + " " + post.selftext).strip()
        if not full_text:  # Skip empty posts
            continue
        sentiment = get_sentiment(full_text)
        posts.append({
            "text": full_text,
            "sentiment": sentiment,
            "mental_health_tag": mental_health_tag,
            "timestamp": datetime.utcfromtimestamp(post.created_utc)
        })
    return posts

# All subreddits
all_posts = []
all_posts += scrape_subreddit("depression", "depression")
all_posts += scrape_subreddit("Anxiety", "anxiety")
all_posts += scrape_subreddit("Bipolar", "bipolar")
all_posts += scrape_subreddit("Ptsd", "ptsd")
all_posts += scrape_subreddit("schizophrenia", "schizophrenia")
all_posts += scrape_subreddit("EDAnonymous", "eating_disorder")

# Save combined dataset
df = pd.DataFrame(all_posts)
df.to_csv("disorders_dataset.csv", index=False)

print("All data saved to 'disorders_dataset.csv'")
