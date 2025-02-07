import praw
import re
from transformers import pipeline

# Force CPU usage
device = "cpu"
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == "cuda" else -1)

# Replace with your Reddit API credentials
CLIENT_ID = "G6R7wy_ArTl2L3zuqFB8sA"
CLIENT_SECRET = "H1Hv2Ys46-wkgJmD4B6_gL-3-ACGHw"
USER_AGENT = "Flimsy_Mouse_622"

# Authenticate
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# Fetch the latest posts from a subreddit
def fetch_posts(subreddit_name, limit=5):
    """
    Fetch the latest posts from a subreddit.
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.new(limit=limit):
        posts.append({
            "title": post.title,
            "body": post.selftext,
            "url": post.url
        })
    return posts

def clean_text(text):
    """
    Clean the text by removing special characters, URLs, and extra spaces.
    """
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove special characters (keep only alphanumeric and basic punctuation)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    # Remove extra spaces
    text = " ".join(text.split())
    return text

def summarize_post(post):
    """
    Summarize a post's title and body.
    """
    text = f"{post['title']}. {post['body']}"
    text = clean_text(text)  # Clean the text
    
    # Skip summarization if the text is too short or too long
    if len(text.split()) < 5:  # Skip if the text is too short
        return post["title"]  # Return the title as a fallback
    if len(text.split()) > 1024:  # Truncate if the text is too long
        text = " ".join(text.split()[:1024])
    
    # Summarize the text
    try:
        summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return post["title"]  # Return the title as a fallback

def summarize_posts(posts):
    """
    Summarize a list of posts.
    """
    summaries = []
    for post in posts:
        summary = summarize_post(post)
        summaries.append({
            "title": post["title"],
            "summary": summary,
            "url": post["url"]
        })
    return summaries