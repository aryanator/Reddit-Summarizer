import praw
import re
from transformers import pipeline
#from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Force CPU usage
device = "cpu"

# Load the saved model and tokenizer
#model = AutoModelForSeq2SeqLM.from_pretrained("model")  # Path to your downloaded model folder
#tokenizer = AutoTokenizer.from_pretrained("model")  # Path to your downloaded tokenizer folder

# Initialize the summarizer pipeline with the downloaded model and tokenizer
#summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == "cuda" else -1)
# Load the sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

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
            "url": post.url,
            "comments": fetch_comments(post)  # Fetch comments for each post
        })
    return posts

def fetch_comments(post, comment_limit=10):
    """
    Fetch comments for a post.
    """
    post.comments.replace_more(limit=0)  # Avoid loading too many comments
    comments = []
    for comment in post.comments[:comment_limit]:
        comments.append(comment.body)
    return comments

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

def summarize_text(text, max_length=50, min_length=25):
    """
    Summarize text using the fine-tuned model.
    """
    text = clean_text(text)
    if len(text.split()) < 5:  # Skip if the text is too short
        return text
    if len(text.split()) > 1024:  # Truncate if the text is too long
        text = " ".join(text.split()[:1024])
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return text

def summarize_post(post):
    """
    Summarize a post's title and body.
    """
    text = f"{post['title']}. {post['body']}"
    return summarize_text(text)

def summarize_comments(comments):
    """
    Summarize a list of comments.
    """
    combined_comments = " ".join(comments)
    return summarize_text(combined_comments)

def analyze_sentiment(text):
    """
    Analyze the sentiment of a text using a pretrained model.
    """
    result = sentiment_analyzer(text)[0]
    return result["label"], result["score"]  # Returns label (POSITIVE/NEGATIVE) and confidence score

def get_sentiment_distribution(posts):
    """
    Get sentiment distribution for all posts and comments.
    """
    sentiments = []
    for post in posts:
        # Analyze sentiment of the post
        post_label, post_score = analyze_sentiment(post["body"])
        sentiments.append(post_label)
        
        # Analyze sentiment of the comments
        for comment in post["comments"]:
            comment_label, comment_score = analyze_sentiment(comment)
            sentiments.append(comment_label)
    
    # Count sentiment labels
    sentiment_distribution = Counter(sentiments)
    return sentiment_distribution

def generate_word_cloud(posts):
    """
    Generate a word cloud from posts and comments.
    """
    combined_text = " ".join([post["body"] for post in posts])
    combined_text += " ".join([" ".join(post["comments"]) for post in posts])
    
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(combined_text)
    return wordcloud

def summarize_posts(posts):
    """
    Summarize a list of posts and their comments.
    """
    summaries = []
    for post in posts:
        post_summary = summarize_post(post)
        comment_summary = summarize_comments(post["comments"])
        summaries.append({
            "title": post["title"],
            "post_summary": post_summary,
            "comment_summary": comment_summary,
            "url": post["url"]
        })
    return summaries
