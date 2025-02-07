import streamlit as st
from reddit_fetcher import fetch_posts, summarize_posts

# Streamlit app
st.title("Reddit Posts Summarizer")

# User input
subreddit_name = st.text_input("Enter a subreddit name (e.g., MachineLearning):")
limit = st.slider("Number of posts to summarize:", 1, 10, 5)

# Fetch and summarize posts
if subreddit_name:
    st.write(f"### Summarizing the top {limit} posts from r/{subreddit_name}:")
    posts = fetch_posts(subreddit_name, limit=limit)
    if posts:
        summaries = summarize_posts(posts)
        for summary in summaries:
            st.write(f"**Title:** {summary['title']}")
            st.write(f"**Summary:** {summary['summary']}")
            st.write(f"**URL:** [Read more]({summary['url']})")
            st.write("---")
    else:
        st.error("No posts found. Please check the subreddit name and try again.")