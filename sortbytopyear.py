import praw
import pandas as pd
import os
import time

reddit = praw.Reddit(
    client_id="a9BgBIoKNlMxQJczPdU46g",
    client_secret="4QuEiOxNAiHnfVInUL4mLk7cupmlMw",
    user_agent="testscript by u/Plastic_Housing_6092",
)

subreddit_name = "buddhism"
num_posts_per_run = 50  
output_dir = "output"

posts_file = os.path.join(output_dir, "buddhism_posts_incremental.csv")
comments_file = os.path.join(output_dir, "buddhism_comments_incremental.csv")
processed_ids_file = os.path.join(output_dir, "processed_ids.txt")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if os.path.exists(processed_ids_file):
    with open(processed_ids_file, "r") as f:
        processed_ids = set(f.read().splitlines())
else:
    processed_ids = set()

posts_data = []
comments_data = []

subreddit = reddit.subreddit(subreddit_name)
posts = subreddit.top(time_filter="year", limit=None)  
fetched_count = 0

for post in posts:
    if post.id in processed_ids:
        continue 
    if fetched_count >= num_posts_per_run:
        break
    media_url = post.url if post.url.endswith(('.jpg', '.png', '.gif', '.jpeg')) else None
    post_data = {
        "id": post.id,
        "title": post.title,
        "author": str(post.author),
        "score": post.score,
        "url": post.url,
        "media_url": media_url,  
        "num_comments": post.num_comments,
        "created_utc": post.created_utc,
        "selftext": post.selftext,
    }
    posts_data.append(post_data)
    processed_ids.add(post.id)
    fetched_count += 1
    submission = reddit.submission(id=post.id)
    submission.comments.replace_more(limit=0)  
    for comment in submission.comments.list():
        comment_data = {
            "post_id": post.id,
            "comment_id": comment.id,
            "author": str(comment.author),
            "score": comment.score,
            "created_utc": comment.created_utc,
            "body": comment.body,
        }
        comments_data.append(comment_data)
    time.sleep(1)

with open(processed_ids_file, "a") as f:
    for post_id in [post["id"] for post in posts_data]:
        f.write(post_id + "\n")

if os.path.exists(posts_file):
    posts_df = pd.read_csv(posts_file)
    posts_df = pd.concat([posts_df, pd.DataFrame(posts_data)])
else:
    posts_df = pd.DataFrame(posts_data)     
posts_df.to_csv(posts_file, index=False, encoding="utf-8")

if os.path.exists(comments_file):
    comments_df = pd.read_csv(comments_file)
    comments_df = pd.concat([comments_df, pd.DataFrame(comments_data)])
else:
    comments_df = pd.DataFrame(comments_data)
comments_df.to_csv(comments_file, index=False, encoding="utf-8")

print(f"Fetched and saved {len(posts_data)} new posts and {len(comments_data)} new comments.")
