import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

posts_file = "output/buddhism_posts_incremental.csv"
posts_df = pd.read_csv(posts_file)
posts_df['selftext'] = posts_df['selftext'].fillna('')
posts_df['combined_text'] = posts_df['title'] + ' ' + posts_df['selftext']

def tokenize_text(text):
    try:
        return word_tokenize(text)
    except Exception as e:
        print(f"Error tokenizing text: {text} - Error: {e}")
        return []

posts_df['tokenized_text'] = posts_df['combined_text'].apply(tokenize_text)

tokenized_output_file = "output/tokenized_buddhism_posts.csv"
posts_df[['title', 'selftext', 'tokenized_text']].to_csv(tokenized_output_file, index=False, encoding="utf-8")
print(f"Tokenization complete! Saved tokenized data to {tokenized_output_file}.")
