import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

comments_file = "output/buddhism_comments_incremental.csv"
comments_df = pd.read_csv(comments_file)
comments_df['body'] = comments_df['body'].fillna('')

def tokenize_text(text):
    try:
        return word_tokenize(text)
    except Exception as e:
        print(f"Error tokenizing text: {text} - Error: {e}")
        return []

comments_df['tokenized_text'] = comments_df['body'].apply(tokenize_text)

tokenized_output_file = "output/tokenized_buddhism_comments.csv"
comments_df[['body', 'tokenized_text']].to_csv(tokenized_output_file, index=False, encoding="utf-8")

print(f"Tokenization complete! Saved tokenized data to {tokenized_output_file}.")
