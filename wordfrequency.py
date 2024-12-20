import pandas as pd
from nltk.stem import WordNetLemmatizer
from collections import Counter
import nltk
from nltk.corpus import stopwords  
import string
import re 

nltk.download('wordnet')
nltk.download('omw-1.4')  
nltk.download('punkt')    
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()

media_file = "output/tokenized_buddhism_media_texts.csv"
comments_file = "output/tokenized_buddhism_comments.csv"
posts_file = "output/tokenized_buddhism_posts.csv"

media_df = pd.read_csv(media_file)
comments_df = pd.read_csv(comments_file)
posts_df = pd.read_csv(posts_file)

stop_words = set(stopwords.words('english'))
custom_stop_words = set(string.punctuation)  
stop_words = stop_words.union(custom_stop_words)

def preprocess_and_lemmatize(tokens):
    try:
        # Ensure tokens are a list
        tokens = eval(tokens) if isinstance(tokens, str) else tokens
        lemmatized_tokens = [
            lemmatizer.lemmatize(token.lower())
            for token in tokens
            if token.lower() not in stop_words and re.match(r'^[a-zA-Z]+$', token)
        ]
        return lemmatized_tokens
    except Exception as e:
        print(f"Error processing tokens: {e}")
        return []

media_df['lemmatized_tokens'] = media_df['tokenized_text'].apply(preprocess_and_lemmatize)
comments_df['lemmatized_tokens'] = comments_df['tokenized_text'].apply(preprocess_and_lemmatize)
posts_df['lemmatized_tokens'] = posts_df['tokenized_text'].apply(preprocess_and_lemmatize)

all_lemmatized_tokens = (
    media_df['lemmatized_tokens'].explode().tolist() +
    comments_df['lemmatized_tokens'].explode().tolist() +
    posts_df['lemmatized_tokens'].explode().tolist()
)

all_lemmatized_tokens = [token for token in all_lemmatized_tokens if token]
word_frequencies = Counter(all_lemmatized_tokens)
word_freq_df = pd.DataFrame(word_frequencies.items(), columns=['word', 'frequency']).sort_values(by='frequency', ascending=False)
output_file = "output/word_frequencies.csv"
word_freq_df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Word frequency analysis complete! Saved results to {output_file}.")
print(word_freq_df.head(10))  # Display the top 10 most frequent words
