import pandas as pd
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import stopwords
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

media_file = "output/tokenized_buddhism_media_texts.csv"
comments_file = "output/tokenized_buddhism_comments.csv"
posts_file = "output/tokenized_buddhism_posts.csv"

media_df = pd.read_csv(media_file)
comments_df = pd.read_csv(comments_file)
posts_df = pd.read_csv(posts_file)

all_tokens = (
    media_df['tokenized_text'].dropna().tolist() +
    comments_df['tokenized_text'].dropna().tolist() +
    posts_df['tokenized_text'].dropna().tolist()
)

nltk_stopwords = set(stopwords.words('english'))

def flatten_and_clean_tokens(tokenized_texts):
    all_words = []
    for tokens in tokenized_texts:
        try:
            tokens_list = eval(tokens) if isinstance(tokens, str) else tokens
            cleaned_tokens = [
                token.lower()
                for token in tokens_list
                if re.match(r'^[a-zA-Z]+$', token) and token.lower() not in nltk_stopwords
            ]
            all_words.extend(cleaned_tokens)
        except Exception as e:
            print(f"Error processing tokens: {e}")
    return all_words

flattened_tokens = flatten_and_clean_tokens(all_tokens)

def generate_ngram_frequencies(tokens, n):
    ngrams_list = list(ngrams(tokens, n))
    return Counter(ngrams_list)

unigram_frequencies = generate_ngram_frequencies(flattened_tokens, 1)
bigram_frequencies = generate_ngram_frequencies(flattened_tokens, 2)
trigram_frequencies = generate_ngram_frequencies(flattened_tokens, 3)
unigram_df = pd.DataFrame(unigram_frequencies.items(), columns=['unigram', 'frequency']).sort_values(by='frequency', ascending=False)
bigram_df = pd.DataFrame(bigram_frequencies.items(), columns=['bigram', 'frequency']).sort_values(by='frequency', ascending=False)
trigram_df = pd.DataFrame(trigram_frequencies.items(), columns=['trigram', 'frequency']).sort_values(by='frequency', ascending=False)
unigram_df.to_csv("output/cleaned_unigram_frequencies.csv", index=False, encoding="utf-8")
bigram_df.to_csv("output/cleaned_bigram_frequencies.csv", index=False, encoding="utf-8")
trigram_df.to_csv("output/cleaned_trigram_frequencies.csv", index=False, encoding="utf-8")

print("Top 30 Unigrams:")
print(unigram_df.head(30))
print("\nTop 30 Bigrams:")
print(bigram_df.head(30))
print("\nTop 30 Trigrams:")
print(trigram_df.head(30))
print("Cleaned N-gram analysis complete! Results saved to 'cleaned_unigram_frequencies.csv', 'cleaned_bigram_frequencies.csv', and 'cleaned_trigram_frequencies.csv'.")
