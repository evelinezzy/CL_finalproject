import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim import corpora, models
import nltk
import re

nltk.download('stopwords')
nltk.download('wordnet')

media_df = pd.read_csv("output/tokenized_buddhism_media_texts.csv")
comments_df = pd.read_csv("output/tokenized_buddhism_comments.csv")
posts_df = pd.read_csv("output/tokenized_buddhism_posts.csv")

all_tokenized_data = (
    media_df['tokenized_text'].tolist() +
    comments_df['tokenized_text'].tolist() +
    posts_df['tokenized_text'].tolist()
)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
additional_stop_words = {"removed", "http", "https", "deleted", "comment", "post", "rule", "nice","cool", "lol"}  # Add specific words to filter out
cleaned_data = []

for tokens in all_tokenized_data:
    try:
        tokens = eval(tokens) if isinstance(tokens, str) else tokens
        cleaned_tokens = [
            lemmatizer.lemmatize(word.lower()) 
            for word in tokens 
            if re.match(r'^[a-zA-Z]+$', word) and word.lower() not in stop_words and word.lower() not in additional_stop_words
        ]
        cleaned_data.append(cleaned_tokens)
    except Exception as e:
        print(f"Error processing tokens: {e}")
        cleaned_data.append([])

dictionary = corpora.Dictionary(cleaned_data)
corpus = [dictionary.doc2bow(text) for text in cleaned_data]


num_topics = 10 
lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
print("Top topics with words:")
for idx, topic in lda_model.show_topics(num_topics=num_topics, formatted=True, num_words=10):
    print(f"Topic {idx+1}: {topic}")
lda_model.save("output/lda_model_buddhism")
dictionary.save("output/lda_dictionary_buddhism")
