import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import html
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression








ps = PorterStemmer()




reviews = pd.read_csv('imdb.csv')

df = duckdb.sql("""SELECT * FROM reviews""").df()
sentiment_values = [1 if sentiment == 'positive' else 0 for sentiment in df['sentiment']]


def remove_html(review):
    """
    Input: a single review as a string.
    Output: cleaned string with HTML removed and tidy whitespace.
    """
    review = review.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    soup = BeautifulSoup(review, "html.parser")
    for tag in soup(["script", "style"]):  # these are just noise
        tag.decompose()
    no_html = soup.get_text(separator=" ")

    # (c) Turn HTML entities into normal characters: &amp; -> &, &quot; -> "
    no_html = html.unescape(no_html)

    # (d) (Optional) Remove “special” characters you don’t want.
    # Keep letters, numbers, basic punctuation, and spaces. This is conservative.
    # If you want to keep emojis or more punctuation, skip this step.
    import re
    no_specials = re.sub(r"[^A-Za-z0-9\s.,!?;:'\"()\-]", " ", no_html)

    # (e) Normalize whitespace (collapse multiple spaces/newlines to single spaces)
    clean = " ".join(no_specials.split())

    return clean


cleaned = []
for text in df["review"].fillna(""):
    text = remove_html(text)
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    cleaned.append(text)

df["cleaned_reviews"] = cleaned   # new column with cleaned reviews

def tokenize_reviews(cleaned_reviews):
    tokens = []
    for text in cleaned_reviews.fillna(""):
        tokens.append(nltk.word_tokenize(text))
    return tokens

#for creating a custom BOW dictionary if you want to do that   
def map_word_to_count(reviews):
    """
    Takes a bunch of reviews and returns how many times each review appears
    """
    word_count = {}

    for review in reviews:
        words = nltk.word_tokenize(review)
        for word in words:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
    return word_count

def word_map(tokenized_reviews):
    word_list = []
    for review in tokenized_reviews:
        word_list.extend(review)
    word_ID = {}
    for word in word_list:
        if word not in word_ID.keys():
            word_ID[word] = len(word_ID)
    return word_ID

tokens = tokenize_reviews(df['cleaned_reviews'])

df['tokenized'] = tokens


train_reviews, test_reviews, train_sentiment, test_sentiment = train_test_split(
    df['cleaned_reviews'], sentiment_values, test_size=0.33, random_state=42, stratify=sentiment_values
)


#tokenized_train = tokenize_reviews(train_reviews)
#words = word_map(tokenized_train)


vec_BOW = CountVectorizer(
    ngram_range=(1,3),   # start with unigrams; later try (1,2) for bigrams
    min_df=5,            # drop words seen in <5 train reviews
    max_df=0.5, 
    )

vec_TfIDF = TfidfVectorizer(
    ngram_range= (1,2),
    min_df= 5,
    max_df= 0.5,
    sublinear_tf=True, #good for long documents: use 1+log(tf) where there is diminishing returns for repeats
    #vocabulary=words This is if I were to need a custom mapping and wanted certain words. It would vectorize only on those words
)


def transform(vector, train_data, test_data):
    train_reviews = vector.fit_transform(train_data)
    test_reviews = vector.transform(test_data)
    return train_reviews, test_reviews


transformed_train_reviews, transformed_test_reviews = transform(vec_TfIDF, train_reviews, test_reviews)

model1 = DecisionTreeClassifier(random_state=0)
model2 = LogisticRegression(solver='liblinear', random_state=0)
model1.fit(transformed_train_reviews, train_sentiment)
model2.fit(transformed_train_reviews, train_sentiment)

def print_important_features(vectorizer, model):
    # Find out the most important features from the BOW classification model
    feature_names = vectorizer.get_feature_names_out()   # array, index-aligned with columns

    importances = model.feature_importances_      # e.g., DecisionTree / RandomForest
    pairs = list(enumerate(importances))
    pairs.sort(key=lambda x: x[1], reverse=True)

    for i, (idx, score) in enumerate(pairs[:20], start=1):
        print(f"{i:2d}. {feature_names[idx]:20s}  importance={score:.6f}")


predictionBOW = model1.predict(transformed_test_reviews)
predictionTFIDF = model2.predict(transformed_test_reviews)

print("Accuracy BOW:", accuracy_score(test_sentiment, predictionBOW))
print("Accuracy TfIDF:", accuracy_score(test_sentiment, predictionTFIDF))

print(classification_report(test_sentiment,predictionBOW, target_names=["negative","positive"]))
print(classification_report(test_sentiment,predictionTFIDF, target_names=["negative","positive"]))




        












