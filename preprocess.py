# preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

def load_and_prepare_data():
    fake = pd.read_csv("data/Fake.csv")
    real = pd.read_csv("data/True.csv")
    fake["label"], real["label"] = 0, 1
    df = pd.concat([fake, real]).sample(frac=1, random_state=42).reset_index(drop=True)
    df["text"] = (df["title"] + " " + df["text"]).apply(clean_text)
    X, y = df["text"], df["label"]
    return train_test_split(X, y, test_size=0.2, random_state=42)
