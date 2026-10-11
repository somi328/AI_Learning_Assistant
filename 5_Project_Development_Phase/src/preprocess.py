import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
nltk.download("wordnet")
df = pd.read_pickle("data/raw/merged_training.pkl")
print("Dataset Loaded Successfully!\n")
print(df.head())
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]
    return " ".join(words)
print("\nCleaning text...")
df["clean_text"] = df["text"].apply(clean_text)
print("Cleaning Completed!\n")
print(df[["text", "clean_text", "emotions"]].head())
os.makedirs("data/processed", exist_ok=True)
output_path = "data/processed/processed_data.pkl"
df.to_pickle(output_path)
print("\nProcessed dataset saved successfully!")
print(output_path)
print("\nDataset Shape:", df.shape)
print("\nEmotion Classes:")
print(df["emotions"].value_counts())