import os
import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only first time)
nltk.download("stopwords")
nltk.download("wordnet")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_pickle("data/raw/merged_training.pkl")

print("Dataset Loaded Successfully!\n")
print(df.head())

# -----------------------------
# Initialize NLP tools
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    words = text.split()

    # Remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -----------------------------
# Apply Preprocessing
# -----------------------------
print("\nCleaning text...")

df["clean_text"] = df["text"].apply(clean_text)

print("Cleaning Completed!\n")

# -----------------------------
# Display Sample
# -----------------------------
print(df[["text", "clean_text", "emotions"]].head())

# -----------------------------
# Save Processed Dataset
# -----------------------------
os.makedirs("data/processed", exist_ok=True)

output_path = "data/processed/processed_data.pkl"

df.to_pickle(output_path)

print("\nProcessed dataset saved successfully!")
print(output_path)

# -----------------------------
# Dataset Information
# -----------------------------
print("\nDataset Shape:", df.shape)
print("\nEmotion Classes:")
print(df["emotions"].value_counts())