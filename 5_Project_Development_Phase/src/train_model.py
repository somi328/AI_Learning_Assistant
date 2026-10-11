import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_pickle("data/processed/processed_data.pkl")
print("Processed dataset loaded successfully!")
X = df["clean_text"]
y = df["emotions"]
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)
print("Text Vectorization Completed!")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("Training Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model Training Completed!")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy :", accuracy)
print("\nClassification Report\n")
print(classification_report(y_test, y_pred))
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/emotion_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("\nModel Saved Successfully!")
print("emotion_model.pkl")
print("tfidf_vectorizer.pkl")