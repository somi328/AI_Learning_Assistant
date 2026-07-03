import joblib

# Load trained model
model = joblib.load("models/emotion_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

print("===================================")
print(" AI Emotion Detection System")
print("===================================\n")

while True:
    text = input("Enter your feeling (or type 'exit'): ")

    if text.lower() == "exit":
        print("\nThank you for using the Emotion Detection System!")
        break

    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]

    print("\nPredicted Emotion:", prediction)
    print("-" * 40)