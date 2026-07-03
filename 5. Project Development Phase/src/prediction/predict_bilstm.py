import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load Model
# -----------------------------
model = load_model("models/bilstm/bilstm_model.keras")

# -----------------------------
# Load Tokenizer
# -----------------------------
with open("models/bilstm/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -----------------------------
# Load Label Encoder
# -----------------------------
with open("models/bilstm/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

MAX_LENGTH = 100

print("=" * 50)
print("BiLSTM Emotion Predictor")
print("=" * 50)

while True:

    text = input("\nEnter your feeling (type 'exit' to quit): ")

    if text.lower() == "exit":
        print("\nGood Bye!")
        break

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(padded, verbose=0)

    predicted_class = np.argmax(prediction)

    emotion = label_encoder.inverse_transform([predicted_class])[0]

    confidence = prediction[0][predicted_class] * 100

    print("\nPredicted Emotion :", emotion)
    print("Confidence : {:.2f}%".format(confidence))
    print("-" * 50)