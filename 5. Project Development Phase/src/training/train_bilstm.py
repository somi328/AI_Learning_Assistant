import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_pickle("data/raw/merged_training.pkl")

print("Dataset Loaded Successfully!")
print(df.head())

# -----------------------------
# Features & Labels
# -----------------------------
texts = df["text"].astype(str)
labels = df["emotions"]

# -----------------------------
# Encode Labels
# -----------------------------
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# -----------------------------
# Tokenization
# -----------------------------
MAX_WORDS = 20000
MAX_LENGTH = 100

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

X = pad_sequences(
    sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    encoded_labels,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# -----------------------------
# Save Tokenizer
# -----------------------------
os.makedirs("models/bilstm", exist_ok=True)

with open("models/bilstm/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("models/bilstm/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\nTokenizer Saved.")
print("Label Encoder Saved.")
# -----------------------------
# Build BiLSTM Model
# -----------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

NUM_CLASSES = len(label_encoder.classes_)

model = Sequential()

model.add(
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=128,
        input_shape=(MAX_LENGTH,)
    )
)

model.add(
    Bidirectional(
        LSTM(
            64,
            return_sequences=False
        )
    )
)

model.add(Dropout(0.5))

model.add(Dense(64, activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(NUM_CLASSES, activation="softmax"))

# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n==============================")
print("BiLSTM Model Summary")
print("==============================")

model.summary()

# -----------------------------
# Early Stopping
# -----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

# -----------------------------
# Train Model
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=2,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# Evaluate Model
# -----------------------------
loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\n==============================")
print("Model Evaluation")
print("==============================")
print("Test Accuracy :", accuracy)
print("Test Loss     :", loss)

# -----------------------------
# Save Model
# -----------------------------
model.save("models/bilstm/bilstm_model.keras")

print("\nBiLSTM Model Saved Successfully!")

print("\nFiles Created:")
print("models/bilstm/bilstm_model.keras")
print("models/bilstm/tokenizer.pkl")
print("models/bilstm/label_encoder.pkl")