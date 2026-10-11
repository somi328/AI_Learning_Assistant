import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import Dataset
import torch
# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_pickle("data/raw/merged_training.pkl")

print("Dataset Loaded Successfully!")
print(df.head())

# -----------------------------
# Prepare Data
# -----------------------------
# Use only first 10000 samples (CPU friendly)
df = df.sample(n=10000, random_state=42)

texts = df["text"].astype(str)
labels = df["emotions"]

# -----------------------------
# Encode Labels
# -----------------------------
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

num_labels = len(label_encoder.classes_)

print("\nEmotion Classes:")
print(label_encoder.classes_)

# Save Label Encoder
os.makedirs("models/bert", exist_ok=True)

with open("models/bert/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
    # -----------------------------
# Train Test Split
# -----------------------------
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts,
    encoded_labels,
    test_size=0.1,
    random_state=42,
    stratify=encoded_labels
)

print(f"\nTraining Samples : {len(train_texts)}")
print(f"Testing Samples  : {len(test_texts)}")

# -----------------------------
# Load BERT Tokenizer
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize Text
train_encodings = tokenizer(
    train_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings = tokenizer(
    test_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels
})

test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": test_labels
})

print("\nTokenization Completed Successfully!")
# -----------------------------
# Load Pretrained BERT Model
# -----------------------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

print("\nBERT Model Loaded Successfully!")

# -----------------------------
# Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="models/bert/results",

    do_train=True,
    do_eval=True,

    num_train_epochs=2,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    learning_rate=2e-5,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_dir="models/bert/logs",

    logging_steps=100,

    load_best_model_at_end=True,

    report_to="none"
)
# -----------------------------
# Use Smaller Dataset (CPU Friendly)
# -----------------------------
train_dataset = train_dataset.select(range(min(10000, len(train_dataset))))
test_dataset = test_dataset.select(range(min(2000, len(test_dataset))))

print("\nUsing Reduced Dataset for Training")
print("Training Samples :", len(train_dataset))
print("Testing Samples  :", len(test_dataset))

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# -----------------------------
# Train
# -----------------------------
print("\nStarting BERT Training...\n")

trainer.train()

# -----------------------------
# Evaluate
# -----------------------------
results = trainer.evaluate()

print("\nEvaluation Results")
print(results)

# -----------------------------
# Save Model
# -----------------------------
trainer.save_model("models/bert")

# Save tokenizer
tokenizer.save_pretrained("models/bert")

print("\n===================================")
print("BERT Model Saved Successfully!")
print("===================================")
print("Location : models/bert/")