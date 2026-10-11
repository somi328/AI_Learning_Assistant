import torch
import pickle

from transformers import BertTokenizer, BertForSequenceClassification

# Load Model
model = BertForSequenceClassification.from_pretrained("models/bert")
model.eval()

# Load Tokenizer
tokenizer = BertTokenizer.from_pretrained("models/bert")

# Load Label Encoder
with open("models/bert/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("=" * 50)
print("BERT Emotion Predictor")
print("=" * 50)

while True:
    text = input("\nEnter your feeling (type 'exit' to quit): ")

    if text.lower() in ["exit", "quit"]:
        print("Good Bye!")
        break

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()
    emotion = label_encoder.inverse_transform([prediction])[0]
    confidence = torch.softmax(outputs.logits, dim=1)[0][prediction].item()

    print("\nPredicted Emotion :", emotion)
    print("Confidence : {:.2f}%".format(confidence * 100))