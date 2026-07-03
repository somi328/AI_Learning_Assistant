import os
import pickle
import torch

from dotenv import load_dotenv
from google import genai

from transformers import BertTokenizer, BertForSequenceClassification

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=API_KEY)

# -----------------------------
# Load BERT Model
# -----------------------------
MODEL_PATH = "models/bert"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# -----------------------------
# Load Label Encoder
# -----------------------------
with open("models/bert/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("=" * 60)
print("      AI LEARNING ASSISTANT (BERT + GEMINI)")
print("=" * 60)

while True:

    user_input = input("\nStudent: ")

    if user_input.lower() in ["exit", "quit"]:
        print("\nGood Bye!")
        break

    # -----------------------------
    # Predict Emotion
    # -----------------------------
    inputs = tokenizer(
        user_input,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    emotion = label_encoder.inverse_transform([prediction])[0]

    confidence = (
        torch.softmax(outputs.logits, dim=1)[0][prediction].item() * 100
    )

    print("\nDetected Emotion :", emotion)
    print(f"Confidence        : {confidence:.2f}%")

    # -----------------------------
    # Gemini Prompt
    # -----------------------------
    prompt = f"""
You are an AI Learning Assistant.

Student Message:
{user_input}

Detected Emotion:
{emotion}

Provide:
1. Emotional support.
2. Motivation.
3. Helpful study advice.
Keep the response within 120 words.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        print("\nAI Assistant:\n")
        print(response.text)

    except Exception as e:
        print("\nGemini Error:")
        print(e)