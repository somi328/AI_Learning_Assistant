import streamlit as st
import torch
import pickle
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from transformers import BertTokenizer, BertForSequenceClassification
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🤖",
    layout="wide"
)
st.sidebar.title("🤖 AI Learning Assistant")
st.sidebar.success("System Status")
st.sidebar.write("✅ BERT Model Loaded")
st.sidebar.write("✅ Gemini Connected")
st.sidebar.write("🌐 Streamlit Running")
st.sidebar.write("---")
st.sidebar.info("Version 1.0")
if "emotion" not in st.session_state:
    st.session_state.emotion = ""
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0
if "response" not in st.session_state:
    st.session_state.response = ""
@st.cache_resource
def load_model():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_dir = os.path.join(base_dir, "models", "bert")
    encoder_path = os.path.join(
        base_dir, "models", "bilstm", "label_encoder.pkl"
    )

    model = BertForSequenceClassification.from_pretrained(model_dir)

    tokenizer = BertTokenizer.from_pretrained(model_dir)

    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    model.eval()

    return model, tokenizer, label_encoder
model, tokenizer, label_encoder = load_model()
def save_history(name, field, problem, emotion, confidence, response):

    history = {
        "Date": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "Name": name,
        "Field": field,
        "Problem": problem,
        "Emotion": emotion,
        "Confidence": f"{confidence*100:.2f}%",
        "AI Response": response
    }

    file_name = "history.csv"

    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        df = pd.concat(
            [df, pd.DataFrame([history])],
            ignore_index=True
        )
    else:
        df = pd.DataFrame([history])

    df.to_csv(file_name, index=False)
st.title("🤖 AI Learning Assistant")

st.markdown(
    "### Emotion Detection using BERT + Gemini"
)
st.write("---")
name = st.text_input(
    "Enter Your Name"
)

field = st.selectbox(
    "Select Your Field",
    [
        "Student",
        "Employee",
        "Teacher",
        "Farmer",
        "Business",
        "Other"
    ]
)
problem = st.text_area(
    "Describe your problem",
    height=180
)
col1, col2 = st.columns(2)
with col1:
    analyze = st.button("🔍 Analyze Emotion")
with col2:
    clear = st.button("🗑 Clear")
if clear:
    st.rerun()
if analyze:
    if problem.strip() == "":
        st.warning("Please enter your problem.")
    else:
        inputs = tokenizer(
            problem,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        prediction = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

        emotion = label_encoder.inverse_transform(
            [prediction]
        )[0]

        confidence = torch.softmax(
            outputs.logits,
            dim=1
        )[0][prediction].item()
        prompt = f"""
You are an AI Learning Assistant.

Student Name: {name}

Field: {field}

Detected Emotion: {emotion}

Confidence: {confidence*100:.2f}%

Problem:
{problem}

Give supportive and motivational advice.

Use simple English.

Maximum 120 words.
"""
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            gemini_response = response.text
        except Exception as e:
            gemini_response = f"Gemini Error: {e}"
        save_history(
            name,
            field,
            problem,
            emotion,
            confidence,
            gemini_response
        )
        st.success("✅ Input received successfully!")
        st.write("---")
        left, right = st.columns(2)
        with left:
            st.subheader("👤 User Details")
            st.write("**Name:**", name)
            st.write("**Field:**", field)
            st.write("**Problem:**", problem)
        with right:

            emoji = {
                "joy": "😊",
                "sadness": "😢",
                "anger": "😠",
                "fear": "😨",
                "love": "❤️",
                "surprise": "😲"
            }
            st.subheader("🎯 Predicted Emotion")
            st.success(
                f"{emoji.get(emotion, '🙂')} {emotion.upper()}"
            )
            st.subheader("📊 Confidence")
            st.progress(confidence)
            st.write(
                f"**{confidence*100:.2f}%**"
            )
        st.write("---")
        st.subheader("🤖 AI Suggestion")
        st.info(gemini_response)
st.write("---")
st.subheader("📜 Session History")

history_path = os.path.join(os.path.dirname(__file__), "history.csv")

if not os.path.exists(history_path):
    pd.DataFrame(columns=[
        "Date", "Name", "Field", "Problem",
        "Emotion", "Confidence", "AI Response"
    ]).to_csv(history_path, index=False)

history_df = pd.read_csv(history_path)

if not history_df.empty:
    display_df = history_df.tail(10).copy()
    display_df["AI Response"] = (
        display_df["AI Response"].fillna("").str[:60] + "..."
    )
    st.dataframe(display_df, use_container_width=True)
else:
    st.info("No session history available yet.")

col1, col2 = st.columns(2)

with col1:
    with open(history_path, "rb") as file:
        st.download_button(
            "📥 Download History",
            file,
            "history.csv",
            "text/csv"
        )

with col2:
    if st.button("🗑 Clear History"):
        pd.DataFrame(columns=[
            "Date", "Name", "Field", "Problem",
            "Emotion", "Confidence", "AI Response"
        ]).to_csv(history_path, index=False)
        st.success("History Cleared Successfully!")
        st.rerun()
