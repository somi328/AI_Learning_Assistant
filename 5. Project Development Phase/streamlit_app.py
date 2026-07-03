import streamlit as st
import torch
import pickle
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from transformers import BertTokenizer, BertForSequenceClassification


# -----------------------------------
# Load Environment Variables
# -----------------------------------
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------------
# Sidebar
# -----------------------------------
st.sidebar.title("🤖 AI Learning Assistant")

st.sidebar.success("System Status")

st.sidebar.write("✅ BERT Model Loaded")
st.sidebar.write("✅ Gemini Connected")
st.sidebar.write("🌐 Streamlit Running")

st.sidebar.write("---")
st.sidebar.info("Version 1.0")

# -----------------------------------
# Session State
# -----------------------------------
if "emotion" not in st.session_state:
    st.session_state.emotion = ""

if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0

if "response" not in st.session_state:
    st.session_state.response = ""
    # -----------------------------------
# Load BERT Model
# -----------------------------------
@st.cache_resource
def load_model():

    model = BertForSequenceClassification.from_pretrained(
        "models/bert"
    )

    tokenizer = BertTokenizer.from_pretrained(
        "models/bert"
    )

    with open(
        "models/bilstm/label_encoder.pkl",
        "rb"
    ) as f:

        label_encoder = pickle.load(f)

    model.eval()

    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = load_model()


# -----------------------------------
# Save History
# -----------------------------------
# -----------------------------------
# Save History
# -----------------------------------
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
# Title
# -----------------------------------
st.title("🤖 AI Learning Assistant")

st.markdown(
    "### Emotion Detection using BERT + Gemini"
)

st.write("---")

# -----------------------------------
# User Inputs
# -----------------------------------
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

# -----------------------------------
# Buttons
# -----------------------------------
col1, col2 = st.columns(2)

with col1:
    analyze = st.button("🔍 Analyze Emotion")

with col2:
    clear = st.button("🗑 Clear")

# -----------------------------------
# Clear Button
# -----------------------------------
if clear:
    st.rerun()
    # -----------------------------------
# Analyze Emotion
# -----------------------------------
if analyze:

    if problem.strip() == "":
        st.warning("Please enter your problem.")

    else:

        # -----------------------------
        # BERT Prediction
        # -----------------------------
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

        # -----------------------------
        # Gemini Prompt
        # -----------------------------
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

        # -----------------------------
        # Gemini Response
        # -----------------------------
        try:

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            gemini_response = response.text

        except Exception as e:

            gemini_response = f"Gemini Error: {e}"

        # -----------------------------
        # Save History
        # -----------------------------
        save_history(
            name,
            field,
            problem,
            emotion,
            confidence,
            gemini_response
        )
        # -----------------------------------
# Display Results
# -----------------------------------

        st.success("✅ Input received successfully!")

        st.write("---")

        left, right = st.columns(2)

        # -----------------------------
        # User Details
        # -----------------------------
        with left:

            st.subheader("👤 User Details")

            st.write("**Name:**", name)
            st.write("**Field:**", field)
            st.write("**Problem:**", problem)

        # -----------------------------
        # Prediction
        # -----------------------------
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

        # -----------------------------
        # Gemini Response
        # -----------------------------
        st.subheader("🤖 AI Suggestion")

        st.info(gemini_response)

# -----------------------------------
# Session History
# -----------------------------------

st.write("---")

st.subheader("📜 Session History")

if os.path.exists("history.csv"):

    history_df = pd.read_csv("history.csv")
    history_df = history_df.tail(10)

    history_df["AI Response"] = history_df["AI Response"].str[:60] + "..."

    st.dataframe(history_df, use_container_width=True)

    col1, col2 = st.columns(2)

with col1:
    with open("history.csv", "rb") as file:
        st.download_button(
            "📥 Download History",
            file,
            "history.csv",
            "text/csv"
        )

with col2:
    if st.button("🗑 Clear History"):
        os.remove("history.csv")
        st.success("History Cleared Successfully!")
        st.rerun()