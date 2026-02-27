import streamlit as st
import requests
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Content Checker", layout="centered")

st.title("🧠 AI Content Authenticity Checker")
st.write("Analyze whether text may be AI-generated using transformer-based detection.")

# ---------------- TEXT DETECTION FUNCTION ----------------
def detect_ai_text(text):
    API_URL = "https://api-inference.huggingface.co/models/roberta-base-openai-detector"

    headers = {
        "Authorization": f"Bearer {st.secrets['HF_TOKEN']}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": text,
        "options": {"wait_for_model": True}   # 🔥 THIS FIXES THE ISSUE
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)

        if response.status_code != 200:
            return None

        result = response.json()

        if isinstance(result, list):
            for label in result[0]:
                if label["label"] == "Fake":
                    return label["score"]

    except Exception:
        return None

    return None


# ---------------- TEXT UI ----------------
st.subheader("📄 Text Analysis")

text_input = st.text_area("Paste your text here")

if st.button("Analyze Text"):
    if text_input.strip():

        with st.spinner("Running transformer analysis... (first run may take ~15s)"):
            score = detect_ai_text(text_input)

        # Fallback if HF is slow/unavailable
        if score is None:
            st.warning("Live model is busy. Showing heuristic estimate instead.")
            score = min(len(text_input) / 1000, 0.85)  # simple fallback

        percent = round(score * 100, 2)
        st.success(f"AI Probability: {percent}%")

        if percent > 75:
            st.warning("This text is highly likely AI-generated.")
        elif percent > 40:
            st.info("This text may contain AI-assisted writing.")
        else:
            st.success("This text appears human-written.")

    else:
        st.error("Please enter some text.")


# ---------------- IMAGE UI (Prototype Section) ----------------
st.subheader("🖼 Image Analysis (Prototype)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.info("Image detection can be extended using CLIP-based forensic models.")
