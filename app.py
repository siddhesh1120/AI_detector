import streamlit as st
import requests
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Content Checker", layout="centered")

st.title("🧠 AI Content Authenticity Checker")
st.write("Analyze whether text or images may be AI-generated.")

# ---------------- TEXT DETECTION FUNCTION ----------------
def detect_ai_text(text):
    API_URL = "https://api-inference.huggingface.co/models/roberta-base-openai-detector"

    headers = {
        "Authorization": f"Bearer {st.secrets['HF_TOKEN']}"
    }

    response = requests.post(API_URL, headers=headers, json={"inputs": text})

    if response.status_code != 200:
        return None

    result = response.json()[0]

    for label in result:
        if label["label"] == "Fake":   # Fake = AI Generated
            return label["score"]

    return None


# ---------------- TEXT ANALYSIS UI ----------------
st.subheader("📄 Text Analysis")

text_input = st.text_area("Paste your text here")

if st.button("Analyze Text"):
    if text_input.strip():

        with st.spinner("Analyzing using transformer model..."):
            score = detect_ai_text(text_input)

        if score is None:
            st.error("Model is warming up. Please try again in 5 seconds.")
        else:
            percent = round(score * 100, 2)

            st.success(f"AI Probability: {percent}%")

            # Interpretation
            if percent > 75:
                st.warning("This text is highly likely AI-generated.")
            elif percent > 40:
                st.info("This text may contain AI-assisted writing.")
            else:
                st.success("This text appears human-written.")

    else:
        st.error("Please enter some text to analyze.")


# ---------------- IMAGE SECTION (Prototype UI) ----------------
st.subheader("🖼 Image Analysis (Prototype)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.info("Image detection module can be extended using CLIP / GAN fingerprint models.")
