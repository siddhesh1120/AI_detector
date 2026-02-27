import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="AI Content Checker", layout="centered")

st.title("🧠 AI Content Authenticity Checker")
st.write("Detection powered by transformer-based classification.")

# ---------------- TEXT DETECTION ----------------
def detect_ai_text(text):
    API_URL = "https://api-inference.huggingface.co/models/openai-community/roberta-base-openai-detector"

    headers = {
        "Authorization": f"Bearer {st.secrets['HF_TOKEN']}"
    }

    response = requests.post(API_URL, headers=headers, json={"inputs": text})

    if response.status_code != 200:
        st.error(f"Inference API Error: {response.status_code}")
        return None

    result = response.json()

    if isinstance(result, list):
        for label in result[0]:
            if label["label"] == "Fake":
                return label["score"]

    return None


# ---------------- TEXT UI ----------------
st.subheader("📄 Text Analysis")

text_input = st.text_area("Paste your text here")

if st.button("Analyze Text"):
    if text_input.strip():

        with st.spinner("Contacting detection model..."):
            score = detect_ai_text(text_input)

        if score is not None:
            percent = round(score * 100, 2)
            st.success(f"AI Probability: {percent}%")

            if percent > 75:
                st.warning("Likely AI-generated.")
            elif percent > 40:
                st.info("Possibly AI-assisted.")
            else:
                st.success("Likely human-written.")
        else:
            st.error("Model did not return a valid response.")

    else:
        st.error("Please enter some text.")


# ---------------- IMAGE SECTION ----------------
st.subheader("🖼 Image Analysis (UI Prototype)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.info("Image detection module can be integrated using CLIP-based forensic models.")
