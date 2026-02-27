import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image

st.set_page_config(page_title="AI Content Checker", layout="centered")

st.title("🧠 AI Content Authenticity Checker")
st.write("Detection powered by transformer-based classification.")

# ---------------- LOAD MODEL (cached so it loads once) ----------------
@st.cache_resource
def load_model():
    model_name = "openai-community/roberta-base-openai-detector"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- TEXT DETECTION ----------------
def detect_ai_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    ai_score = probs[0][1].item()
    return ai_score


# ---------------- TEXT UI ----------------
st.subheader("📄 Text Analysis")

text_input = st.text_area("Paste your text here")

if st.button("Analyze Text"):
    if text_input.strip():
        with st.spinner("Running transformer analysis..."):
            score = detect_ai_text(text_input)

        percent = round(score * 100, 2)

        st.success(f"AI Probability: {percent}%")

        if percent > 75:
            st.warning("Likely AI-generated.")
        elif percent > 40:
            st.info("Possibly AI-assisted.")
        else:
            st.success("Likely human-written.")

    else:
        st.error("Please enter some text.")


# ---------------- IMAGE SECTION ----------------
st.subheader("🖼 Image Analysis (Prototype)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.info("Image detection module can be extended using CLIP-based forensic models.")
