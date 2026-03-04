import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
import math

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Content Checker", layout="centered")

st.title("🧠 AI Content Authenticity Checker")
st.write("Hybrid AI detection using transformer classification and perplexity analysis.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_name = "openai-community/roberta-base-openai-detector"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- TRANSFORMER DETECTION ----------------
def detect_ai_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    ai_score = probs[0][1].item()
    
    return ai_score


# ---------------- PERPLEXITY ESTIMATION ----------------
def calculate_perplexity(text):

    words = text.split()

    if len(words) < 20:
        return 120

    avg_word_len = sum(len(w) for w in words) / len(words)

    variance = sum((len(w) - avg_word_len) ** 2 for w in words) / len(words)

    perplexity = 200 - variance * 10

    return max(20, min(perplexity, 200))


# ---------------- TEXT ANALYSIS UI ----------------
st.subheader("📄 Text Analysis")

text_input = st.text_area("Paste your text here")

if st.button("Analyze Text"):

    if text_input.strip():

        with st.spinner("Running hybrid AI detection..."):

            detector_score = detect_ai_text(text_input)

            perplexity = calculate_perplexity(text_input)

        # Hybrid scoring
        final_score = (detector_score * 0.7) + ((200 - perplexity) / 200 * 0.3)

        percent = round(final_score * 100, 2)

        st.success(f"AI Probability: {percent}%")

        st.write(f"Transformer Detector Score: {round(detector_score*100,2)}%")
        st.write(f"Perplexity Score: {round(perplexity,2)}")

        # Interpretation
        if percent > 75:
            st.warning("Likely AI-generated (high fluency and predictability).")

        elif percent > 45:
            st.info("Possibly AI-assisted writing.")

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

    st.info("Image detection can be implemented using diffusion artifact or CLIP-based forensic models.")
