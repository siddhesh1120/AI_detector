import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Content Checker", layout="centered")

st.title("🧠 AI Content Authenticity Checker")
st.write("Detect whether text is AI-generated or human-written.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_name = "roberta-base-openai-detector"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- TEXT DETECTION ----------------
def detect_ai_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    # Correct label mapping for roberta-base-openai-detector:
    # Index 0 = Fake (AI), Index 1 = Real (Human)
    ai_score = probs[0][0].item()
    human_score = probs[0][1].item()

    return human_score, ai_score

# ---------------- UI ----------------
st.subheader("📄 Text Analysis")

text_input = st.text_area("Paste your text here")

if st.button("Analyze Text"):

    if text_input.strip():

        with st.spinner("Analyzing..."):
            human_score, ai_score = detect_ai_text(text_input)

        ai_percent = round(ai_score * 100, 2)
        human_percent = round(human_score * 100, 2)

        st.write(f"🧑 Human Probability: {human_percent}%")
        st.write(f"🤖 AI Probability: {ai_percent}%")

        st.progress(ai_score)

        if ai_score > 0.65:
            st.error("⚠️ Likely AI-generated")
        elif ai_score > 0.45:
            st.warning("⚠️ Possibly AI-assisted")
        else:
            st.success("✅ Likely human-written")

    else:
        st.error("Please enter some text.")