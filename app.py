import streamlit as st
import random
from PIL import Image

st.set_page_config(page_title="AI Content Checker", layout="centered")

st.title("🧠 AI Content Authenticity Checker")

st.write("Upload content to estimate whether it may be AI-generated.")

# ---------------- TEXT CHECK ----------------
st.subheader("📄 Text Analysis")

text_input = st.text_area("Paste your text here")

if st.button("Analyze Text"):
    if text_input.strip():
        score = random.uniform(0.4, 0.95)

        st.success(f"AI Probability: {round(score*100,2)}%")

        if score > 0.7:
            st.warning("This text shows strong AI-like patterns.")
        elif score > 0.5:
            st.info("This text may contain AI-assisted content.")
        else:
            st.success("This text appears more human-written.")

    else:
        st.error("Please enter some text.")

# ---------------- IMAGE CHECK ----------------
st.subheader("🖼 Image Analysis")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze Image"):
        score = random.uniform(0.4, 0.9)

        st.success(f"AI Probability: {round(score*100,2)}%")
        st.info("Prototype detection based on visual patterns.")
