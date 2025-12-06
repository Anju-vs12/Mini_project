import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from PIL import Image, ImageFilter, ImageOps
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pytesseract
import joblib
import nltk
import requests
from deep_translator import MyMemoryTranslator

# ---------- NLTK ----------
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)

# ---------- Labels ----------
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
THRESHOLD = 0.5

# ---------- Tokenizer ----------
def tokenize(text):
    text = text.lower()
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text)
    words = nopunct.split(' ')
    words = [word.encode('ascii', 'ignore').decode('ascii') for word in words]
    lmtzr = WordNetLemmatizer()
    words = [lmtzr.lemmatize(w) for w in words if len(w) > 2]
    return words

# ---------- Load Model ----------
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("lr_model.pkl")

# ---------- Prediction ----------
def predict_toxicity(input_text: str):
    sample_text = [input_text]
    X_test_vec = vectorizer.transform(sample_text)
    prob = model.predict_proba(X_test_vec)
    predicted_labels = [labels[i] for i, p in enumerate(prob[0]) if p >= THRESHOLD]
    prob_dict = {labels[i]: float(prob[0][i]) for i in range(len(labels))}
    return predicted_labels, prob_dict

# ---------- Translators ----------
translator = MyMemoryTranslator(source='malayalam', target='english')
def translate_malayalam_to_english(text: str) -> str:
    try:
        return translator.translate(text)
    except:
        return text

def manglish_to_malayalam(text):
    url = "https://inputtools.google.com/request"
    params = {
        "text": text,
        "itc": "ml-t-i0-und",
        "num": 1,
        "cp": 0,
        "cs": 1,
        "ie": "utf-8",
        "oe": "utf-8"
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data[0] == "SUCCESS":
        return data[1][0][1][0]
    else:
        return None

def malayalam_to_english(text):
    url = "https://translate.googleapis.com/translate_a/single"
    params = {
        "client": "gtx",
        "sl": "ml",
        "tl": "en",
        "dt": "t",
        "q": text
    }
    response = requests.get(url, params=params)
    result = response.json()
    return result[0][0][0]

# ---------- OCR ----------
def preprocess_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    img = ImageOps.grayscale(pil_img)
    base_w, base_h = img.size
    scale = 2 if max(base_w, base_h) < 1200 else 1
    if scale != 1:
        img = img.resize((base_w * scale, base_h * scale), Image.LANCZOS)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    arr = np.array(img)
    thresh = arr.mean()
    arr_bin = (arr > thresh).astype(np.uint8) * 255
    return Image.fromarray(arr_bin)

def extract_text_from_image(uploaded_file) -> str:
    try:
        img = Image.open(uploaded_file).convert('RGB')
    except:
        return ""
    proc = preprocess_image_for_ocr(img)
    try:
        text = pytesseract.image_to_string(proc, lang='eng')
        return text.replace('\n', ' ').strip()
    except:
        return ""

# ---------- Streamlit Page Config ----------
st.set_page_config(page_title="Toxicity Detector", layout="wide")

# ---------- Custom Styling ----------
st.markdown("""
<style>
    body {
        background-color: #f0f2f6;
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #f7f8fc 0%, #eef2f7 100%);
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    .main-card {
        background-color: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 25px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background: linear-gradient(90deg, #3498db, #2ecc71);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        border: none;
        transition: 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2ecc71, #3498db);
        transform: scale(1.03);
    }
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
    }
    .footer {
        text-align: center;
        font-size: 13px;
        color: #777;
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Language Selection ----------
if 'lang' not in st.session_state:
    st.title("🚀 Toxicity Detection")
    st.subheader("Multilingual — English, Malayalam & Manglish")
  
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(" English"):
                st.session_state['lang'] = 'en'
                st.stop()
        with col2:
            if st.button(" Malayalam"):
                st.session_state['lang'] = 'ml'
                st.stop()
        with col3:
            if st.button(" Manglish"):
                st.session_state['lang'] = 'manglish'
                st.stop()
    st.stop()

# ---------- English ----------
if st.session_state.get('lang') == 'en':
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.header("🇬🇧 English Toxicity Detection")
    st.markdown("Type or upload an image to analyze English text.")
    mode = st.radio("Input mode", ("Manual text", "Upload image"))
    text = ""
    if mode == "Manual text":
        text = st.text_area("Enter English text here:", height=180)
    else:
        file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
        if file:
            st.image(file, caption="Uploaded Image", use_container_width=True)
            with st.spinner("Extracting text..."):
                text = extract_text_from_image(file)
            if not text:
                st.warning("No readable text found.")
    if st.button("🔍 Predict Toxicity (English)"):
        if not text.strip():
            st.error("Please provide some text.")
        else:
            with st.spinner("Analyzing..."):
                detected, probs = predict_toxicity(text)
            if detected:
                st.error(f"⚠ Offensive categories detected: {', '.join(detected)}")
            else:
                st.success("✅ No offensive categories detected.")
            df = pd.DataFrame(list(probs.items()), columns=["Category", "Probability"])
            df['Probability'] = (df['Probability']*100).round(2).astype(str) + '%'
            st.table(df.sort_values(by='Probability', ascending=False))
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Malayalam ----------
if st.session_state.get('lang') == 'ml':
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.header("🇮🇳 Malayalam → English Toxicity Detection")
    mal_text = st.text_area("Enter Malayalam text here:", height=180)
    if st.button("🌐 Translate & Predict"):
        if not mal_text.strip():
            st.error("Please enter Malayalam text.")
        else:
            with st.spinner("Translating..."):
                eng = translate_malayalam_to_english(mal_text)
            st.info(f"*Translated English:* {eng}")
            with st.spinner("Analyzing..."):
                detected, probs = predict_toxicity(eng)
            if detected:
                st.error(f"⚠ Offensive categories detected: {', '.join(detected)}")
            else:
                st.success("✅ No offensive categories detected.")
            df = pd.DataFrame(list(probs.items()), columns=["Category", "Probability"])
            df['Probability'] = (df['Probability']*100).round(2).astype(str) + '%'
            st.table(df.sort_values(by='Probability', ascending=False))
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Manglish ----------
if st.session_state.get('lang') == 'manglish':
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.header("🔤 Manglish → Malayalam → English Detection")
    mang_text = st.text_area("Enter Manglish text:", height=180)
    if st.button("🚀 Predict (Manglish → Ml → En)"):
        if not mang_text.strip():
            st.error("Please enter some Manglish text.")
        else:
            with st.spinner("Converting Manglish → Malayalam..."):
                mal = manglish_to_malayalam(mang_text)
            if mal:
                st.success(f"*Malayalam:* {mal}")
                with st.spinner("Translating Malayalam → English..."):
                    eng = malayalam_to_english(mal)
                st.info(f"*English Translation:* {eng}")
                with st.spinner("Running detection..."):
                    detected, probs = predict_toxicity(eng)
                if detected:
                    st.error(f"⚠ Offensive categories detected: {', '.join(detected)}")
                else:
                    st.success("✅ No offensive categories detected.")
                df = pd.DataFrame(list(probs.items()), columns=["Category", "Probability"])
                df['Probability'] = (df['Probability']*100).round(2).astype(str) + '%'
                st.table(df.sort_values(by='Probability', ascending=False))
            else:
                st.error("Conversion failed. Try again.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Back ----------
st.markdown("---")
if st.button("🔙 Back to Language Selection"):
    if 'lang' in st.session_state:
        del st.session_state['lang']
    st.stop()

# ---------- Footer ----------
