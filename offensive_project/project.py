import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from timeit import default_timer as timer
from PIL import Image, ImageFilter, ImageOps
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score
from statistics import mean
from sklearn.metrics import hamming_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

from sklearn.metrics import roc_auc_score, confusion_matrix
import statistics
from sklearn.metrics import recall_score

from wordcloud import WordCloud
from collections import Counter

from sklearn.pipeline import Pipeline

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import warnings
import joblib
import nltk
from deep_translator import MyMemoryTranslator
import requests  # <-- added for Manglish conversion

# ---------------------------

# ---------- NLTK downloads ----------
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)

# ---------- Labels and threshold ----------
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
THRESHOLD = 0.5

#tokenizer

def W_Cloud(token):
    """
    Visualize the most common words contributing to the token.
    """
    threat_context = train[train[token] == 1]
    threat_text = threat_context.comment_text
    neg_text = pd.Series(threat_text).str.cat(sep=' ')
    wordcloud = WordCloud(width=1600, height=800,
                          max_font_size=200).generate(neg_text)

    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud.recolor(colormap="Blues"), interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Most common words assosiated with {token} comment", size=20)
    plt.show()



# 🧩 Custom tokenizer 

def tokenize(text):
    '''
    Tokenize text and return a non-unique list of tokenized words found in the text. 
    Normalize to lowercase, strip punctuation, remove stop words, filter non-ascii characters.
    Lemmatize the words and lastly drop words of length < 3.
    '''
    text = text.lower()
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text)
    words = nopunct.split(' ')
    # remove any non ascii
    words = [word.encode('ascii', 'ignore').decode('ascii') for word in words]
    lmtzr = WordNetLemmatizer()
    words = [lmtzr.lemmatize(w) for w in words]
    words = [w for w in words if len(w) > 2]
    return words

# ---------- Load vectorizer & model (your trained artifacts) ----------
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("lr_model.pkl")

# ---------- Prediction wrapper ----------
def predict_toxicity(input_text: str):
    sample_text = [input_text]
    X_test_vec = vectorizer.transform(sample_text)
    prob = model.predict_proba(X_test_vec)
    predicted_labels = [labels[i] for i, p in enumerate(prob[0]) if p >= THRESHOLD]
    prob_dict = {labels[i]: float(prob[0][i]) for i in range(len(labels))}
    return predicted_labels, prob_dict


# ---------- Manglish Conversion Functions ----------
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






# ---------- Translator ----------
translator = MyMemoryTranslator(source='malayalam', target='english')

def translate_malayalam_to_english(text: str) -> str:
    try:
        return translator.translate(text)
    except:
        return text

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
    
# ---------- Streamlit UI ----------
st.set_page_config(page_title="Toxicity Detector", layout="wide")

# Apply custom styling
st.markdown("""
    <style>
        .stApp {background-color: #f5f5f5;}
        h1 {color: #2c3e50;}
        .big-font {font-size:20px !important;}
        .stButton>button {background-color:#3498db;color:white;height:3em;width:12em;border-radius:10px;}
    </style>
""", unsafe_allow_html=True)

# ---------- Home Page ----------
if 'lang' not in st.session_state:
    st.title("🚀 Toxicity Detection — English, Malayalam & Manglish")
    st.markdown("### Choose your language to proceed")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("English"):
            st.session_state['lang'] = 'en'
            st.stop()
    with col2:
        if st.button("Malayalam"):
            st.session_state['lang'] = 'ml'
            st.stop()
    with col3:
        if st.button("Manglish"):
            st.session_state['lang'] = 'manglish'
            st.stop()
    st.stop()

# ---------- English Page ----------
if st.session_state.get('lang') == 'en':
    st.header("English — Enter text or upload an image")
    st.markdown("You can either type/paste text manually, or upload an image containing text (JPG/PNG).")
    input_mode = st.radio("Input mode", ("Manual text", "Upload image"))
    text_to_predict = ""
    if input_mode == "Manual text":
        text_to_predict = st.text_area("Enter English text here", height=200)
    else:
        uploaded_file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            img=Image.open(uploaded_file)
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            with st.spinner("Extracting text from image..."):
                extracted = extract_text_from_image(uploaded_file)
            if extracted:
                text_to_predict = extracted
            else:
                st.warning("Could not extract text from the uploaded image. Try a clearer image.")
    if st.button("Predict Toxicity (English)"):
        if not text_to_predict.strip():
            st.error("Please provide some text.")
        else:
            with st.spinner("Running prediction..."):
                detected, probs = predict_toxicity(text_to_predict)
            if detected:
                st.error(f"⚠ Offensive categories detected: {', '.join(detected)}")
            else:
                st.success("✅ No offensive categories detected.")
            df = pd.DataFrame(list(probs.items()), columns=["Category", "Probability"])
            df['Probability'] = (df['Probability']*100).round(2).astype(str) + '%'
            st.table(df.sort_values(by='Probability', ascending=False).reset_index(drop=True))

# ---------- Malayalam Page ----------
if st.session_state.get('lang') == 'ml':
    st.header("Malayalam — Enter text to translate + detect")
    mal_text = st.text_area("Enter Malayalam text here", height=200)
    if st.button("Translate & Predict (Malayalam → English)"):
        if not mal_text.strip():
            st.error("Please enter Malayalam text.")
        else:
            with st.spinner("Translating to English..."):
                eng_text = translate_malayalam_to_english(mal_text)
            st.markdown("Translated English text:")
            st.write(eng_text)
            with st.spinner("Running prediction..."):
                detected, probs = predict_toxicity(eng_text)
            if detected:
                st.error(f"⚠ Offensive categories detected: {', '.join(detected)}")
            else:
                st.success("✅ No offensive categories detected.")
            df = pd.DataFrame(list(probs.items()), columns=["Category", "Probability"])
            df['Probability'] = (df['Probability']*100).round(2).astype(str) + '%'
            st.table(df.sort_values(by='Probability', ascending=False).reset_index(drop=True))

# ---------- Manglish Page ----------
if st.session_state.get('lang') == 'manglish':
    st.header("Manglish — Enter text (Malayalam in Latin letters)")
    manglish_text = st.text_area("Enter Manglish text here", height=200)

    if st.button(" Predict (Manglish → Ml → En)"):
        if not manglish_text.strip():
            st.error("Please enter some Manglish text.")
        else:
            with st.spinner("Converting Manglish to Malayalam..."):
                mal_text = manglish_to_malayalam(manglish_text)
            if mal_text:
                st.markdown("*Converted Malayalam text:*")
                st.write(mal_text)
                with st.spinner("Translating Malayalam to English..."):
                    eng_text = malayalam_to_english(mal_text)
                st.markdown("*Translated English text:*")
                st.write(eng_text)
                with st.spinner("Running toxicity detection..."):
                    detected, probs = predict_toxicity(eng_text)
                if detected:
                    st.error(f"⚠ Offensive categories detected: {', '.join(detected)}")
                else:
                    st.success("✅ No offensive categories detected.")
                df = pd.DataFrame(list(probs.items()), columns=["Category", "Probability"])
                df['Probability'] = (df['Probability']*100).round(2).astype(str) + '%'
                st.table(df.sort_values(by='Probability', ascending=False).reset_index(drop=True))
            else:
                st.error("Manglish to Malayalam conversion failed. Try again.")

# ---------- Back Button ----------
st.markdown("---")
if st.button("🔙 Back to language selection"):
    if 'lang' in st.session_state:
        del st.session_state['lang']
    st.stop()