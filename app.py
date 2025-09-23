import streamlit as st
st.set_option('browser.gatherUsageStats', False)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import speech_recognition as sr
import pyttsx3

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="My Smart Library Assistant", layout="wide")

# Background CSS
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(rgba(10, 25, 47, 0.9), rgba(10, 25, 47, 0.9)),
            url('https://images.unsplash.com/photo-1512820790803-83ca734da794') no-repeat center center fixed;
            background-size: cover;
        }
        .big-title {font-size:50px; color:#a5f3fc; text-align:center; margin-top:30px;}
        .sub-title {font-size:20px; color:#e5e7eb; text-align:center; margin-bottom:40px;}
        .section-title {font-size:28px; color:#64ffda; margin-top:30px;}
        .scroll-container {display:flex; overflow-x:auto; gap:20px; padding:10px;}
        .book-card {
            min-width:200px;
            background: rgba(255,255,255,0.1);
            border-radius:12px;
            padding:10px;
            text-align:center;
            color:white;
        }
        .book-card img {border-radius:8px; margin-bottom:8px;}
        a {text-decoration:none;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='big-title'>📚 My Smart Library Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Explore your favorite books with AI</div>", unsafe_allow_html=True)

# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_data():
    books = pd.read_csv("Books.csv", encoding="latin-1", low_memory=False)
    books = books.dropna(subset=["Book-Title"])
    return books

books = load_data()

# -------------------------
# TF-IDF Model
# -------------------------
@st.cache_resource
def build_model(books):
    tfidf = TfidfVectorizer(stop_words="english", min_df=2)
    tfidf_matrix = tfidf.fit_transform(books["Book-Title"].astype(str))
    return tfidf, tfidf_matrix

tfidf, tfidf_matrix = build_model(books)

# -------------------------
# Recommend Books
# -------------------------
def recommend_books(title, num=5):
    title = title.lower()
    matches = books[books["Book-Title"].str.lower().str.contains(title, na=False)]
    if matches.empty:
        return pd.DataFrame()
    idx = matches.index[0]
    cosine_sim = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num+1]
    book_indices = [i[0] for i in sim_scores]
    return books.iloc[book_indices][["Book-Title", "Book-Author", "Year-Of-Publication", "Image-URL-M"]]

# -------------------------
# Voice Assistant
# -------------------------
def voice_search():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak the book title or author")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        return text
    except:
        st.error("Sorry, could not recognize your voice.")
        return None

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# -------------------------
# (Rest of your existing UI code remains unchanged)
# -------------------------