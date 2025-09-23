import os
import zipfile

# -------------------------
# Folder and file setup
# -------------------------
project_name = "smart-library-assistant"
os.makedirs(project_name, exist_ok=True)
os.makedirs(os.path.join(project_name, ".streamlit"), exist_ok=True)

# 1️⃣ app.py content
app_py_content = '''import streamlit as st
st.set_option('browser.gatherUsageStats', False)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import speech_recognition as sr
import pyttsx3

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
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Smart AI Library", layout="wide")

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

st.markdown("<div class='big-title'>📘 Smart AI Library</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Search, Explore & Discover Books Instantly</div>", unsafe_allow_html=True)

# -------------------------
# Dataset Stats
# -------------------------
total_books = len(books)
min_year = books["Year-Of-Publication"].min()
max_year = books["Year-Of-Publication"].max()

st.markdown("<h2 class='section-title'>📊 Dataset Insights</h2>", unsafe_allow_html=True)
st.info(f"📚 Total Books: {total_books:,}\\n\\n📅 Years Range: {min_year} ➝ {max_year}")

# Download dataset as CSV
st.download_button(
    label="⬇️ Download Full Dataset (CSV)",
    data=books.to_csv(index=False).encode("utf-8"),
    file_name="Smart_AI_Library_Books.csv",
    mime="text/csv",
)

# -------------------------
# Search Box + Voice
# -------------------------
col1, col2 = st.columns([3,1])
with col1:
    search_query = st.text_input("🔎 Search books by title, author, or year:")
with col2:
    if st.button("🎤 Voice Search"):
        voice_text = voice_search()
        if voice_text:
            st.success(f"You said: {voice_text}")
            search_query = voice_text

# -------------------------
# Search Results
# -------------------------
if st.button("Search"):
    if search_query:
        results = books[
            books["Book-Title"].str.contains(search_query, case=False, na=False) |
            books["Book-Author"].str.contains(search_query, case=False, na=False) |
            books["Year-Of-Publication"].astype(str).str.contains(search_query, na=False)
        ].head(10)

        if not results.empty:
            st.markdown("<h2 class='section-title'>📚 Search Results</h2>", unsafe_allow_html=True)
            st.markdown("<div class='scroll-container'>", unsafe_allow_html=True)

            for _, row in results.iterrows():
                st.markdown(
                    f"""
                    <div class='book-card'>
                        <img src="{row['Image-URL-M']}" width="120"><br>
                        <b>{row['Book-Title']}</b><br>
                        ✍️ {row['Book-Author']}<br>
                        ({row['Year-Of-Publication']})<br><br>
                        <a href="{row['Image-URL-M']}" target="_blank" style="color:#64ffda;">📖 Read Online</a> | 
                        <a href="{row['Image-URL-M']}" download style="color:#64ffda;">⬇️ Download</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

            # Voice read first book
            first_book = results.iloc[0]
            speak(f"I found {len(results)} books. First one is {first_book['Book-Title']} by {first_book['Book-Author']}.")

        else:
            st.warning("No results found.")

        # Recommendations
        st.markdown("<h2 class='section-title'>🤖 You may also like</h2>", unsafe_allow_html=True)
        recs = recommend_books(search_query, num=5)

        if not recs.empty:
            st.markdown("<div class='scroll-container'>", unsafe_allow_html=True)
            for _, row in recs.iterrows():
                st.markdown(
                    f"""
                    <div class='book-card'>
                        <img src="{row['Image-URL-M']}" width="120"><br>
                        <b>{row['Book-Title']}</b><br>
                        ✍️ {row['Book-Author']}<br>
                        ({row['Year-Of-Publication']})
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Guide Section
# -------------------------
st.markdown("---")
st.header("📖 How to Use the Smart AI Library")
st.write("""
1. Type or **speak** the book title/author/year in the search box.  
2. Scroll results horizontally with book covers.  
3. Click **Read Online** to preview or **Download** to save the book.  
4. Listen to the **voice assistant** announce results.  
5. Explore recommendations to discover new books.  
""")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("© 2025 Smart AI Library | Designed with ❤️", unsafe_allow_html=True)
'''

with open(os.path.join(project_name, "app.py"), "w", encoding="utf-8") as f:
    f.write(app_py_content)

# 2️⃣ .streamlit/config.toml
config_content = '''[server]
headless = true
enableCORS = false
port = 8501
'''

with open(os.path.join(project_name, ".streamlit/config.toml"), "w", encoding="utf-8") as f:
    f.write(config_content)

# 3️⃣ requirements.txt
requirements = '''streamlit
pandas
scikit-learn
speechrecognition
pyttsx3
'''

with open(os.path.join(project_name, "requirements.txt"), "w", encoding="utf-8") as f:
    f.write(requirements)

# -------------------------
# 4️⃣ Create ZIP
# ------------------------- 