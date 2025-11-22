import os
import sys
import streamlit as st

# Ensure backend is accessible for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.model import MovieRecommender

st.set_page_config(page_title="Movie Generator AI", layout="wide")

# Optional CSS
css_path = os.path.join(os.path.dirname(__file__), "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Page title
st.markdown("<h1 class='title'>🎬 Movie Generator AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Find your next movie based on your mood.</p>", unsafe_allow_html=True)

# Emotion selection
emotion = st.selectbox(
    "How are you feeling today?",
    ["happy", "sad", "excited", "relaxed", "scared"],
)

st.write("")

# Button
if st.button("Generate Recommendations"):
    recommender = MovieRecommender()
    st.write(f"Showing recommendations for emotion: **{emotion}**")

    results = recommender.recommend_by_emotion(emotion)

    if results.empty:
        st.warning("No recommendations found. Try another mood.")
    else:
        cols = st.columns(4)
        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i % 4]:
                poster = row["poster_url"]
                title = row["title"]
                overview = row["overview"]
                genres = row["genre_label"]

                if isinstance(poster, str) and poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/500x750?text=No+Image", use_container_width=True)

                st.markdown(f"<h3 class='movie-title'>{title}</h3>", unsafe_allow_html=True)
                st.caption(f"Genres: {genres}")

                with st.expander("Overview"):
                    st.write(overview)
