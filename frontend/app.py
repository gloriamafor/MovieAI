import streamlit as st
import pandas as pd
from backend.model import MovieRecommender

st.set_page_config(page_title="Movie Generator AI", layout="wide")

# Custom CSS
with open("frontend/assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🎬 Movie Generator AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Find your next movie based on your mood!</p>", unsafe_allow_html=True)

emotion = st.selectbox("Select your mood", ["Happy", "Sad", "Excited", "Relaxed", "Scared"])
st.write("")

if st.button("Generate Recommendations"):
    recommender = MovieRecommender()
    results = recommender.recommend_by_emotion(emotion)

    if results.empty:
        st.warning("No movies found for this mood. Try another one!")
    else:
        cols = st.columns(5)
        for i, (idx, row) in enumerate(results.iterrows()):
            with cols[i % 5]:
                st.image(row['poster_url'], use_container_width=True)
                st.markdown(f"<h3 class='movie-title'>{row['title']}</h3>", unsafe_allow_html=True)
                st.caption(f"Genre: {row['genre']}")