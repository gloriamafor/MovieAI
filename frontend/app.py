import os
import sys

import streamlit as st

# Make sure we can import from ../backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.model import MovieRecommender  # type: ignore
from backend.emotion_analysis import detect_emotion_and_mood  # type: ignore

# ------------------------
# Page config & optional CSS
# ------------------------
st.set_page_config(page_title="Movie Generator AI", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------------
# Backend client
# ------------------------
@st.cache_resource(show_spinner=False)
def get_recommender():
    return MovieRecommender()

recommender = get_recommender()

# ------------------------
# UI Layout
# ------------------------
st.title("🎬 Movie Generator AI")
st.markdown(
    "Powered by TMDB + Emotion AI. Search by title, discover by country/genre/mood, "
    "or let the app pick movies based on how you feel."
)

tab_search, tab_country, tab_genre, tab_mood, tab_emotion = st.tabs(
    ["🔍 Search", "🌍 Country", "🎭 Genre", "😊 Mood", "🧠 Emotion"]
)

# ---------------------------------
# 🔍 SEARCH TAB
# ---------------------------------
with tab_search:
    st.subheader("Search TMDB by Title or Keyword")

    with st.sidebar:
        st.header("Search Options")
        year_input = st.number_input(
            "Year (optional)", min_value=1900, max_value=2100, value=2025, step=1
        )
        use_year = st.checkbox("Filter by year", value=False)
        include_adult = st.checkbox("Include adult content", value=False)

    query = st.text_input(
        "Movie title or keyword",
        placeholder="e.g. Avengers, romance, sci-fi, Spider-Man",
    )
    search_btn = st.button("Search TMDB")

    if search_btn:
        if not query.strip():
            st.warning("Please enter a search term first.")
        else:
            with st.spinner("Searching TMDB for movies..."):
                try:
                    year_arg = int(year_input) if use_year else None
                    movies = recommender.search_movies(
                        query=query,
                        year=year_arg,
                        include_adult=include_adult,
                    )
                except Exception as e:
                    st.error(f"Error while calling TMDB: {e}")
                    movies = []

            if not movies:
                st.warning("No movies found. Try a different search term.")
            else:
                st.markdown(f"### Results for: `{query}` ({len(movies)} found)")
                for movie in movies:
                    st.markdown("---")
                    cols = st.columns([1, 2])

                    with cols[0]:
                        poster_url = movie.get("poster_url")
                        if poster_url:
                            st.image(poster_url, use_container_width=True)
                        else:
                            st.image(
                                "https://via.placeholder.com/500x750?text=No+Image",
                                use_container_width=True,
                            )

                    with cols[1]:
                        title = movie.get("title") or "Untitled"
                        st.subheader(title)

                        meta_parts = []
                        if movie.get("release_date"):
                            meta_parts.append(f"📅 {movie['release_date']}")
                        if movie.get("vote_average") is not None:
                            meta_parts.append(
                                f"⭐ {movie['vote_average']} ({movie.get('vote_count', 0)} votes)"
                            )
                        if movie.get("original_language"):
                            meta_parts.append(
                                f"🌐 {movie['original_language'].upper()}"
                            )
                        if meta_parts:
                            st.caption(" • ".join(meta_parts))

                        overview = movie.get("overview") or "No overview available."
                        st.write(overview)

# ---------------------------------
# 🌍 COUNTRY TAB
# ---------------------------------
with tab_country:
    st.subheader("Discover Movies by Country")

    country = st.selectbox(
        "Select country (origin):",
        ["US", "GB", "NG", "FR", "KR", "IN", "JP", "ES", "IT", "DE"],
    )

    if st.button("Find Movies from Country"):
        with st.spinner("Loading movies..."):
            movies = recommender.discover_by_country(country)

        if not movies:
            st.warning("No movies found.")
        else:
            for movie in movies:
                st.markdown("---")
                cols = st.columns([1, 2])

                with cols[0]:
                    poster_url = movie.get("poster_url")
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.image(
                            "https://via.placeholder.com/500x750?text=No+Image",
                            use_container_width=True,
                        )

                with cols[1]:
                    st.subheader(movie.get("title", "Untitled"))
                    if movie.get("release_date"):
                        st.caption(movie["release_date"])
                    st.write(movie.get("overview") or "No overview available.")

# ---------------------------------
# 🎭 GENRE TAB
# ---------------------------------
with tab_genre:
    st.subheader("Discover Movies by Genre")

    with st.spinner("Loading genres from TMDB..."):
        genre_map = recommender.get_genres()

    selected = st.multiselect("Select genres:", list(genre_map.keys()))

    if st.button("Find Movies by Genre"):
        if not selected:
            st.warning("Select at least one genre.")
        else:
            genre_ids = [genre_map[name] for name in selected]
            with st.spinner("Loading movies..."):
                movies = recommender.discover_by_genre(genre_ids)

            if not movies:
                st.warning("No movies found for those genres.")
            else:
                for movie in movies:
                    st.markdown("---")
                    cols = st.columns([1, 2])

                    with cols[0]:
                        poster_url = movie.get("poster_url")
                        if poster_url:
                            st.image(poster_url, use_container_width=True)
                        else:
                            st.image(
                                "https://via.placeholder.com/500x750?text=No+Image",
                                use_container_width=True,
                            )

                    with cols[1]:
                        st.subheader(movie.get("title", "Untitled"))
                        if movie.get("release_date"):
                            st.caption(movie["release_date"])
                        st.write(movie.get("overview") or "No overview available.")

# ---------------------------------
# 😊 MOOD TAB (manual mood selection)
# ---------------------------------
with tab_mood:
    st.subheader("Discover Movies by Mood (Manual Choice)")

    mood = st.selectbox(
        "Pick a mood:",
        ["happy", "sad", "thrilling", "romantic", "funny", "cozy", "intense", "scary"],
    )

    if st.button("Find Movies for Mood"):
        with st.spinner("Finding movies for your mood..."):
            movies = recommender.discover_by_mood(mood)

        if not movies:
            st.warning("No movies found for that mood.")
        else:
            for movie in movies:
                st.markdown("---")
                cols = st.columns([1, 2])

                with cols[0]:
                    poster_url = movie.get("poster_url")
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.image(
                            "https://via.placeholder.com/500x750?text=No+Image",
                            use_container_width=True,
                        )

                with cols[1]:
                    st.subheader(movie.get("title", "Untitled"))
                    if movie.get("release_date"):
                        st.caption(movie["release_date"])
                    st.write(movie.get("overview") or "No overview available.")

# ---------------------------------
# 🧠 EMOTION TAB (text → emotion → mood → movies)
# ---------------------------------
with tab_emotion:
    st.subheader("Emotion-Based Recommendations")

    st.markdown(
        "Describe how you're feeling or what kind of vibe you want. "
        "We'll analyze your emotion and match it to a movie mood."
    )

    mood_text = st.text_area(
        "How are you feeling?",
        placeholder="e.g. I feel stressed and overwhelmed but I want something light and comforting.",
    )

    if st.button("Analyze Emotion & Recommend"):
        if not mood_text.strip():
            st.warning("Please enter something first.")
        else:
            with st.spinner("Analyzing your emotion..."):
                info = detect_emotion_and_mood(mood_text)

            raw_emotion = info["raw_emotion"]
            score = info["score"]
            mood = info["mood"]

            st.markdown(
                f"**Detected Emotion:** 🎭 `{raw_emotion}` "
                f"(confidence: {score})"
            )
            st.markdown(f"**Mapped Mood:** 😊 `{mood}`")

            with st.spinner("Finding movies for your mood..."):
                movies = recommender.discover_by_mood(mood)

            if not movies:
                st.warning("No movies found for that mood. Try again with a different description.")
            else:
                st.markdown("### Recommended Movies")
                for movie in movies[:10]:
                    st.markdown("---")
                    cols = st.columns([1, 2])

                    with cols[0]:
                        poster_url = movie.get("poster_url")
                        if poster_url:
                            st.image(poster_url, use_container_width=True)
                        else:
                            st.image(
                                "https://via.placeholder.com/500x750?text=No+Image",
                                use_container_width=True,
                            )

                    with cols[1]:
                        st.subheader(movie.get("title", "Untitled"))
                        if movie.get("release_date"):
                            st.caption(movie["release_date"])
                        st.write(movie.get("overview") or "No overview available.")
