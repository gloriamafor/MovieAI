'''import os
from typing import Dict, List, Any, Iterable

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def _require_tmdb_key() -> str:
    if not TMDB_API_KEY:
        raise RuntimeError(
            "TMDB_API_KEY is not set. Create a .env file with TMDB_API_KEY=your_key_here."
        )
    return TMDB_API_KEY


def fetch_genres() -> Dict[str, int]:
    """
    Fetch TMDb genres and return a mapping: genre_name_lower -> genre_id
    """
    api_key = _require_tmdb_key()
    url = "https://api.themoviedb.org/3/genre/movie/list"
    params = {"api_key": api_key, "language": "en-US"}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return {g["name"].lower(): g["id"] for g in data.get("genres", [])}


def fetch_movies_by_genres(
    genres: Iterable[int] | Iterable[str], page: int = 1
) -> List[Dict[str, Any]]:
    """
    Fetch popular movies from TMDb given one or more genre IDs.
    """
    api_key = _require_tmdb_key()

    if isinstance(genres, (list, tuple, set)):
        genre_param = ",".join(str(g) for g in genres)
    else:
        genre_param = str(genres)

    url = "https://api.themoviedb.org/3/discover/movie"
    params = {
        "api_key": api_key,
        "with_genres": genre_param,
        "sort_by": "popularity.desc",
        "page": page,
        "language": "en-US",
        "include_adult": "false",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", [])


def generate_ai_poster(title: str, genre: str, overview: str) -> str | None:
    """
    Use OpenAI's image API to generate a movie poster.
    Returns the image URL, or None if OPENAI_API_KEY is not configured.
    """
    if not OPENAI_API_KEY:
        # Graceful fallback: caller can decide to use a TMDb poster instead
        return None

    client = OpenAI()

    prompt = (
        f"High-quality cinematic movie poster for a {genre} film titled '{title}'. "
        f"The film is about: {overview[:500]}..."
    )

    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="512x768",
        n=1,
    )
    return result.data[0].url
'''
import os
import random
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")


class MovieRecommender:
    GENRE_COLUMNS = [
        "Action","Adventure","Animation","Comedy","Crime","Documentary",
        "Drama","Family","Fantasy","History","Horror","Music","Mystery",
        "Romance","Science Fiction","TV Movie","Thriller","War","Western"
    ]

    EMOTION_GENRE_MAP = {
        "happy": ["Comedy", "Adventure", "Family", "Animation", "Fantasy", "Romance"],
        "sad": ["Drama", "Music", "Documentary"],
        "excited": ["Action", "Science Fiction", "Thriller", "War"],
        "relaxed": ["Romance", "Family", "Comedy"],
        "scared": ["Horror", "Mystery", "Thriller"],
    }

    def __init__(self):
        base = os.path.dirname(__file__)

        pre_path  = os.path.join(base, "data", "movies_preprocessed.csv")
        meta_path = os.path.join(base, "data", "movies_metadata.csv")

        df_pre  = pd.read_csv(pre_path)
        df_meta = pd.read_csv(meta_path)

        # Normalize titles
        df_pre["title_clean"]  = df_pre["title"].astype(str).str.strip().str.lower()
        df_meta["title_clean"] = df_meta["title"].astype(str).str.strip().str.lower()

        # Use only needed metadata columns
        df_meta = df_meta[["title_clean", "poster_path", "overview"]].rename(
            columns={"overview": "overview_meta"}
        )

        # Exact merge (fast)
        df = df_pre.merge(df_meta, on="title_clean", how="left")

        # Prefer preprocessed overview
        df["overview"] = df.apply(
            lambda r: r["overview"]
            if isinstance(r["overview"], str) and r["overview"].strip() != ""
            else r["overview_meta"],
            axis=1,
        )
        df = df.drop(columns=["overview_meta"])

        # Clean poster_path values from metadata
        def clean_poster(p):
            p = str(p)
            if p.lower() in ["", "0", "nan", "none", "null"]:
                return None
            if not p.startswith("/"):
                return None
            return p

        df["poster_path"] = df["poster_path"].apply(clean_poster)

        self.movies = df
        self._add_genre_label_column()

        # In-memory cache: title_clean -> full poster URL
        self.poster_cache: dict[str, str | None] = {}

    def _add_genre_label_column(self):
        def genres(row):
            return ", ".join([g for g in self.GENRE_COLUMNS if row.get(g, 0) == 1]) or "Unknown"
        self.movies["genre_label"] = self.movies.apply(genres, axis=1)

    # --- TMDB lookup for missing posters ---
    def _fetch_tmdb_poster_url(self, title: str) -> str | None:
        """Search TMDB by title and return a full poster URL if found."""
        if not TMDB_API_KEY:
            return None

        try:
            resp = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params={
                    "api_key": TMDB_API_KEY,
                    "query": title,
                    "include_adult": "false",
                    "language": "en-US",
                },
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results") or []
            if not results:
                return None

            # Take the best match (first result)
            poster_path = results[0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
            return None
        except Exception:
            return None

    def recommend_by_emotion(self, emotion, n_results=10):
        emotion = emotion.lower().strip()
        genres = self.EMOTION_GENRE_MAP.get(emotion, ["Drama"])

        mask = pd.Series(False, index=self.movies.index)
        for g in genres:
            if g in self.movies.columns:
                mask |= (self.movies[g] == 1)

        movies = self.movies[mask]
        if movies.empty:
            return movies

        sample = movies.sample(min(n_results, len(movies)))

        poster_urls = []
        for _, row in sample.iterrows():
            title = row["title"]
            title_key = row["title_clean"]
            poster_path = row["poster_path"]

            # 1) If we already looked this title up, use cache
            if title_key in self.poster_cache:
                poster_urls.append(self.poster_cache[title_key])
                continue

            # 2) If we have a poster_path from metadata, use it
            if isinstance(poster_path, str):
                url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                poster_urls.append(url)
                self.poster_cache[title_key] = url
                continue

            # 3) Otherwise, call TMDB API by title
            tmdb_url = self._fetch_tmdb_poster_url(title)
            poster_urls.append(tmdb_url)
            self.poster_cache[title_key] = tmdb_url

        sample["poster_url"] = poster_urls
        return sample
