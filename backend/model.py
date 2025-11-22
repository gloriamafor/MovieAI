import os
import random
from typing import Optional, List, Dict, Any

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")


class MovieRecommender:
    GENRE_COLUMNS: List[str] = [
        "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
        "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
        "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western",
    ]

    EMOTION_GENRE_MAP: Dict[str, List[str]] = {
        "happy":   ["Comedy", "Adventure", "Family", "Animation", "Fantasy", "Romance"],
        "sad":     ["Drama", "Music", "Documentary"],
        "excited": ["Action", "Science Fiction", "Thriller", "War"],
        "relaxed": ["Romance", "Family", "Comedy"],
        "scared":  ["Horror", "Mystery", "Thriller"],
    }

    def __init__(self) -> None:
        base = os.path.dirname(__file__)
        pre_path = os.path.join(base, "data", "movies_preprocessed.csv")
        meta_path = os.path.join(base, "data", "movies_metadata.csv")

        df_pre = pd.read_csv(pre_path)
        df_meta = pd.read_csv(meta_path)

        # Clean titles
        df_pre["title_clean"] = df_pre["title"].astype(str).str.strip().str.lower()
        df_meta["title_clean"] = df_meta["title"].astype(str).str.strip().str.lower()

        # Only need these columns
        df_meta = df_meta[["title_clean", "poster_path", "overview"]].rename(
            columns={"overview": "overview_meta"}
        )

        # Exact title merge
        df = df_pre.merge(df_meta, on="title_clean", how="left")

        # Pick best overview
        df["overview"] = df.apply(
            lambda r: r["overview"]
            if isinstance(r["overview"], str) and r["overview"].strip() != ""
            else r["overview_meta"],
            axis=1,
        ).fillna("No description available.")

        df = df.drop(columns=["overview_meta"], errors="ignore")

        # Clean metadata poster path
        def clean_poster(p: Any) -> Optional[str]:
            p = str(p)
            if p.lower() in ["", "0", "nan", "none", "null"]:
                return None
            if not p.startswith("/"):
                return None
            return p

        df["poster_path"] = df["poster_path"].apply(clean_poster)

        self.movies = df
        self._add_genre_label_column()

        # Cache for TMDB poster lookups
        self.poster_cache: Dict[str, Optional[str]] = {}

    def _add_genre_label_column(self):
        def infer_genres(row):
            genres = [g for g in self.GENRE_COLUMNS if row.get(g, 0) == 1]
            return ", ".join(genres) if genres else "Unknown"
        self.movies["genre_label"] = self.movies.apply(infer_genres, axis=1)

    # ------------------------------
    # TMDB SEARCH for missing posters
    # ------------------------------
    def _fetch_tmdb_poster_url(self, title: str) -> Optional[str]:
        if not TMDB_API_KEY:
            return None

        try:
            resp = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params={
                    "api_key": TMDB_API_KEY,
                    "query": title,
                    "include_adult": False,
                    "language": "en-US",
                },
                timeout=5,
            )
            data = resp.json()
            results = data.get("results") or []
            if not results:
                return None

            # Pick most popular result with poster
            results = sorted(results, key=lambda r: r.get("popularity", 0), reverse=True)
            for r in results:
                poster_path = r.get("poster_path")
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"

            return None

        except Exception:
            return None

    def _attach_posters(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        posters = []

        for _, row in df.iterrows():
            title = row["title"]
            title_clean = row["title_clean"]
            poster_path = row["poster_path"]

            # 1. use cached
            if title_clean in self.poster_cache:
                posters.append(self.poster_cache[title_clean])
                continue

            # 2. use local poster_path
            if isinstance(poster_path, str):
                url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                self.poster_cache[title_clean] = url
                posters.append(url)
                continue

            # 3. use TMDB search
            tmdb_url = self._fetch_tmdb_poster_url(title)
            posters.append(tmdb_url)
            self.poster_cache[title_clean] = tmdb_url

        df["poster_url"] = posters
        return df

    # ------------------------------
    # Emotion Recommendation Only
    # ------------------------------
    def recommend_by_emotion(self, emotion: str, n: int = 10) -> pd.DataFrame:
        emotion = emotion.lower().strip()
        genres = self.EMOTION_GENRE_MAP.get(emotion, ["Drama"])

        mask = pd.Series(False, index=self.movies.index)
        for g in genres:
            if g in self.movies:
                mask |= self.movies[g] == 1

        subset = self.movies[mask]
        if subset.empty:
            return subset

        sample = subset.sample(min(n, len(subset)))
        sample = self._attach_posters(sample)

        return sample[["title", "overview", "genre_label", "poster_url"]]
