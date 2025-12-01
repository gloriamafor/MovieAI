import os
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")


class MovieRecommender:
    """
    Simple TMDB client used by the frontend.

    Supports:
      - search_movies: text search
      - discover_by_country: movies by origin country
      - get_genres: TMDB genre list
      - discover_by_genre: movies by genre ID(s)
      - discover_by_mood: uses mood → genres mapping
      - get_movie_details: full detail for a movie ID
    """

    # Mood -> genres (TMDB genre names)
    MOOD_GENRES: Dict[str, List[str]] = {
        "happy": ["Comedy", "Family", "Romance"],
        "sad": ["Drama"],
        "thrilling": ["Action", "Thriller", "Mystery"],
        "romantic": ["Romance", "Drama"],
        "funny": ["Comedy"],
        "cozy": ["Animation", "Family"],
        "intense": ["Action", "Crime", "Thriller"],
        "scary": ["Horror", "Thriller"],
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or TMDB_API_KEY
        if not self.api_key:
            raise ValueError(
                "TMDB_API_KEY is not set. "
                "Set it in your environment or in a .env file."
            )

        self.base_url = "https://api.themoviedb.org/3"
        self.image_base = "https://image.tmdb.org/t/p/w500"

    # ------------------------
    # Internal helper
    # ------------------------
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if params is None:
            params = {}
        params["api_key"] = self.api_key

        url = f"{self.base_url}{path}"
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ------------------------
    # Text search
    # ------------------------
    def search_movies(
        self,
        query: str,
        year: Optional[int] = None,
        page: int = 1,
        include_adult: bool = False,
        language: str = "en-US",
    ) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        params: Dict[str, Any] = {
            "query": query,
            "page": page,
            "include_adult": include_adult,
            "language": language,
        }
        if year:
            params["year"] = year

        data = self._get("/search/movie", params=params)
        results = data.get("results", [])

        movies: List[Dict[str, Any]] = []
        for m in results:
            poster_path = m.get("poster_path")
            poster_url = f"{self.image_base}{poster_path}" if poster_path else None

            movies.append(
                {
                    "id": m.get("id"),
                    "title": m.get("title"),
                    "overview": m.get("overview"),
                    "release_date": m.get("release_date"),
                    "vote_average": m.get("vote_average"),
                    "vote_count": m.get("vote_count"),
                    "original_language": m.get("original_language"),
                    "poster_url": poster_url,
                }
            )
        return movies

    # ------------------------
    # Genres
    # ------------------------
    def get_genres(self) -> Dict[str, int]:
        """
        Returns dict like {"Action": 28, "Comedy": 35, ...}
        """
        data = self._get("/genre/movie/list", params={"language": "en-US"})
        genres = data.get("genres", [])
        return {g["name"]: g["id"] for g in genres}

    # ------------------------
    # Discover by country
    # ------------------------
    def discover_by_country(self, country_code: str) -> List[Dict[str, Any]]:
        params = {
            "with_origin_country": country_code.upper(),
            "sort_by": "popularity.desc",
        }
        data = self._get("/discover/movie", params=params)
        results = data.get("results", [])

        movies: List[Dict[str, Any]] = []
        for m in results:
            poster_path = m.get("poster_path")
            poster_url = f"{self.image_base}{poster_path}" if poster_path else None
            movies.append(
                {
                    "id": m.get("id"),
                    "title": m.get("title"),
                    "overview": m.get("overview"),
                    "release_date": m.get("release_date"),
                    "poster_url": poster_url,
                    "vote_average": m.get("vote_average"),
                }
            )
        return movies

    # ------------------------
    # Discover by genre
    # ------------------------
    def discover_by_genre(self, genre_ids: List[int]) -> List[Dict[str, Any]]:
        if not genre_ids:
            return []

        params = {
            "with_genres": ",".join(str(g) for g in genre_ids),
            "sort_by": "popularity.desc",
        }
        data = self._get("/discover/movie", params=params)
        results = data.get("results", [])

        movies: List[Dict[str, Any]] = []
        for m in results:
            poster_path = m.get("poster_path")
            poster_url = f"{self.image_base}{poster_path}" if poster_path else None
            movies.append(
                {
                    "id": m.get("id"),
                    "title": m.get("title"),
                    "overview": m.get("overview"),
                    "release_date": m.get("release_date"),
                    "poster_url": poster_url,
                    "vote_average": m.get("vote_average"),
                }
            )
        return movies

    # ------------------------
    # Discover by mood
    # ------------------------
    def discover_by_mood(self, mood: str) -> List[Dict[str, Any]]:
        mood = mood.lower()
        if mood not in self.MOOD_GENRES:
            return []

        genre_map = self.get_genres()
        genre_names = self.MOOD_GENRES[mood]
        genre_ids: List[int] = [
            genre_map[name]
            for name in genre_names
            if name in genre_map
        ]
        if not genre_ids:
            return []
        return self.discover_by_genre(genre_ids)

    # ------------------------
    # Full details (optional)
    # ------------------------
    def get_movie_details(self, movie_id: int, language: str = "en-US") -> Dict[str, Any]:
        data = self._get(f"/movie/{movie_id}", params={"language": language})

        poster_path = data.get("poster_path")
        backdrop_path = data.get("backdrop_path")

        data["poster_url"] = (
            f"{self.image_base}{poster_path}" if poster_path else None
        )
        data["backdrop_url"] = (
            f"{self.image_base}{backdrop_path}" if backdrop_path else None
        )
        return data
