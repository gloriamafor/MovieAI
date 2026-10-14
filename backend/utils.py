import os
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
    """
    if not OPENAI_API_KEY:
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
