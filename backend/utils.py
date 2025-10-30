import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# --- TMDB API: Fetch Movies Based on Genre or Keywords ---
def fetch_movies_by_genres(genres):
    genre_query = ",".join(genres)
    url = f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&with_genres={genre_query}&sort_by=popularity.desc"
    response = requests.get(url)
    data = response.json()
    return data.get("results", [])

def fetch_genres():
    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}&language=en-US"
    response = requests.get(url)
    return {g["name"].lower(): g["id"] for g in response.json().get("genres", [])}

# --- DALL·E Poster Generation ---
def generate_ai_poster(title, genre, overview):
    client = OpenAI()
    prompt = f"Movie poster for a {genre} film titled '{title}'. Theme: {overview}."
    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="512x768"
    )
    return result.data[0].url