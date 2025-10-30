import pandas as pd
import random

class MovieRecommender:
    def __init__(self, data_path="backend/data/movies.csv"):
        self.movies = pd.read_csv(data_path)

    def recommend_by_emotion(self, emotion):
        emotion_map = {
            "happy": ["Comedy", "Adventure", "Romance"],
            "sad": ["Drama", "Inspirational"],
            "excited": ["Action", "Sci-Fi", "Thriller"],
            "relaxed": ["Romance", "Slice of Life"],
            "scared": ["Horror", "Mystery"]
        }
        genres = emotion_map.get(emotion.lower(), ["Drama"])
        filtered = self.movies[self.movies['genre'].isin(genres)]
        return filtered.sample(5) if len(filtered) > 5 else filtered