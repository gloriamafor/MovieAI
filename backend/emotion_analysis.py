from functools import lru_cache
from typing import Dict

from transformers import pipeline


@lru_cache(maxsize=1)
def _get_emotion_pipeline():
    """
    Load the HuggingFace emotion model once and cache it.
    """
    return pipeline(
        "sentiment-analysis",
        model="j-hartmann/emotion-english-distilroberta-base",
    )


# Map raw emotion labels -> our mood labels
EMOTION_TO_MOOD: Dict[str, str] = {
    "joy": "happy",
    "anger": "intense",
    "sadness": "sad",
    "fear": "scary",
    "neutral": "cozy",
    "surprise": "thrilling",
    "disgust": "intense",
    # fallback if your model returns others
}


def detect_emotion_and_mood(text: str) -> Dict[str, str]:
    """
    Run emotion analysis and map it to a mood string we can use
    in MovieRecommender.discover_by_mood.
    Returns:
      {
        "raw_emotion": "joy",
        "score": "0.98",
        "mood": "happy"
      }
    """
    if not text.strip():
        return {
            "raw_emotion": "neutral",
            "score": "0.0",
            "mood": "cozy",
        }

    pipe = _get_emotion_pipeline()
    result = pipe(text)[0]
    raw = result["label"].lower()
    score = float(result["score"])

    mood = EMOTION_TO_MOOD.get(raw, "cozy")

    return {
        "raw_emotion": raw,
        "score": f"{score:.3f}",
        "mood": mood,
    }
