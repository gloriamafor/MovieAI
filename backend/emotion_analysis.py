from functools import lru_cache
from typing import Dict, List, Tuple

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

# Raw model emotion -> mood (used only when the user does NOT state a vibe)
EMOTION_TO_MOOD: Dict[str, str] = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "intense",
    "fear": "scary",
    "surprise": "thrilling",
    "disgust": "intense",
    "neutral": "cozy",
}

# Desired-vibe keywords -> mood. This captures what the user WANTS to watch,
# which is often different from (even the opposite of) how they currently feel.
MOOD_KEYWORDS: Dict[str, List[str]] = {
    "funny": ["funny", "comedy", "hilarious", "laugh", "humor", "humour",
              "silly", "goofy", "comedic"],
    "happy": ["happy", "feel good", "feel-good", "feelgood", "uplifting",
              "cheerful", "wholesome", "upbeat", "joyful", "lighthearted",
              "light-hearted", "fun", "heartwarming"],
    "cozy": ["cozy", "cosy", "comforting", "comfort", "chill", "relaxing",
             "relax", "easy", "gentle", "warm", "light", "soothing", "calm",
             "low key", "low-key"],
    "romantic": ["romantic", "romance", "love story", "date night",
                 "rom com", "rom-com", "romcom"],
    "sad": ["sad", "cry", "tearjerker", "tear jerker", "emotional",
            "melancholy", "moving", "somber", "sombre", "heartbreak"],
    "thrilling": ["thrilling", "thriller", "exciting", "suspense",
                  "suspenseful", "adventure", "adventurous", "adrenaline",
                  "edge of my seat", "mind-bending", "mind bending"],
    "intense": ["intense", "action", "action packed", "action-packed",
                "gritty", "dark", "high stakes", "high-stakes", "gripping"],
    "scary": ["scary", "horror", "spooky", "creepy", "frightening",
              "terrifying", "haunted", "slasher", "frightful"],
}

NEGATORS = {"not", "no", "nothing", "dont", "don't", "avoid",
            "without", "except", "nope", "never"}


def _detect_requested_mood(text: str) -> Tuple[str, str]:
    """
    Look for an explicitly requested vibe in the text.
    Returns (mood, matched_keyword), or ("", "") if none found.
    A keyword is ignored if a negator sits within the 3 words before it,
    so "nothing scary" or "no horror" don't count as requests for those.
    """
    t = " " + text.lower() + " "
    scores: Dict[str, int] = {}
    hit: Dict[str, str] = {}

    for mood, words in MOOD_KEYWORDS.items():
        for w in words:
            start = 0
            while True:
                idx = t.find(w, start)
                if idx == -1:
                    break
                before_words = t[:idx].replace(",", " ").split()[-3:]
                negated = any(n in NEGATORS for n in before_words)
                if not negated:
                    scores[mood] = scores.get(mood, 0) + 1
                    hit.setdefault(mood, w)
                start = idx + len(w)

    if not scores:
        return "", ""
    best = max(scores, key=scores.get)
    return best, hit[best]


def detect_emotion_and_mood(text: str) -> Dict[str, str]:
    """
    Decide which mood to use for recommendations, in two layers:

      1. If the user explicitly asks for a vibe ("something funny",
         "nothing scary"), use that requested mood. This is what the user
         WANTS, which is what should drive the picks.
      2. Otherwise, detect the emotion in their text with the transformer
         model and map that emotion to a mood.

    Returns raw_emotion and score (always, so the UI can show what the model
    detected), the chosen mood, the source of that mood, and any matched
    keyword.
    """
    if not text.strip():
        return {"raw_emotion": "neutral", "score": "0.0", "mood": "cozy",
                "source": "default", "matched": ""}

    # Layer 1: explicit request
    requested, phrase = _detect_requested_mood(text)

    # Layer 2: emotion model (run it regardless, so we can display it)
    pipe = _get_emotion_pipeline()
    result = pipe(text)[0]
    raw = result["label"].lower()
    score = float(result["score"])
    emotion_mood = EMOTION_TO_MOOD.get(raw, "cozy")

    if requested:
        mood, source = requested, "requested"
    else:
        mood, source = emotion_mood, "emotion"

    return {"raw_emotion": raw, "score": f"{score:.3f}", "mood": mood,
            "source": source, "matched": phrase}