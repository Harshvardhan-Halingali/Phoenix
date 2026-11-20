import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.modules.text_detector import text_detector, TextDetector

def predict(text: str) -> dict:
    """
    Classifies text as 'Human' or 'AI' using the fine-tuned RoBERTa-base checkpoint.
    """
    res = text_detector.predict(text)
    return {
        "label": res["label"],
        "confidence": res["confidence"],
        "probs": res["probs"],
        "human_score": res["human_score"],
        "ai_score": res["ai_score"],
        "prediction": res["prediction"],
        "truncated": res["truncated"]
    }

def predict_batch(texts: list, batch_size: int = 16) -> list:
    """
    Batch prediction for multiple text samples.
    """
    return text_detector.predict_batch(texts, batch_size=batch_size)

if __name__ == "__main__":
    sample = "The proliferation of artificial intelligence in contemporary software engineering marks a paradigm shift."
    print("Testing detector.py with sample text:")
    result = predict(sample)
    print(result)
