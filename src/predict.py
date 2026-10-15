"""Inference and prediction module for Spam Message Classification.

Loads pre-trained model and fitted vectorizer from models/,
applies exact training-time text preprocessing,
and provides single-message and batch classification.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib

# Ensure project root is on sys.path for standalone script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import clean_text

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DEFAULT_VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"


class SpamPredictor:
    """Encapsulates model loading and inference for SMS spam classification."""

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        vectorizer_path: Union[str, Path] = DEFAULT_VECTORIZER_PATH
    ):
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.model = None
        self.vectorizer = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load trained model and fitted vectorizer from disk."""
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. "
                f"Please run 'python src/train.py' first to train and save the model."
            )
        if not self.vectorizer_path.is_file():
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found at '{self.vectorizer_path}'. "
                f"Please run 'python src/train.py' first to generate feature artifacts."
            )

        try:
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model artifacts: {e}") from e

    def predict(self, message: str) -> Dict[str, Any]:
        """Classify a single SMS text message.

        Parameters
        ----------
        message : str
            The raw SMS message text.

        Returns
        -------
        dict
            Dictionary containing:
            - 'is_spam': bool
            - 'label': int (0 for Ham, 1 for Spam)
            - 'category': str ('Ham (Not Spam)' or 'Spam')
            - 'spam_probability': float (0.0 to 1.0)
            - 'ham_probability': float (0.0 to 1.0)
            - 'confidence': float (highest probability, 0.0 to 1.0)
            - 'cleaned_text': str
            - 'raw_message': str

        Raises
        ------
        ValueError
            If message is empty, None, or only whitespace.
        """
        if message is None or not str(message).strip():
            raise ValueError("Input message cannot be empty. Please enter a valid SMS message.")

        raw_str = str(message)
        # Apply EXACT same preprocessing as training time
        cleaned = clean_text(raw_str, remove_stopwords=False, preserve_negations=True)

        if not cleaned.strip():
            # In rare cases where message consisted purely of stripped punctuation
            cleaned = raw_str.strip().lower()

        # Transform using fitted TF-IDF vectorizer
        features = self.vectorizer.transform([cleaned])

        # Model prediction
        pred_label = int(self.model.predict(features)[0])

        # Estimated probabilities if model supports predict_proba
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
            ham_prob = float(probs[0])
            spam_prob = float(probs[1])
        else:
            # Fallback if model only supports decision function
            spam_prob = 1.0 if pred_label == 1 else 0.0
            ham_prob = 1.0 - spam_prob

        confidence = max(ham_prob, spam_prob)
        category = "Spam" if pred_label == 1 else "Ham (Not Spam)"

        return {
            "is_spam": bool(pred_label == 1),
            "label": pred_label,
            "category": category,
            "spam_probability": round(spam_prob, 4),
            "ham_probability": round(ham_prob, 4),
            "confidence": round(confidence, 4),
            "cleaned_text": cleaned,
            "raw_message": raw_str,
        }


# Module-level cached predictor instance for fast UI & API reuse
_CACHED_PREDICTOR: Optional[SpamPredictor] = None


def get_predictor(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    vectorizer_path: Union[str, Path] = DEFAULT_VECTORIZER_PATH,
    force_reload: bool = False
) -> SpamPredictor:
    """Retrieve or create the cached SpamPredictor instance."""
    global _CACHED_PREDICTOR
    if _CACHED_PREDICTOR is None or force_reload:
        _CACHED_PREDICTOR = SpamPredictor(model_path=model_path, vectorizer_path=vectorizer_path)
    return _CACHED_PREDICTOR


def predict_message(
    message: str,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    vectorizer_path: Union[str, Path] = DEFAULT_VECTORIZER_PATH
) -> Dict[str, Any]:
    """Convenience helper to predict a single message."""
    predictor = get_predictor(model_path=model_path, vectorizer_path=vectorizer_path)
    return predictor.predict(message)


if __name__ == "__main__":
    # Test cases when executed as a script
    test_samples = [
        "Hey Mom, I'm heading home now. See you soon!",
        "WINNER!! You have won a guaranteed £1000 prize or cash reward. Call 08452810075 now!",
        "Can you send me the lecture notes when you get time?",
        "URGENT: Your bank account has been flagged. Verify details at http://security-update-fake.com"
    ]

    print("Running inference test on sample messages:\n")
    for text in test_samples:
        res = predict_message(text)
        print(f"Message : '{text}'")
        print(f"Result  : {res['category']} (Spam Probability: {res['spam_probability']*100:.1f}%, Confidence: {res['confidence']*100:.1f}%)")
        print("-" * 70)
