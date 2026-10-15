"""Unit tests for prediction module and model lifecycle (src/predict.py and src/train.py)."""

from pathlib import Path
import pytest
from src.predict import SpamPredictor, predict_message
from src.train import train_models


def test_empty_message_raises_value_error():
    """Verify that empty string or whitespace input raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        predict_message("")
    assert "Input message cannot be empty" in str(exc_info.value)

    with pytest.raises(ValueError):
        predict_message("    \n\t  ")

    with pytest.raises(ValueError):
        predict_message(None)


def test_missing_model_files_raise_file_not_found(tmp_path: Path):
    """Verify SpamPredictor raises FileNotFoundError if model artifacts are missing."""
    fake_model = tmp_path / "non_existent_model.joblib"
    fake_vect = tmp_path / "non_existent_vect.joblib"

    with pytest.raises(FileNotFoundError) as exc_info:
        SpamPredictor(model_path=fake_model, vectorizer_path=fake_vect)
    assert "Model file not found" in str(exc_info.value)


def test_prediction_output_format():
    """Verify that predictions return a well-formed dictionary with proper types."""
    res = predict_message("Hey, are we still meeting today?")

    assert isinstance(res, dict)
    assert "is_spam" in res
    assert "label" in res
    assert "category" in res
    assert "spam_probability" in res
    assert "ham_probability" in res
    assert "confidence" in res
    assert "cleaned_text" in res

    assert isinstance(res["is_spam"], bool)
    assert res["label"] in (0, 1)
    assert 0.0 <= res["spam_probability"] <= 1.0
    assert 0.0 <= res["ham_probability"] <= 1.0
    assert 0.5 <= res["confidence"] <= 1.0
    assert abs((res["spam_probability"] + res["ham_probability"]) - 1.0) < 0.01


def test_end_to_end_training_on_synthetic_fixture(tmp_path: Path):
    """Verify train_models runs cleanly on a small synthetic CSV fixture without real Kaggle data."""
    sample_dataset = tmp_path / "synthetic_spam.csv"
    # Create balanced synthetic samples
    rows = [
        "v1,v2",
        'ham,"Hello friend how are you doing today?"',
        'ham,"Call me back when you get a chance"',
        'ham,"Lunch was really good see you tomorrow"',
        'ham,"Can you please share the homework assignments?"',
        'ham,"Meeting at 4pm in room 201"',
        'spam,"WIN $5000 now text YES to 88888"',
        'spam,"Free luxury holiday claim your voucher today"',
        'spam,"Congratulations you won £1000 cash reward"',
        'spam,"URGENT verify account at http://fakebank.com now"',
        'spam,"Exclusive deal get rich quick call 99999"',
    ]
    sample_dataset.write_text("\n".join(rows), encoding="utf-8")

    custom_models_dir = tmp_path / "models"
    custom_reports_dir = tmp_path / "reports"

    result = train_models(
        dataset_path=sample_dataset,
        models_dir=custom_models_dir,
        reports_dir=custom_reports_dir,
        test_size=0.3,
        random_state=42,
        min_df=1,
        verbose=False
    )

    assert result["best_model_name"] in ["Multinomial Naive Bayes", "Logistic Regression"]
    assert Path(result["model_path"]).is_file()
    assert Path(result["vectorizer_path"]).is_file()
    assert Path(result["confusion_matrix_path"]).is_file()

    # Test that predictor can load this synthetic model
    predictor = SpamPredictor(
        model_path=result["model_path"],
        vectorizer_path=result["vectorizer_path"]
    )
    pred_res = predictor.predict("Free gift voucher claim now")
    assert isinstance(pred_res["is_spam"], bool)
