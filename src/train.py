"""Model training, comparison, and persistence module for Spam Message Classification.

Workflow:
1. Loads cleaned dataset via src.data_loader
2. Applies text preprocessing via src.preprocess
3. Performs stratified 80/20 train-test split
4. Fits TfidfVectorizer STRICTLY on training split (no data leakage)
5. Trains and compares Multinomial Naive Bayes and Logistic Regression
6. Evaluates both models on held-out test set
7. Selects best model using documented rule prioritizing Spam F1-Score
8. Persists selected model, fitted vectorizer, and evaluation reports
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
from pathlib import Path

# Ensure project root is on sys.path for standalone script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_dataset
from src.preprocess import clean_text
from src.evaluate import calculate_metrics, plot_confusion_matrix, print_evaluation_summary, save_metrics


def train_models(
    dataset_path: Union[str, Path] = "dataset/spam.csv",
    models_dir: Union[str, Path] = "models",
    reports_dir: Union[str, Path] = "reports",
    test_size: float = 0.2,
    random_state: int = 42,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    remove_stopwords: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """Train, compare, and save spam classification models.

    Parameters
    ----------
    dataset_path : str or Path
        Location of the input dataset file.
    models_dir : str or Path
        Directory where serialized model artifacts will be saved.
    reports_dir : str or Path
        Directory where evaluation reports and plots will be saved.
    test_size : float, default=0.2
        Fraction of the dataset reserved for held-out testing (stratified).
    random_state : int, default=42
        Reproducible random seed.
    ngram_range : tuple of (int, int), default=(1, 2)
        Unigram and bigram token features.
    min_df : int, default=2
        Minimum document frequency threshold for vocabulary inclusion.
    remove_stopwords : bool, default=False
        Whether to filter stopwords during preprocessing.
    verbose : bool, default=True
        Whether to print status logs and metric summaries.

    Returns
    -------
    dict
        Dictionary containing comparison results, best model name, and file paths.
    """
    models_path = Path(models_dir)
    reports_path = Path(reports_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate dataset
    if verbose:
        print("[1/5] Loading and validating dataset...")
    df = load_dataset(dataset_path, verbose=verbose)

    # 2. Preprocess text deterministically
    if verbose:
        print("[2/5] Preprocessing text messages...")
    cleaned_messages = [
        clean_text(msg, remove_stopwords=remove_stopwords, preserve_negations=True)
        for msg in df["message"]
    ]
    y = df["label"].values

    # Adjust min_df dynamically if dataset is extremely small (e.g. in synthetic unit tests)
    effective_min_df = min_df if len(cleaned_messages) >= 50 else 1

    # 3. Stratified Train/Test Split (80% train, 20% test)
    # Stratification ensures both train and test partitions maintain identical 87/13 class ratios
    if verbose:
        print(f"[3/5] Performing stratified {int((1-test_size)*100)}/{int(test_size*100)} train/test split...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        cleaned_messages,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # 4. Feature Extraction using TF-IDF (Leakage-Safe: Fit ONLY on training data)
    if verbose:
        print(f"      Fitting TfidfVectorizer on training set (ngram_range={ngram_range}, min_df={effective_min_df})...")
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=effective_min_df,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)

    if verbose:
        print(f"      TF-IDF Vocabulary Size: {len(vectorizer.vocabulary_):,} features")

    # 5. Train Candidates: Multinomial Naive Bayes and Logistic Regression
    if verbose:
        print("[4/5] Training candidate classifiers...")

    candidate_models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.2),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            solver="liblinear"
        )
    }

    comparison_results: Dict[str, Any] = {}
    fitted_candidates = {}

    for name, model in candidate_models.items():
        if verbose:
            print(f"      Training {name}...")
        model.fit(X_train_tfidf, y_train)
        fitted_candidates[name] = model

        # Evaluate on the exact same held-out test split
        y_pred = model.predict(X_test_tfidf)
        metrics = calculate_metrics(y_test, y_pred, model_name=name)
        comparison_results[name] = metrics

        if verbose:
            print_evaluation_summary(metrics)

    # 6. Model Selection Rule:
    # Prioritize Spam-class F1-score (harmonic mean of Precision and Recall on the minority Spam class).
    # If F1-scores are tied or differ by less than 0.005, select the candidate with higher Precision
    # to protect users from legitimate messages mistakenly classified as spam (False Positives).
    if verbose:
        print("[5/5] Selecting champion model and persisting artifacts...")

    best_model_name = None
    best_f1 = -1.0
    best_prec = -1.0

    for name, metrics in comparison_results.items():
        curr_f1 = metrics["f1_score"]
        curr_prec = metrics["precision"]

        if curr_f1 > best_f1:
            best_model_name = name
            best_f1 = curr_f1
            best_prec = curr_prec
        elif abs(curr_f1 - best_f1) <= 0.005 and curr_prec > best_prec:
            # Tie-break preference: higher precision (fewer false positives)
            best_model_name = name
            best_f1 = curr_f1
            best_prec = curr_prec

    best_model = fitted_candidates[best_model_name]
    best_metrics = comparison_results[best_model_name]

    if verbose:
        print(f"      -> Champion Model: {best_model_name} (F1: {best_f1*100:.2f}%, Precision: {best_prec*100:.2f}%)")

    # 7. Persist Artifacts using Joblib
    model_save_path = models_path / "best_model.joblib"
    vectorizer_save_path = models_path / "tfidf_vectorizer.joblib"
    metadata_save_path = models_path / "model_metadata.json"

    joblib.dump(best_model, model_save_path)
    joblib.dump(vectorizer, vectorizer_save_path)

    metadata = {
        "model_name": best_model_name,
        "algorithm": best_model.__class__.__name__,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "ngram_range": list(ngram_range),
        "min_df": effective_min_df,
        "test_size": test_size,
        "random_state": random_state,
        "training_samples": len(X_train_raw),
        "test_samples": len(X_test_raw),
        "metrics": {
            "accuracy": best_metrics["accuracy"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "f1_score": best_metrics["f1_score"],
        }
    }
    with open(metadata_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 8. Save Reports & Confusion Matrix Plot
    comparison_save_path = reports_path / "model_comparison.json"
    metrics_save_path = reports_path / "metrics.json"
    cm_plot_save_path = reports_path / "confusion_matrix.png"

    with open(comparison_save_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    save_metrics(best_metrics, save_path=metrics_save_path)
    plot_confusion_matrix(
        best_metrics["confusion_matrix"]["matrix"],
        model_name=best_model_name,
        save_path=cm_plot_save_path
    )

    if verbose:
        print("\nArtifacts and Reports successfully saved:")
        print(f"  • Trained Model    : {model_save_path}")
        print(f"  • Fitted Vectorizer: {vectorizer_save_path}")
        print(f"  • Model Metadata   : {metadata_save_path}")
        print(f"  • Metrics JSON     : {metrics_save_path}")
        print(f"  • Comparison JSON  : {comparison_save_path}")
        print(f"  • Confusion Matrix : {cm_plot_save_path}")

    return {
        "best_model_name": best_model_name,
        "best_metrics": best_metrics,
        "comparison_results": comparison_results,
        "model_path": str(model_save_path),
        "vectorizer_path": str(vectorizer_save_path),
        "confusion_matrix_path": str(cm_plot_save_path),
    }


if __name__ == "__main__":
    train_models()
