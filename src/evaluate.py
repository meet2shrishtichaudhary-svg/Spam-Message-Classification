"""Model evaluation module for Spam Message Classification.

Computes accuracy, precision, recall, F1-score, confusion matrix,
and exports publication-quality metrics and confusion matrix plots to reports/.
Positive class is defined strictly as Spam = 1.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for scripts and headless servers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    model_name: str = "Model",
    pos_label: int = 1
) -> Dict[str, Any]:
    """Compute comprehensive classification evaluation metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels (0 for Ham, 1 for Spam).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels.
    model_name : str, default="Model"
        Name of the classifier for reporting.
    pos_label : int, default=1
        Positive class indicator. 1 corresponds to Spam.

    Returns
    -------
    dict
        Dictionary containing calculated metrics, confusion matrix values,
        and classification report breakdown.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, pos_label=pos_label, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, pos_label=pos_label, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=pos_label, zero_division=0))

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    report_dict = classification_report(
        y_true_arr,
        y_pred_arr,
        target_names=["Ham (0)", "Spam (1)"],
        output_dict=True,
        zero_division=0
    )

    report_str = classification_report(
        y_true_arr,
        y_pred_arr,
        target_names=["Ham (0)", "Spam (1)"],
        zero_division=0
    )

    return {
        "model_name": model_name,
        "positive_class": "Spam (1)",
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "true_negatives_tn": int(tn),
            "false_positives_fp": int(fp),
            "false_negatives_fn": int(fn),
            "true_positives_tp": int(tp),
        },
        "classification_report": report_dict,
        "classification_report_text": report_str,
    }


def print_evaluation_summary(metrics: Dict[str, Any]) -> None:
    """Print an easy-to-read ASCII evaluation summary to standard output."""
    cm = metrics["confusion_matrix"]
    print("=" * 65)
    print(f"       MODEL EVALUATION: {metrics['model_name'].upper()}")
    print("=" * 65)
    print(f"Overall Accuracy       : {metrics['accuracy'] * 100:.2f}%")
    print(f"Spam Precision (PPV)   : {metrics['precision'] * 100:.2f}% (Spam = 1)")
    print(f"Spam Recall (Sensitivity): {metrics['recall'] * 100:.2f}% (Spam = 1)")
    print(f"Spam F1-Score (Harmonic) : {metrics['f1_score'] * 100:.2f}%")
    print("-" * 65)
    print("Confusion Matrix Breakdown:")
    print(f"  • True Negatives  (TN): {cm['true_negatives_tn']:,} (Ham correctly identified as Ham)")
    print(f"  • False Positives (FP): {cm['false_positives_fp']:,} (Ham mistakenly flagged as Spam)")
    print(f"  • False Negatives (FN): {cm['false_negatives_fn']:,} (Spam missed, delivered to inbox)")
    print(f"  • True Positives  (TP): {cm['true_positives_tp']:,} (Spam correctly flagged)")
    print("-" * 65)
    print("Detailed Classification Report:")
    print(metrics["classification_report_text"])
    print("=" * 65)


def save_metrics(
    metrics: Dict[str, Any],
    save_path: Union[str, Path] = "reports/metrics.json"
) -> Path:
    """Save metrics dictionary to a JSON file."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path


def plot_confusion_matrix(
    cm_matrix: Union[List[List[int]], np.ndarray],
    model_name: str = "Classifier",
    save_path: Union[str, Path] = "reports/confusion_matrix.png"
) -> Path:
    """Generate and save an aesthetically styled confusion matrix heatmap.

    Parameters
    ----------
    cm_matrix : 2x2 list or ndarray
        Confusion matrix [[TN, FP], [FN, TP]].
    model_name : str
        Model title for the plot.
    save_path : str or Path
        Destination filepath for the PNG plot.

    Returns
    -------
    Path
        Path to the saved PNG image.
    """
    cm = np.array(cm_matrix)
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    # Format annotations showing count and percentage
    total = np.sum(cm)
    labels = np.array([
        [f"TN\n{cm[0, 0]}\n({cm[0, 0]/total:.1%})", f"FP\n{cm[0, 1]}\n({cm[0, 1]/total:.1%})"],
        [f"FN\n{cm[1, 0]}\n({cm[1, 0]/total:.1%})", f"TP\n{cm[1, 1]}\n({cm[1, 1]/total:.1%})"]
    ])

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        square=True,
        xticklabels=["Ham (0)", "Spam (1)"],
        yticklabels=["Ham (0)", "Spam (1)"],
        ax=ax,
        annot_kws={"fontsize": 11, "fontweight": "bold"}
    )

    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Actual Ground Truth Label", fontsize=11, fontweight="bold", labelpad=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    # Self-test with sample predictions
    y_test_sample = [0, 0, 0, 1, 1, 0, 1, 0]
    y_pred_sample = [0, 0, 1, 1, 1, 0, 0, 0]
    res = calculate_metrics(y_test_sample, y_pred_sample, model_name="TestClassifier")
    print_evaluation_summary(res)
    plot_path = plot_confusion_matrix(res["confusion_matrix"]["matrix"], model_name="TestClassifier")
    print(f"Confusion matrix plot saved to: {plot_path}")
