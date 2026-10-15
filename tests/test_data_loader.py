"""Unit tests for dataset loading and validation module (src/data_loader.py)."""

import pytest
import pandas as pd
from pathlib import Path
from src.data_loader import load_dataset, get_dataset_summary


def test_missing_dataset_raises_file_not_found(tmp_path: Path):
    """Ensure FileNotFoundError is raised with clear message when file is missing."""
    non_existent = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_dataset(filepath=non_existent, verbose=False)
    assert "Dataset not found" in str(exc_info.value)


def test_kaggle_format_normalization(tmp_path: Path):
    """Verify Kaggle v1/v2 columns and Unnamed extra columns are cleaned and normalized."""
    sample_csv = tmp_path / "sample_kaggle.csv"
    sample_csv.write_text(
        "v1,v2,Unnamed: 2,Unnamed: 3\n"
        'ham,"Hello friend, are you free tonight?",,\n'
        'spam,"WINNER! Claim your $1000 prize now by texting YES",,\n',
        encoding="latin-1",
    )

    df = load_dataset(filepath=sample_csv, verbose=False)

    assert list(df.columns) == ["label", "message"]
    assert len(df) == 2
    assert df.iloc[0]["label"] == 0
    assert df.iloc[0]["message"] == "Hello friend, are you free tonight?"
    assert df.iloc[1]["label"] == 1
    assert df.iloc[1]["message"] == "WINNER! Claim your $1000 prize now by texting YES"


def test_empty_and_whitespace_removal(tmp_path: Path):
    """Verify that empty, NaN, and whitespace-only messages are removed."""
    sample_csv = tmp_path / "sample_empty.csv"
    sample_csv.write_text(
        "label,message\n"
        "ham,Valid message here\n"
        "ham,   \n"
        'spam,""\n',
        encoding="utf-8",
    )

    df = load_dataset(filepath=sample_csv, verbose=False)
    assert len(df) == 1
    assert df.iloc[0]["message"] == "Valid message here"


def test_duplicate_removal_keeps_first(tmp_path: Path):
    """Verify that duplicate messages are deduplicated keeping the first occurrence."""
    sample_csv = tmp_path / "sample_dupes.csv"
    sample_csv.write_text(
        "v1,v2\n"
        'spam,"Claim your free reward now!"\n'
        'spam,"Claim your free reward now!"\n'
        'ham,"See you later"\n'
        'spam,"Claim your free reward now!"\n',
        encoding="utf-8",
    )

    df = load_dataset(filepath=sample_csv, verbose=False)
    assert len(df) == 2
    assert df[df["message"] == "Claim your free reward now!"].shape[0] == 1


def test_invalid_columns_raise_value_error(tmp_path: Path):
    """Verify that datasets lacking label or message columns raise ValueError."""
    sample_csv = tmp_path / "sample_invalid.csv"
    sample_csv.write_text(
        "col_a,col_b\n"
        "data1,data2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_dataset(filepath=sample_csv, verbose=False)
    assert "Unsupported dataset format" in str(exc_info.value)


def test_invalid_label_value_raises_value_error(tmp_path: Path):
    """Verify that unexpected label strings raise ValueError."""
    sample_csv = tmp_path / "sample_bad_label.csv"
    sample_csv.write_text(
        "v1,v2\n"
        "unknown_category,Hello there\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_dataset(filepath=sample_csv, verbose=False)
    assert "Unrecognized label value" in str(exc_info.value)


def test_dataset_summary_calculation():
    """Verify get_dataset_summary calculates counts and percentages accurately."""
    df = pd.DataFrame({
        "label": [0, 0, 0, 1],
        "message": ["m1", "m2", "m3", "m4"],
    })
    summary = get_dataset_summary(df)
    assert summary["total_messages"] == 4
    assert summary["ham_count"] == 3
    assert summary["spam_count"] == 1
    assert summary["ham_percentage"] == 75.0
    assert summary["spam_percentage"] == 25.0
