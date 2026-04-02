import pandas as pd
import pytest

from ml_pipeline import (
    ALLOWED_MONTHS,
    TARGET_COLUMN,
    align_feature_sets,
    filter_allowed_months,
    split_features_target,
    summarize_dataset,
)


def test_filter_allowed_months_removes_unexpected_competencies() -> None:
    dataframe = pd.DataFrame({"competenciadec": [202401, 202601, 202512]})

    filtered = filter_allowed_months(dataframe)

    assert filtered["competenciadec"].tolist() == [202401, 202512]


def test_filter_allowed_months_raises_without_competenciadec() -> None:
    dataframe = pd.DataFrame({"outra_coluna": [1, 2, 3]})

    with pytest.raises(ValueError, match="competenciadec"):
        filter_allowed_months(dataframe)


def test_align_feature_sets_removes_constant_and_train_only_columns() -> None:
    x_train = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "constante": [0, 0, 0],
            "apenas_treino": [7, 8, 9],
        }
    )
    x_test = pd.DataFrame({"a": [5, 6]})

    aligned_train, aligned_test, removed = align_feature_sets(x_train, x_test)

    assert aligned_train.columns.tolist() == ["a"]
    assert aligned_test.columns.tolist() == ["a"]
    assert "constante" in removed
    assert "apenas_treino" in removed


def test_split_features_target_filters_invalid_target_values() -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40],
            TARGET_COLUMN: [-1, 1, 0, 2],
        }
    )

    x, y = split_features_target(dataframe)

    assert x.columns.tolist() == ["feature"]
    assert x["feature"].tolist() == [10, 20]
    assert y.tolist() == [-1, 1]


def test_split_features_target_raises_without_target() -> None:
    dataframe = pd.DataFrame({"feature": [1, 2]})

    with pytest.raises(ValueError, match=TARGET_COLUMN):
        split_features_target(dataframe)


def test_summarize_dataset_returns_shapes_and_distributions() -> None:
    month = next(iter(ALLOWED_MONTHS))
    train_df = pd.DataFrame(
        {
            TARGET_COLUMN: [-1, 1, -1],
            "competenciadec": [month, month, 202512],
        }
    )
    test_df = pd.DataFrame({"competenciadec": [202601, 202601]})

    summary = summarize_dataset(train_df, test_df)

    assert summary["train_shape"] == (3, 2)
    assert summary["test_shape"] == (2, 1)
    assert summary["target_distribution"] == {-1: 2, 1: 1}
    assert isinstance(summary["competenciadec_distribution"], dict)
