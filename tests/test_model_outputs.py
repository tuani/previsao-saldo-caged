import pandas as pd

from ml_pipeline import TARGET_COLUMN, evaluate_model, predict_test, save_predictions_csv


class DummyModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, _x):
        return self.predictions


def test_predict_test_adds_target_column() -> None:
    x_test = pd.DataFrame({"feature": [1, 2, 3]})
    model = DummyModel(predictions=[-1, 1, -1])

    predicted = predict_test(model, x_test)

    assert predicted.columns.tolist() == ["feature", TARGET_COLUMN]
    assert predicted[TARGET_COLUMN].tolist() == [-1, 1, -1]


def test_evaluate_model_returns_expected_metrics_structure() -> None:
    x_valid = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y_valid = pd.Series([-1, 1, -1, 1])
    model = DummyModel(predictions=[-1, 1, 1, 1])

    metrics = evaluate_model(model, x_valid, y_valid)

    assert set(metrics.keys()) == {"f1_binary", "f1_macro", "accuracy", "report_frame"}
    assert 0 <= metrics["f1_binary"] <= 1
    assert 0 <= metrics["f1_macro"] <= 1
    assert 0 <= metrics["accuracy"] <= 1
    assert list(metrics["report_frame"].index) == ["-1", "1", "macro avg", "weighted avg"]


def test_save_predictions_csv_preserves_output_column(tmp_path) -> None:
    dataframe = pd.DataFrame({"coluna": [1, 2], TARGET_COLUMN: [-1, 1]})

    output_path = tmp_path / "saida.csv"
    save_predictions_csv(dataframe, output_path)
    loaded = pd.read_csv(output_path)

    assert loaded.columns.tolist() == ["coluna", TARGET_COLUMN]
    assert loaded[TARGET_COLUMN].tolist() == [-1, 1]
