from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

DATA_DIR = "data"
TRAIN_FILE = f"{DATA_DIR}/caged_curitiba_consolidado_train.csv"
TEST_FILE = f"{DATA_DIR}/caged_curitiba_consolidado_test.csv"
OUTPUT_FILE = f"{DATA_DIR}/caged_curitiba_consolidado_test_com_previsoes.csv"
TARGET_COLUMN = "saldomovimentacao"
ALLOWED_MONTHS = {202401, 202411, 202412, 202501, 202511, 202512}
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

NUMERIC_COLUMNS = [
    "competenciamov",
    "regiao",
    "uf",
    "municipio",
    "subclasse",
    "cbo2002ocupacao",
    "categoria",
    "graudeinstrucao",
    "idade",
    "horascontratuais",
    "racacor",
    "sexo",
    "tipoempregador",
    "tipoestabelecimento",
    "tipodedeficiencia",
    "indtrabintermitente",
    "indtrabparcial",
    "salario",
    "tamestabjan",
    "indicadoraprendiz",
    "origemdainformacao",
    "competenciadec",
    "unidadesalariocodigo",
    "valorsalariofixo",
]


@dataclass
class TrainingArtifacts:
    model_name: str
    model: Pipeline
    x_train: pd.DataFrame
    x_valid: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    removed_columns: list[str]


def load_csv(path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(
        path,
        sep=";",
        encoding="latin-1",
        decimal=",",
        low_memory=False,
    )
    if dataframe.shape[1] == 1:
        dataframe = pd.read_csv(
            path,
            sep=",",
            encoding="latin-1",
            decimal=".",
            low_memory=False,
        )

    dataframe.columns = [column.strip().lower() for column in dataframe.columns]
    return dataframe


def convert_numeric_column(dataframe: pd.DataFrame, column_name: str) -> None:
    if column_name not in dataframe.columns:
        return

    series = dataframe[column_name].astype(str).str.strip()
    series = series.replace({"": None, "nan": None, "None": None})

    has_comma = series.dropna().str.contains(",", regex=False).any()
    has_dot = series.dropna().str.contains(".", regex=False).any()

    if has_comma and has_dot:
        series = series.str.replace(".", "", regex=False)
        series = series.str.replace(",", ".", regex=False)
    elif has_comma:
        series = series.str.replace(",", ".", regex=False)

    dataframe[column_name] = pd.to_numeric(series, errors="coerce")


def add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "cbo2002ocupacao" in dataframe.columns:
        dataframe["cbo_grupo"] = (
            dataframe["cbo2002ocupacao"].fillna(-1).astype(int).astype(str).str[:2]
        )

    if "subclasse" in dataframe.columns:
        dataframe["subclasse_grupo"] = (
            dataframe["subclasse"].fillna(-1).astype(int).astype(str).str[:3]
        )

    if "secao" in dataframe.columns and "subclasse" in dataframe.columns:
        dataframe["secao_subclasse"] = (
            dataframe["secao"].astype(str)
            + "_"
            + dataframe["subclasse"].fillna(-1).astype(int).astype(str).str[:3]
        )

    if "competenciamov" in dataframe.columns:
        dataframe["mes_mov"] = (dataframe["competenciamov"] % 100).astype("Int64")
        dataframe["ano_mov"] = (dataframe["competenciamov"] // 100).astype("Int64")

    if "competenciadec" in dataframe.columns:
        dataframe["mes_dec"] = (dataframe["competenciadec"] % 100).astype("Int64")
        dataframe["ano_dec"] = (dataframe["competenciadec"] // 100).astype("Int64")

    if {"competenciadec", "competenciamov"}.issubset(dataframe.columns):
        dataframe["defasagem_competencia"] = (
            dataframe["competenciadec"] - dataframe["competenciamov"]
        )

    if {"salario", "valorsalariofixo"}.issubset(dataframe.columns):
        salary_base = dataframe["valorsalariofixo"].replace(0, pd.NA)
        dataframe["razao_salario"] = (dataframe["salario"] / salary_base).astype(float)
        dataframe["dif_salario"] = (
            dataframe["salario"] - dataframe["valorsalariofixo"]
        ).astype(float)

    if "idade" in dataframe.columns:
        dataframe["faixa_idade"] = pd.cut(
            dataframe["idade"],
            bins=[0, 17, 24, 34, 44, 54, 64, 120],
            labels=[
                "ate_17",
                "18_24",
                "25_34",
                "35_44",
                "45_54",
                "55_64",
                "65_mais",
            ],
            include_lowest=True,
        ).astype(str)

    return dataframe


def preprocess_dataframe(dataframe: pd.DataFrame, include_target: bool) -> pd.DataFrame:
    dataframe = dataframe.copy()

    numeric_columns = list(NUMERIC_COLUMNS)
    if include_target:
        numeric_columns.append(TARGET_COLUMN)

    for column in numeric_columns:
        convert_numeric_column(dataframe, column)

    return add_engineered_features(dataframe)


def filter_allowed_months(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "competenciadec" not in dataframe.columns:
        raise ValueError("A coluna 'competenciadec' nao foi encontrada no dataset.")
    return dataframe[dataframe["competenciadec"].isin(ALLOWED_MONTHS)].copy()


def split_features_target(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"A coluna alvo '{TARGET_COLUMN}' nao foi encontrada no arquivo de treino."
        )

    filtered = dataframe[dataframe[TARGET_COLUMN].isin([-1, 1])].copy()
    y = filtered[TARGET_COLUMN].astype(int)
    x = filtered.drop(columns=[TARGET_COLUMN])
    return x, y


def align_feature_sets(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    removed_columns: list[str] = []

    constant_columns = [
        column
        for column in x_train.columns
        if x_train[column].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        x_train = x_train.drop(columns=constant_columns)
        removed_columns.extend(constant_columns)

    common_columns = [column for column in x_train.columns if column in x_test.columns]
    train_only_columns = [column for column in x_train.columns if column not in x_test.columns]
    if train_only_columns:
        removed_columns.extend(train_only_columns)

    return x_train[common_columns].copy(), x_test[common_columns].copy(), removed_columns


def build_classifier(x_train: pd.DataFrame) -> Pipeline:
    numeric_columns = x_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = x_train.select_dtypes(exclude=["number"]).columns.tolist()
    return Pipeline(
        steps=[
            (
                "preprocessamento",
                ColumnTransformer(
                    transformers=[
                        (
                            "numericas",
                            Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                            numeric_columns,
                        ),
                        (
                            "categoricas",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    (
                                        "encoder",
                                        OrdinalEncoder(
                                            handle_unknown="use_encoded_value",
                                            unknown_value=-1,
                                        ),
                                    ),
                                ]
                            ),
                            categorical_columns,
                        ),
                    ]
                ),
            ),
            (
                "modelo",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_estimators=250,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_best_model(
    x: pd.DataFrame,
    y: pd.Series,
) -> TrainingArtifacts:
    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_classifier(x_train)
    model.fit(x_train, y_train)

    return TrainingArtifacts(
        model_name="RandomForestClassifier",
        model=model,
        x_train=x_train,
        x_valid=x_valid,
        y_train=y_train,
        y_valid=y_valid,
        removed_columns=[],
    )


def evaluate_model(
    model: Pipeline, x_valid: pd.DataFrame, y_valid: pd.Series
) -> dict[str, object]:
    predictions = model.predict(x_valid)
    report_dict = classification_report(
        y_valid,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report_dict).transpose()

    for column in ["precision", "recall", "f1-score"]:
        if column in report_frame.columns:
            report_frame[column] = report_frame[column] * 100

    if "support" in report_frame.columns:
        report_frame["support"] = report_frame["support"].astype(int)

    lines_to_show = ["-1", "1", "macro avg", "weighted avg"]
    report_frame = report_frame.loc[[line for line in lines_to_show if line in report_frame.index]]

    return {
        "f1_binary": f1_score(y_valid, predictions),
        "f1_macro": f1_score(y_valid, predictions, average="macro"),
        "accuracy": accuracy_score(y_valid, predictions),
        "report_frame": report_frame,
    }


def predict_test(model: Pipeline, x_test: pd.DataFrame) -> pd.DataFrame:
    predictions = model.predict(x_test)
    result = x_test.copy()
    result[TARGET_COLUMN] = predictions
    return result


def save_predictions_csv(dataframe: pd.DataFrame, output_path: str | Path) -> None:
    dataframe.to_csv(output_path, index=False, encoding="utf-8")


def summarize_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, object]:
    return {
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "target_distribution": train_df[TARGET_COLUMN].value_counts().to_dict()
        if TARGET_COLUMN in train_df.columns
        else {},
        "competenciadec_distribution": train_df["competenciadec"].value_counts().sort_index().to_dict()
        if "competenciadec" in train_df.columns
        else {},
    }


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
