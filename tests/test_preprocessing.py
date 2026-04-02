import pandas as pd
import pytest

from ml_pipeline import (
    TARGET_COLUMN,
    add_engineered_features,
    convert_numeric_column,
    preprocess_dataframe,
)


def test_convert_numeric_column_handles_comma_and_dot() -> None:
    dataframe = pd.DataFrame({"salario": ["1.234,56", "2000,00", "", "None"]})

    convert_numeric_column(dataframe, "salario")

    assert dataframe.loc[0, "salario"] == pytest.approx(1234.56, rel=1e-6)
    assert dataframe.loc[1, "salario"] == pytest.approx(2000.00, rel=1e-6)
    assert pd.isna(dataframe.loc[2, "salario"])
    assert pd.isna(dataframe.loc[3, "salario"])


def test_add_engineered_features_creates_expected_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "cbo2002ocupacao": [123456],
            "subclasse": [6201501],
            "secao": ["J"],
            "competenciamov": [202411],
            "competenciadec": [202412],
            "salario": [3000.0],
            "valorsalariofixo": [2500.0],
            "idade": [29],
        }
    )

    transformed = add_engineered_features(dataframe)

    assert transformed.loc[0, "cbo_grupo"] == "12"
    assert transformed.loc[0, "subclasse_grupo"] == "620"
    assert transformed.loc[0, "secao_subclasse"] == "J_620"
    assert transformed.loc[0, "mes_mov"] == 11
    assert transformed.loc[0, "ano_mov"] == 2024
    assert transformed.loc[0, "defasagem_competencia"] == 1
    assert transformed.loc[0, "razao_salario"] == pytest.approx(1.2, rel=1e-6)


def test_preprocess_dataframe_adds_engineered_columns_and_converts_target() -> None:
    dataframe = pd.DataFrame(
        {
            "competenciamov": ["202411"],
            "competenciadec": ["202412"],
            "cbo2002ocupacao": ["123456"],
            "subclasse": ["6201501"],
            "secao": ["J"],
            "idade": ["29"],
            "salario": ["3.000,00"],
            "valorsalariofixo": ["2500,00"],
            TARGET_COLUMN: ["1"],
        }
    )

    transformed = preprocess_dataframe(dataframe, include_target=True)

    assert transformed.loc[0, TARGET_COLUMN] == 1
    assert transformed.loc[0, "mes_mov"] == 11
    assert transformed.loc[0, "mes_dec"] == 12
    assert transformed.loc[0, "secao_subclasse"] == "J_620"
    assert transformed.loc[0, "faixa_idade"] == "25_34"
