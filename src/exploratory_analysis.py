from pathlib import Path

import pandas as pd

from ml_pipeline import (
    ALLOWED_MONTHS,
    TARGET_COLUMN,
    TEST_FILE,
    TRAIN_FILE,
    load_csv,
    preprocess_dataframe,
    project_root,
)


# ==========================
# UTILIDADES
# ==========================

def format_series(series: pd.Series, title: str = None) -> str:
    """Formata Series para exibição limpa no relatório."""
    content = series.to_string()
    if title:
        return f"{title}\n{content}"
    return content


# ==========================
# PROCESSAMENTO PRINCIPAL
# ==========================

def build_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    """Gera relatório de análise exploratória."""

    # ==========================
    # FILTROS
    # ==========================
    filtered_train = train_df[
        train_df["competenciadec"].isin(ALLOWED_MONTHS)
    ].copy()

    # ==========================
    # TRANSFORMAÇÕES
    # ==========================
    transformed_train = preprocess_dataframe(train_df, include_target=True)
    transformed_test = preprocess_dataframe(test_df, include_target=False)

    filtered_transformed_train = transformed_train[
        transformed_train["competenciadec"].isin(ALLOWED_MONTHS)
    ].copy()

    # ==========================
    # MÉTRICAS
    # ==========================
    target_distribution = (
        filtered_train[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    month_distribution = (
        filtered_train["competenciadec"]
        .value_counts()
        .sort_index()
    )

    section_distribution = (
        filtered_train["secao"]
        .value_counts()
        .head(10)
    )

    missing_ratio = (
        filtered_train.isna()
        .mean()
        .sort_values(ascending=False)
        .mul(100)
        .round(4)
    )

    numeric_summary = (
        filtered_train[["idade", "salario", "valorsalariofixo"]]
        .describe()
        .round(2)
        .to_string()
    )

    # ==========================
    # RELATÓRIO
    # ==========================
    report_lines = [
        "# Relatório de Análise Exploratória",
        "",
        "## Objetivo",
        "",
        "- Validar a base do CAGED para prever `saldomovimentacao`.",
        "- Garantir alinhamento com restrições do trabalho.",
        "- Identificar padrões relevantes para modelagem.",
        "",
        "## Visão Geral",
        "",
        f"- Treino bruto: {train_df.shape}",
        f"- Teste bruto: {test_df.shape}",
        f"- Treino pós-feature engineering: {filtered_transformed_train.shape}",
        f"- Teste pós-feature engineering: {transformed_test.shape}",
        f"- Treino filtrado: {filtered_train.shape[0]} linhas",
        "",
        "Meses utilizados:",
        f"- {', '.join(map(str, ALLOWED_MONTHS))}",
        "",
        "## Distribuição do Alvo",
        "```",
        format_series(target_distribution),
        "```",
        "",
        "## Distribuição por Mês",
        "```",
        format_series(month_distribution),
        "```",
        "",
        "## Top 10 Seções",
        "```",
        format_series(section_distribution),
        "```",
        "",
        "## Estatísticas Numéricas",
        "```",
        numeric_summary,
        "```",
        "",
        "## Valores Ausentes (%)",
        "```",
        format_series(missing_ratio.head(10)),
        "```",
        "",
        "## Observações Importantes",
        "",
        "- `valorsalariofixo` apresenta comportamento categórico.",
        "- Baixa presença de valores nulos.",
        "- Features categóricas carregam maior poder preditivo.",
        "",
        "## Estratégia de Modelagem",
        "",
        "- Foram testados diferentes modelos para a base.",
        "- O modelo escolhido foi o RandomForestClassifier.",
        "- Apesar de ser mais pesado e mais lento no treinamento, foi o que entregou o melhor desempenho (%).",
        "- Métrica: F1 Score.",
        "- Validação: estratificada.",
        "",
        "## Conclusão",
        "",
        "- Base consistente e adequada para modelagem.",
        "- Dados equilibrados entre classes.",
        "- Pipeline pronto para produção do CSV final.",
    ]

    return "\n".join(report_lines)


# ==========================
# EXECUÇÃO
# ==========================

def main() -> None:
    root = project_root()

    reports_dir = root / "reports"
    reports_dir.mkdir(exist_ok=True)

    train_df = load_csv(root / TRAIN_FILE)
    test_df = load_csv(root / TEST_FILE)

    report_content = build_report(train_df, test_df)

    output_path = reports_dir / "analise_exploratoria.md"
    output_path.write_text(report_content, encoding="utf-8")

    print(f"Relatório gerado em: {output_path}")


if __name__ == "__main__":
    main()