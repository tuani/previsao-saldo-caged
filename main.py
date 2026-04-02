import sys

from ml_pipeline import (
    OUTPUT_FILE,
    TEST_FILE,
    TRAIN_FILE,
    TARGET_COLUMN,
    align_feature_sets,
    evaluate_model,
    filter_allowed_months,
    load_csv,
    predict_test,
    preprocess_dataframe,
    project_root,
    save_predictions_csv,
    split_features_target,
    summarize_dataset,
    train_best_model,
)


def print_dataset_info(summary: dict[str, object]) -> None:
    train_shape = summary["train_shape"]
    test_shape = summary["test_shape"]
    target_distribution = summary["target_distribution"]
    competenciadec_distribution = summary["competenciadec_distribution"]

    print("========== INFORMACOES DO DATASET ==========")
    print(f"Treino - linhas: {train_shape[0]} | colunas: {train_shape[1]}")
    print(f"Teste  - linhas: {test_shape[0]} | colunas: {test_shape[1]}")
    print(f"Distribuicao do alvo: {target_distribution}")
    print(f"Competencias mantidas no treino: {competenciadec_distribution}")
    print("============================================")


def print_removed_columns(removed_columns: list[str]) -> None:
    print("\n========== SELECAO DE ATRIBUTOS ==========")
    if removed_columns:
        print("Colunas removidas por nao ajudarem o modelo ou nao existirem no teste:")
        for column in removed_columns:
            print(f"- {column}")
    else:
        print("Nenhuma coluna foi removida.")
    print("==========================================")

def print_evaluation(
    model_name: str, metrics: dict[str, object], train_size: int, valid_size: int
) -> None:
    report_frame = metrics["report_frame"]

    print("\n========== RESULTADOS DO MODELO ==========")
    print(f"Modelo utilizado: {model_name}")
    print(f"Quantidade de dados de treino: {train_size}")
    print(f"Quantidade de dados de validacao: {valid_size}")
    print(f"F1 Score (classe 1): {metrics['f1_binary'] * 100:.2f}%")
    print(f"F1 Score Macro: {metrics['f1_macro'] * 100:.2f}%")
    print(f"Acuracia: {metrics['accuracy'] * 100:.2f}%")
    print("\nClassification Report (%):")
    print(report_frame.to_string(float_format=lambda value: f"{value:.2f}"))
    print("==========================================")


def main() -> None:
    root = project_root()
    train_path = root / TRAIN_FILE
    test_path = root / TEST_FILE

    if not train_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {test_path}")

    train_raw = load_csv(train_path)
    test_raw = load_csv(test_path)

    train_processed = preprocess_dataframe(train_raw, include_target=True)
    test_processed = preprocess_dataframe(test_raw, include_target=False)
    train_filtered = filter_allowed_months(train_processed)

    summary = summarize_dataset(train_filtered, test_processed)
    print_dataset_info(summary)

    x, y = split_features_target(train_filtered)
    x_aligned, test_aligned, removed_columns = align_feature_sets(x, test_processed)
    print_removed_columns(removed_columns)

    artifacts = train_best_model(x_aligned, y)
    metrics = evaluate_model(artifacts.model, artifacts.x_valid, artifacts.y_valid)
    print_evaluation(
        artifacts.model_name,
        metrics,
        train_size=len(artifacts.x_train),
        valid_size=len(artifacts.x_valid),
    )

    artifacts.model.fit(x_aligned, y)
    predicted_test = predict_test(artifacts.model, test_aligned)
    final_output = test_raw.copy()
    final_output[TARGET_COLUMN] = predicted_test[TARGET_COLUMN].values

    output_path = root / OUTPUT_FILE
    save_predictions_csv(final_output, output_path)
    print(f"\nArquivo gerado: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Erro na execucao: {error}", file=sys.stderr)
        sys.exit(1)
