import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TRAIN_FILE = "caged_curitiba_consolidado_train.csv"
TEST_FILE = "caged_curitiba_consolidado_test.csv"
OUTPUT_FILE = "caged_curitiba_consolidado_test_com_previsoes.csv"
TARGET_COLUMN = "saldomovimentacao"
ALLOWED_MONTHS = {202401, 202411, 202412, 202501, 202511, 202512}


def carregar_dados(caminho_arquivo):
    # Tenta primeiro no formato pedido no trabalho.
    dataframe = pd.read_csv(
        caminho_arquivo,
        sep=";",
        encoding="latin-1",
        decimal=",",
        low_memory=False,
    )

    # Se o arquivo vier separado por vírgula, faz uma segunda leitura.
    if dataframe.shape[1] == 1:
        dataframe = pd.read_csv(
            caminho_arquivo,
            sep=",",
            encoding="latin-1",
            decimal=".",
            low_memory=False,
        )

    dataframe.columns = [coluna.strip().lower() for coluna in dataframe.columns]
    return dataframe


def converter_coluna_numerica(dataframe, nome_coluna):
    if nome_coluna not in dataframe.columns:
        return

    serie = dataframe[nome_coluna].astype(str).str.strip()
    serie = serie.replace({"": None, "nan": None, "None": None})

    tem_virgula = serie.dropna().str.contains(",", regex=False).any()
    tem_ponto = serie.dropna().str.contains(".", regex=False).any()

    if tem_virgula and tem_ponto:
        serie = serie.str.replace(".", "", regex=False)
        serie = serie.str.replace(",", ".", regex=False)
    elif tem_virgula:
        serie = serie.str.replace(",", ".", regex=False)

    dataframe[nome_coluna] = pd.to_numeric(serie, errors="coerce")


def criar_atributos_simples(dataframe):
    if "cbo2002ocupacao" in dataframe.columns:
        dataframe["cbo_grupo"] = (
            dataframe["cbo2002ocupacao"]
            .fillna(-1)
            .astype(int)
            .astype(str)
            .str[:2]
        )

    if "subclasse" in dataframe.columns:
        dataframe["subclasse_grupo"] = (
            dataframe["subclasse"]
            .fillna(-1)
            .astype(int)
            .astype(str)
            .str[:3]
        )

    return dataframe


def preprocessar_dados(dataframe, tem_alvo):
    dataframe = dataframe.copy()

    colunas_numericas = [
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

    if tem_alvo:
        colunas_numericas.append(TARGET_COLUMN)

    for coluna in colunas_numericas:
        converter_coluna_numerica(dataframe, coluna)

    dataframe = criar_atributos_simples(dataframe)
    return dataframe


def filtrar_meses_validos(dataframe):
    if "competenciadec" not in dataframe.columns:
        raise ValueError("A coluna 'competenciadec' nao foi encontrada no dataset.")

    return dataframe[dataframe["competenciadec"].isin(ALLOWED_MONTHS)].copy()


def separar_treino_e_alvo(dataframe):
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"A coluna alvo '{TARGET_COLUMN}' não foi encontrada no arquivo de treino.")

    dataframe = dataframe.copy()
    dataframe = dataframe[dataframe[TARGET_COLUMN].isin([-1, 1])].copy()

    y = dataframe[TARGET_COLUMN].astype(int)
    x = dataframe.drop(columns=[TARGET_COLUMN])
    return x, y


def selecionar_atributos(x_treino, x_teste):
    colunas_removidas = []

    # Remove colunas constantes no treino, pois nao ajudam na previsao.
    colunas_constantes = [
        coluna
        for coluna in x_treino.columns
        if x_treino[coluna].nunique(dropna=False) <= 1
    ]

    if colunas_constantes:
        x_treino = x_treino.drop(columns=colunas_constantes)
        colunas_removidas.extend(colunas_constantes)

    # Remove colunas que nao existem nos dois datasets para manter o mesmo layout de entrada.
    colunas_em_comum = [coluna for coluna in x_treino.columns if coluna in x_teste.columns]
    colunas_fora_do_teste = [coluna for coluna in x_treino.columns if coluna not in x_teste.columns]

    if colunas_fora_do_teste:
        colunas_removidas.extend(colunas_fora_do_teste)

    x_treino = x_treino[colunas_em_comum].copy()
    x_teste = x_teste[colunas_em_comum].copy()

    return x_treino, x_teste, colunas_removidas


def montar_pipeline(x_treino):
    colunas_numericas = x_treino.select_dtypes(include=["number"]).columns.tolist()
    colunas_categoricas = x_treino.select_dtypes(exclude=["number"]).columns.tolist()

    transformador_numerico = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformador_categorico = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessador = ColumnTransformer(
        transformers=[
            ("numericas", transformador_numerico, colunas_numericas),
            ("categoricas", transformador_categorico, colunas_categoricas),
        ]
    )

    modelo = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessamento", preprocessador),
            ("modelo", modelo),
        ]
    )

    return pipeline


def treinar_modelo(x, y):
    x_treino, x_validacao, y_treino, y_validacao = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = montar_pipeline(x_treino)
    pipeline.fit(x_treino, y_treino)

    return pipeline, x_treino, x_validacao, y_treino, y_validacao


def avaliar_modelo(modelo, x_treino, x_validacao, y_validacao):
    previsoes_validacao = modelo.predict(x_validacao)
    f1 = f1_score(y_validacao, previsoes_validacao)
    acuracia = (previsoes_validacao == y_validacao).mean()
    relatorio = classification_report(
        y_validacao,
        previsoes_validacao,
        output_dict=True,
        zero_division=0,
    )
    tabela_relatorio = pd.DataFrame(relatorio).transpose()

    for coluna in ["precision", "recall", "f1-score"]:
        if coluna in tabela_relatorio.columns:
            tabela_relatorio[coluna] = tabela_relatorio[coluna] * 100

    if "support" in tabela_relatorio.columns:
        tabela_relatorio["support"] = tabela_relatorio["support"].astype(int)

    linhas_exibidas = ["-1", "1", "macro avg", "weighted avg"]
    tabela_relatorio = tabela_relatorio.loc[linhas_exibidas]

    print("\n========== RESULTADOS DO MODELO ==========")
    print("Modelo utilizado: LogisticRegression (class_weight='balanced')")
    print(f"Quantidade de dados de treino: {len(x_treino)}")
    print(f"Quantidade de dados de validacao: {len(x_validacao)}")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print(f"Acuracia: {acuracia * 100:.2f}%")
    print("\nClassification Report (%):")
    print(tabela_relatorio.to_string(float_format=lambda valor: f"{valor:.2f}"))
    print("==========================================")


def mostrar_informacoes_dataset(treino, teste, x_treino, x_validacao):
    print("========== INFORMACOES DO DATASET ==========")
    print(f"Treino - linhas: {treino.shape[0]} | colunas: {treino.shape[1]}")
    print(f"Teste  - linhas: {teste.shape[0]} | colunas: {teste.shape[1]}")
    print(f"Dados de treino: {len(x_treino)}")
    print(f"Dados de validacao: {len(x_validacao)}")
    print("============================================")


def mostrar_colunas_removidas(colunas_removidas):
    print("\n========== SELECAO DE ATRIBUTOS ==========")
    if colunas_removidas:
        print("Colunas removidas por nao ajudarem o modelo:")
        for coluna in colunas_removidas:
            print(f"- {coluna}")
    else:
        print("Nenhuma coluna foi removida.")
    print("==========================================")


def gerar_previsoes(modelo, dataframe_teste):
    previsoes = modelo.predict(dataframe_teste)

    resultado = dataframe_teste.copy()
    resultado[TARGET_COLUMN] = previsoes

    return resultado


def salvar_previsoes_csv(dataframe_resultado, caminho_saida):
    dataframe_resultado.to_csv(caminho_saida, index=False, encoding="utf-8")
    print(f"\nArquivo gerado: {caminho_saida}")


def main():
    pasta_projeto = Path(__file__).resolve().parent
    caminho_treino = pasta_projeto / TRAIN_FILE
    caminho_teste = pasta_projeto / TEST_FILE

    if not caminho_treino.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho_treino}")

    if not caminho_teste.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho_teste}")

    dados_treino = carregar_dados(caminho_treino)
    dados_teste_original = carregar_dados(caminho_teste)
    dados_teste = dados_teste_original.copy()

    dados_treino = preprocessar_dados(dados_treino, tem_alvo=True)
    dados_teste = preprocessar_dados(dados_teste, tem_alvo=False)
    dados_treino = filtrar_meses_validos(dados_treino)

    x, y = separar_treino_e_alvo(dados_treino)
    x, dados_teste_modelo, colunas_removidas = selecionar_atributos(x, dados_teste)
    modelo, x_treino, x_validacao, y_treino, y_validacao = treinar_modelo(x, y)

    mostrar_informacoes_dataset(dados_treino, dados_teste, x_treino, x_validacao)
    mostrar_colunas_removidas(colunas_removidas)
    avaliar_modelo(modelo, x_treino, x_validacao, y_validacao)

    modelo.fit(x, y)
    previsoes_teste = gerar_previsoes(modelo, dados_teste_modelo)
    resultado_teste = dados_teste_original.copy()
    resultado_teste[TARGET_COLUMN] = previsoes_teste[TARGET_COLUMN].values
    salvar_previsoes_csv(resultado_teste, pasta_projeto / OUTPUT_FILE)


if __name__ == "__main__":
    try:
        main()
    except Exception as erro:
        print(f"Erro na execucao: {erro}", file=sys.stderr)
        sys.exit(1)
