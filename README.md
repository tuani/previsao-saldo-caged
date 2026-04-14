# Projeto CAGED

Projeto de classificacao para prever a coluna `saldomovimentacao`:

- `1` = admissao
- `-1` = desligamento

O modelo usado no projeto e o `RandomForestClassifier`.

## Arquivos principais

- `src/main.py`: treina o modelo e gera o CSV final
- `exploratory_analysis.ipynb`: analise exploratoria (abrir no Jupyter ou VS Code)
- `src/ml_pipeline.py`: funcoes de carga, tratamento e modelagem
- `tests/`: testes em `pytest`
- `data/`: arquivos CSV

## Como executar

Cria o ambiente virtual:

```bash
python3 -m venv .venv
```

Ativa o ambiente virtual:

```bash
source .venv/bin/activate
```

Instala as dependencias do projeto:

```bash
python3 -m pip install -r requirements.txt
```

Analise exploratoria: abra o notebook `exploratory_analysis.ipynb` no Jupyter ou no editor.

Treina o modelo e gera o CSV final com previsoes:

```bash
python3 src/main.py
```

Executa os testes automatizados:

```bash
pytest -v
```

## Limpeza

Remove caches do Python e do pytest:

```bash
find . -type d \( -name "__pycache__" -o -name ".pytest_cache" \) -prune -exec rm -rf {} +
```

Remove arquivos compilados do Python:

```bash
find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
```

## Arquivos CSV

- `data/caged_curitiba_consolidado_train.csv`
- `data/caged_curitiba_consolidado_test.csv`
- `data/caged_curitiba_consolidado_test_com_previsoes.csv`

## Saida

O arquivo final mantém o layout do teste e adiciona a coluna `saldomovimentacao`.
