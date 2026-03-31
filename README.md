# Projeto ML (CAGED)

Projeto simples de aprendizado de máquina para prever a coluna `saldomovimentacao`:

- `1` = admissão
- `-1` = desligamento

O modelo usa `LogisticRegression` com pré-processamento básico em `scikit-learn`.

## Base utilizada

Arquivos esperados na raiz do projeto:

- `caged_curitiba_consolidado_train.csv`
- `caged_curitiba_consolidado_test.csv`

No treino, o script filtra apenas os meses permitidos na coluna `competenciadec`:

- `202401`
- `202411`
- `202412`
- `202501`
- `202511`
- `202512`

O arquivo de teste é usado para gerar previsões para janeiro de 2026.

## Como rodar

Crie e ative o ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
python3 -m pip install -r requirements.txt
```

Execute o projeto:

```bash
python3 main.py
```

## O que o programa faz

O script:

- carrega os arquivos CSV
- padroniza os nomes das colunas
- converte colunas numéricas
- trata `horascontratuais`
- cria atributos simples com `cbo2002ocupacao` e `subclasse`
- remove atributos que não ajudam o modelo, como colunas constantes
- divide treino e validação com `train_test_split`
- treina uma `LogisticRegression`
- avalia o modelo com `F1 Score`
- gera o arquivo final com as previsões

## Saída no terminal

Durante a execução, o programa mostra:

- informações do dataset
- colunas removidas na seleção de atributos
- modelo utilizado
- quantidade de dados de treino e validação
- `F1 Score` em porcentagem
- acurácia em porcentagem
- `classification report` em porcentagem
- caminho do arquivo gerado

## Arquivo gerado

Ao final da execução, o script cria:

- `caged_curitiba_consolidado_test_com_previsoes.csv`

Esse arquivo mantém o layout original do arquivo de teste e adiciona a coluna:

- `saldomovimentacao`
