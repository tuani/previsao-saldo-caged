# Relatório de Análise Exploratória

## Objetivo

- Validar a base do CAGED para prever `saldomovimentacao`.
- Garantir alinhamento com restrições do trabalho.
- Identificar padrões relevantes para modelagem.

## Visão Geral

- Treino bruto: (742427, 26)
- Teste bruto: (140020, 25)
- Treino pós-feature engineering: (741271, 37)
- Teste pós-feature engineering: (140020, 36)
- Treino filtrado: 741271 linhas

Meses utilizados:
- 202512, 202401, 202501, 202411, 202412, 202511

## Distribuição do Alvo
```
saldomovimentacao
-1    51.02
 1    48.98
```

## Distribuição por Mês
```
competenciadec
202401    128485
202411    123091
202412    110627
202501    141911
202511    123462
202512    113695
```

## Top 10 Seções
```
secao
N    224117
G    160726
C     75677
F     65225
I     53489
H     40250
Q     24667
M     23620
P     19789
J     14687
```

## Estatísticas Numéricas
```
           idade     salario  valorsalariofixo
count  741268.00   741271.00         741271.00
mean       33.10     2318.93              4.65
std        11.51     5755.78              2.82
min        14.00        0.00              1.00
25%        24.00     1685.20              5.00
50%        31.00     1928.00              5.00
75%        41.00     2300.00              5.00
max        97.00  1229200.00             99.00
```

## Valores Ausentes (%)
```
idade                   0.0004
competenciamov          0.0000
tipoempregador          0.0000
unidadesalariocodigo    0.0000
competenciadec          0.0000
origemdainformacao      0.0000
indicadoraprendiz       0.0000
tamestabjan             0.0000
salario                 0.0000
indtrabparcial          0.0000
```

## Observações Importantes

- `valorsalariofixo` apresenta comportamento categórico.
- Baixa presença de valores nulos.
- Features categóricas carregam maior poder preditivo.

## Estratégia de Modelagem

- Foram testados diferentes modelos para a base.
- O modelo escolhido foi o RandomForestClassifier.
- Apesar de ser mais pesado e mais lento no treinamento, foi o que entregou o melhor desempenho (%).
- Métrica: F1 Score.
- Validação: estratificada.

## Conclusão

- Base consistente e adequada para modelagem.
- Dados equilibrados entre classes.
- Pipeline pronto para produção do CSV final.