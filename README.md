# Credit Risk Observability

Projeto de observabilidade, qualidade de dados e monitoramento de drift para um modelo de classificação de risco de crédito.

O projeto utiliza o **Credit Risk Dataset**, disponibilizado publicamente no OpenML com o ID [`43454`](https://www.openml.org/d/43454). O download não exige conta nem token do Kaggle.

## Objetivo

Construir uma camada de sustentação e confiabilidade para um modelo de Credit Scoring, contemplando:

* validação e contratos de dados;
* bloqueio de lotes inválidos;
* treinamento de um modelo baseline;
* simulação de dados de produção;
* detecção estatística de Data Drift;
* análise da degradação do modelo;
* geração de relatório visual com Evidently;
* testes automatizados;
* documentação e governança.

## Dataset

O dataset possui aproximadamente 32 mil registros e representa solicitações de crédito.

A variável-alvo é `loan_status`:

* `0`: cliente não inadimplente;
* `1`: cliente inadimplente.

O dataset é carregado diretamente do OpenML:

```text
OpenML Dataset ID: 43454
Registros validados: 32.409
Problema: classificação binária financeira
```

## Tecnologias

* Python 3.12
* uv
* pandas
* scikit-learn
* Pandera
* SciPy
* Evidently
* PyArrow
* pytest
* Ruff
* joblib

## Arquitetura do pipeline

```text
OpenML 43454
      |
      v
Limpeza determinística
      |
      v
Contrato de dados Pandera
      |
      +----------------------+
      |                      |
      v                      v
Modelo baseline       Bloqueio de lote inválido
      |
      v
Dataset de referência
      |
      v
Simulação de produção
      |
      v
Predições do modelo
      |
      v
PSI + Kolmogorov-Smirnov
      |
      v
Relatório Evidently + comparação de desempenho
```

## Estrutura do projeto

```text
.
├── artifacts/
│   ├── metrics/
│   └── models/
├── configs/
├── data/
│   ├── production/
│   ├── raw/
│   ├── reference/
│   └── samples/
├── governance/
├── reports/
│   ├── data_quality/
│   └── drift/
├── scripts/
├── src/
│   └── home_credit_observability/
│       ├── data/
│       ├── drift/
│       ├── features/
│       ├── models/
│       ├── pipelines/
│       ├── reporting/
│       └── validation/
└── tests/
    ├── integration/
    └── unit/
```

## Princípios de projeto

A implementação da detecção de drift utiliza princípios SOLID:

* **Single Responsibility Principle:** simulação, predição, detecção e geração de relatórios possuem componentes separados.
* **Open/Closed Principle:** novos detectores podem ser adicionados sem alterar os detectores existentes.
* **Liskov Substitution Principle:** estratégias compatíveis com o protocolo de detecção podem ser substituídas.
* **Interface Segregation Principle:** o contrato `DriftDetector` define somente a operação necessária para detectar drift.
* **Dependency Inversion Principle:** o serviço de análise depende da abstração dos detectores, recebidos por injeção de dependência.

Os módulos, classes, métodos e funções implementados nas etapas possuem docstrings para documentar suas responsabilidades.

## Instalação

Clone o repositório:

```bash
git clone https://github.com/RafaExMachina/home-credit-observability.git
cd home-credit-observability
```

Sincronize o ambiente:

```bash
uv sync --locked
```

Confirme a versão do Python:

```bash
uv run python --version
```

## Etapa 1 — Validação de dados e modelo baseline

### Download do dataset

```bash
uv run python -m home_credit_observability.cli download
```

### Validação do dataset

```bash
uv run python -m home_credit_observability.cli validate
```

Resultado esperado:

```text
Contrato aprovado para 32,409 registros.
```

### Teste com lote inválido

```bash
uv run python -m home_credit_observability.cli validate-invalid
```

O lote propositalmente inválido contém violações como:

* alvo fora do domínio binário;
* renda negativa;
* idade fora do intervalo permitido;
* registro duplicado.

O contrato Pandera deve bloquear a ingestão e gerar um relatório com as violações.

### Treinamento do modelo

```bash
uv run python -m home_credit_observability.cli train
```

O modelo baseline utiliza Regressão Logística com:

* pré-processamento numérico;
* imputação de valores ausentes;
* padronização;
* codificação One-Hot;
* balanceamento de classes;
* divisão estratificada entre treino e teste.

### Métricas do baseline

| Métrica   | Resultado |
| --------- | --------: |
| Accuracy  |    0,8059 |
| Precision |    0,5389 |
| Recall    |    0,7821 |
| F1-score  |    0,6381 |
| ROC AUC   |    0,8716 |

## Etapa 2 — Simulação e detecção de drift

A Etapa 2 cria um dataset de produção com 8.000 registros e simula mudanças nas condições financeiras dos clientes.

As alterações controladas afetam:

* renda do cliente;
* taxa de juros;
* percentual da renda comprometido pelo empréstimo.

Execute o pipeline:

```bash
uv run python -m home_credit_observability.cli drift
```

Também é possível definir o tamanho da amostra e a semente:

```bash
uv run python -m home_credit_observability.cli drift \
  --sample-size 8000 \
  --random-state 42
```

### Métodos estatísticos

O projeto utiliza dois métodos complementares:

#### Population Stability Index — PSI

O PSI mede a intensidade da mudança entre as distribuições.

Critérios utilizados:

* PSI menor que `0,10`: distribuição estável;
* PSI entre `0,10` e `0,25`: mudança moderada;
* PSI maior ou igual a `0,25`: drift severo.

#### Kolmogorov–Smirnov — KS

O teste KS compara duas distribuições numéricas.

O drift é sinalizado quando:

```text
p-value < 0,05
```

### Resultados da detecção

Foram executados 14 testes:

```text
7 variáveis numéricas × 2 métodos estatísticos
```

Seis testes sinalizaram drift. PSI e KS concordaram nas três variáveis afetadas:

| Variável              |    PSI | Resultado |
| --------------------- | -----: | --------- |
| `person_income`       | 0,3005 | Drift     |
| `loan_int_rate`       | 2,6385 | Drift     |
| `loan_percent_income` | 0,2928 | Drift     |

As seguintes variáveis permaneceram estáveis:

* `person_age`;
* `person_emp_length`;
* `loan_amnt`;
* `cb_person_cred_hist_length`.

### Impacto no desempenho

O pipeline gera predições para os datasets de referência e produção e compara as métricas do modelo.

| Métrica   | Referência | Produção | Variação |
| --------- | ---------: | -------: | -------: |
| Accuracy  |     0,8114 |   0,8498 |  +0,0383 |
| Precision |     0,5487 |   0,6443 |  +0,0956 |
| Recall    |     0,7771 |   0,6823 |  -0,0948 |
| F1-score  |     0,6432 |   0,6627 |  +0,0195 |
| ROC AUC   |     0,8712 |   0,8662 |  -0,0050 |

Embora a acurácia tenha aumentado, o recall caiu aproximadamente 9,48 pontos percentuais.

Em um problema de risco de crédito, essa queda é relevante porque indica que o modelo deixou de identificar uma parcela maior dos clientes inadimplentes. Isso demonstra por que a acurácia não deve ser analisada isoladamente em datasets desbalanceados.

## Relatório Evidently

O Evidently compara os dados de referência com os dados de produção e gera um relatório HTML interativo.

Após executar o pipeline, abra o relatório:

```bash
xdg-open reports/drift/evidently_drift_report.html
```

## Artefatos gerados

### Etapa 1

```text
data/raw/credit_risk_openml_43454.csv
data/reference/reference.parquet
data/samples/invalid_batch.csv
reports/data_quality/invalid_batch_failures.csv
artifacts/models/baseline_pipeline.joblib
artifacts/metrics/baseline_metrics.json
```

### Etapa 2

```text
data/production/production_drifted.parquet
reports/drift/statistical_drift_results.json
reports/drift/model_performance_comparison.json
reports/drift/evidently_drift_report.html
```

Os datasets, modelos e relatórios gerados são ignorados pelo Git. Eles podem ser reproduzidos por meio dos comandos do pipeline.

## Qualidade do código

Execute o linter:

```bash
uv run ruff check .
```

Verifique a formatação:

```bash
uv run ruff format --check .
```

Execute os testes:

```bash
uv run pytest -v
```

Resultado atual:

```text
15 testes aprovados
```

Execute todas as verificações:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

## Scripts auxiliares

```bash
uv run python scripts/download_data.py
uv run python scripts/validate_data.py
uv run python scripts/train_model.py
uv run python scripts/run_drift.py
```

## Versionamento

* `v0.1.0`: validação de dados, contrato Pandera e modelo baseline.
* `v0.2.0`: simulação de produção, detecção de drift e relatório Evidently.

A versão `v0.2.0` será criada após a integração da Etapa 2 na branch `main`.

## Próximas etapas

* consolidar logs e relatórios visuais;
* criar dashboard de monitoramento;
* adicionar métricas para Prometheus;
* documentar políticas de monitoramento e resposta;
* consolidar a documentação de privacidade e LGPD.

## Autor

**Rafael Alves da Costa**

GitHub: [RafaExMachina](https://github.com/RafaExMachina)
