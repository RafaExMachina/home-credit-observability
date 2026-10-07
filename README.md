# Credit Risk Observability

Projeto de observabilidade, qualidade de dados e monitoramento de drift para um modelo de classificação de risco de crédito.

O projeto utiliza o **Credit Risk Dataset**, disponibilizado publicamente no OpenML com o ID [`43454`](https://www.openml.org/d/43454). O download não exige conta nem token do Kaggle.

## Objetivo

Construir uma camada de sustentação e confiabilidade para um modelo de Credit Scoring, contemplando:

- validação e contratos de dados;
- bloqueio de lotes inválidos;
- treinamento de um modelo baseline;
- simulação de dados de produção;
- detecção estatística de Data Drift;
- análise da degradação do modelo;
- geração de relatório visual com Evidently;
- logs estruturados de execução e alertas;
- métricas operacionais no formato Prometheus;
- dashboard HTML interativo;
- testes automatizados;
- documentação e governança.

## Dataset

O dataset possui aproximadamente 32 mil registros e representa solicitações de crédito.

A variável-alvo é `loan_status`:

- `0`: cliente não inadimplente;
- `1`: cliente inadimplente.

```text
OpenML Dataset ID: 43454
Registros validados: 32.409
Problema: classificação binária financeira
```

## Tecnologias

- Python 3.12
- uv
- pandas
- scikit-learn
- Pandera
- SciPy
- Evidently
- Plotly
- Prometheus Client
- PyArrow
- pytest
- Ruff
- joblib

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
      |
      v
Logs + métricas + políticas de alerta
      |
      v
Dashboard Plotly + exposição Prometheus
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
│   ├── drift/
│   ├── logs/
│   └── monitoring/
├── scripts/
├── src/
│   └── home_credit_observability/
│       ├── data/
│       ├── drift/
│       ├── features/
│       ├── models/
│       ├── observability/
│       ├── pipelines/
│       ├── reporting/
│       └── validation/
└── tests/
    ├── integration/
    └── unit/
```

## Princípios de projeto

A implementação utiliza princípios SOLID:

- **Single Responsibility Principle:** validação, simulação, predição, detecção, logs, métricas, alertas e dashboards possuem componentes separados.
- **Open/Closed Principle:** novos detectores, exportadores e políticas podem ser adicionados sem alterar os componentes existentes.
- **Liskov Substitution Principle:** implementações compatíveis com os protocolos podem ser substituídas.
- **Interface Segregation Principle:** os contratos definem somente as operações necessárias para cada responsabilidade.
- **Dependency Inversion Principle:** os serviços dependem de abstrações e recebem colaboradores por injeção de dependência.

Os módulos, classes, métodos e funções implementados possuem docstrings que documentam suas responsabilidades.

## Instalação

```bash
git clone https://github.com/RafaExMachina/home-credit-observability.git
cd home-credit-observability
uv sync --locked
uv run python --version
```

## Etapa 1 — Validação de dados e modelo baseline

### Download e validação

```bash
uv run python -m home_credit_observability.cli download
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

- alvo fora do domínio binário;
- renda negativa;
- idade fora do intervalo permitido;
- registro duplicado.

O contrato Pandera bloqueia a ingestão e gera um relatório com as violações.

### Treinamento do modelo

```bash
uv run python -m home_credit_observability.cli train
```

O modelo baseline utiliza Regressão Logística com pré-processamento numérico, imputação, padronização, codificação One-Hot, balanceamento de classes e divisão estratificada.

### Métricas do baseline

| Métrica | Resultado |
|---|---:|
| Accuracy | 0,8059 |
| Precision | 0,5389 |
| Recall | 0,7821 |
| F1-score | 0,6381 |
| ROC AUC | 0,8716 |

## Etapa 2 — Simulação e detecção de drift

A Etapa 2 cria um dataset de produção com 8.000 registros e simula mudanças controladas na renda do cliente, na taxa de juros e no percentual da renda comprometido pelo empréstimo.

```bash
uv run python -m home_credit_observability.cli drift
```

Também é possível configurar a execução:

```bash
uv run python -m home_credit_observability.cli drift \
  --sample-size 8000 \
  --random-state 42
```

### Métodos estatísticos

O **Population Stability Index (PSI)** mede a intensidade da mudança entre as distribuições. O projeto considera drift severo quando `PSI >= 0,25`.

O teste **Kolmogorov–Smirnov (KS)** compara duas distribuições numéricas. O drift é sinalizado quando `p-value < 0,05`.

### Resultados da detecção

Foram executados 14 testes, correspondentes a sete variáveis numéricas analisadas por dois métodos. Seis testes sinalizaram drift e PSI e KS concordaram nas três variáveis afetadas:

| Variável | PSI | Resultado |
|---|---:|---|
| `person_income` | 0,3005 | Drift |
| `loan_int_rate` | 2,6385 | Drift |
| `loan_percent_income` | 0,2928 | Drift |

Permaneceram estáveis `person_age`, `person_emp_length`, `loan_amnt` e `cb_person_cred_hist_length`.

### Impacto no desempenho

| Métrica | Referência | Produção | Variação |
|---|---:|---:|---:|
| Accuracy | 0,8114 | 0,8498 | +0,0383 |
| Precision | 0,5487 | 0,6443 | +0,0956 |
| Recall | 0,7771 | 0,6823 | -0,0948 |
| F1-score | 0,6432 | 0,6627 | +0,0195 |
| ROC AUC | 0,8712 | 0,8662 | -0,0050 |

Embora a acurácia tenha aumentado, o recall caiu aproximadamente 9,48 pontos percentuais. Em risco de crédito, essa queda indica que o modelo deixou de identificar uma parcela maior dos clientes inadimplentes.

## Etapa 3 — Observabilidade e monitoramento

A Etapa 3 centraliza os resultados de drift, as métricas do modelo e os eventos operacionais em uma camada de monitoramento reproduzível.

Execute o pipeline completo:

```bash
uv run python -m home_credit_observability.cli monitor
```

Também é possível configurar a amostra e a semente:

```bash
uv run python -m home_credit_observability.cli monitor \
  --sample-size 8000 \
  --random-state 42
```

### Logs estruturados

Os eventos são persistidos em JSON Lines e incluem timestamp UTC, nível, nome do evento e atributos. Os principais eventos são:

- `monitoring_started`;
- `monitoring_completed`;
- `alert_triggered`.

### Métricas de saúde

O pipeline consolida:

- total de predições;
- proporção de predições nulas;
- proporção de classificações positivas;
- confiança média;
- latência por registro em p50, p95 e p99;
- quantidade de variáveis e testes com drift;
- recall e variação do recall;
- ROC AUC em produção;
- duração total do pipeline.

As métricas também são exportadas no formato de exposição de texto do Prometheus.

### Políticas de alerta

| Alerta | Severidade | Condição |
|---|---|---|
| `PREDICTION_NULL_RATIO` | critical | proporção de predições nulas maior que zero |
| `INFERENCE_LATENCY_P95` | warning | latência p95 maior que 200 ms |
| `DATA_DRIFT_DETECTED` | warning | uma ou mais variáveis com drift |
| `MODEL_RECALL_DEGRADATION` | critical | variação do recall menor que -0,05 |
| `MODEL_ROC_AUC_LOW` | critical | ROC AUC menor que 0,80 |

### Resultado do cenário monitorado

```text
Status geral: critical
Predições processadas: 8.000
Predições nulas: 0%
Variáveis com drift: 3
Testes com alerta de drift: 6
Recall em produção: 0,6823
Variação do recall: -0,0948
ROC AUC em produção: 0,8662
Alertas ativos: 2
```

Alertas produzidos:

- `DATA_DRIFT_DETECTED`, com severidade `warning`;
- `MODEL_RECALL_DEGRADATION`, com severidade `critical`.

O estado global é `critical` porque a queda do recall ultrapassou o limite operacional definido.

### Dashboard

O dashboard Plotly reúne indicadores de saúde, PSI por variável, comparação de desempenho e alertas ativos.

Para servi-lo localmente:

```bash
uv run python -m http.server 8765 \
  --bind 127.0.0.1 \
  --directory reports/monitoring
```

Acesse [http://127.0.0.1:8765/observability_dashboard.html](http://127.0.0.1:8765/observability_dashboard.html).

## Relatório Evidently

O Evidently compara os dados de referência com os dados de produção e gera um relatório HTML interativo em `reports/drift/evidently_drift_report.html`.

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

### Etapa 3

```text
reports/logs/observability.jsonl
reports/monitoring/health_metrics.json
reports/monitoring/alerts.json
reports/monitoring/prometheus_metrics.prom
reports/monitoring/observability_dashboard.html
```

Os datasets, modelos e relatórios gerados são ignorados pelo Git e podem ser reproduzidos pelos comandos do pipeline.

## Qualidade do código

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
uv build
```

Resultado atual:

```text
20 testes aprovados
```

## Scripts auxiliares

```bash
uv run python scripts/download_data.py
uv run python scripts/validate_data.py
uv run python scripts/train_model.py
uv run python scripts/run_drift.py
uv run python scripts/run_monitoring.py
```

## Versionamento

- `v0.1.0`: validação de dados, contrato Pandera e modelo baseline.
- `v0.2.0`: simulação de produção, detecção de drift e relatório Evidently.
- `v0.3.0` (próxima versão): observabilidade, logs estruturados, métricas Prometheus, alertas e dashboard.

## Próximas etapas

- documentar o plano de privacidade e conformidade com a LGPD;
- consolidar o dicionário de dados e a política de monitoramento;
- finalizar o model card;
- preparar a apresentação e o vídeo demonstrativo.

## Autor

**Rafael Alves da Costa**

GitHub: [RafaExMachina](https://github.com/RafaExMachina)
