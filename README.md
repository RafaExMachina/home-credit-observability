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
- plano de privacidade e proteção de dados;
- governança do modelo e análise causal;
- roteiro de apresentação pelo método STAR;
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
      |
      v
Governança LGPD + análise causal + model card
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
│   ├── causal_analysis.md
│   ├── data_dictionary.md
│   ├── lgpd_privacy_plan.md
│   ├── model_card.md
│   ├── monitoring_policy.md
│   └── video_star_script.md
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

## Etapa 4 — Governança, LGPD e causalidade

A Etapa 4 documenta como o sistema deve tratar dados pessoais, decisões automatizadas, retenção, segurança, vieses e responsabilidades. O projeto é acadêmico e não está autorizado a conceder crédito real.

### Inventário e classificação dos dados

O OpenML 43454 não contém nome, CPF, endereço, telefone ou e-mail. Entretanto, idade, renda, emprego, histórico de crédito e condições do empréstimo podem ser dados pessoais ou quase-identificadores quando associados a uma pessoa em produção.

Dados financeiros recebem proteção elevada, mas não são classificados automaticamente como dados pessoais sensíveis na definição estrita da LGPD. O dicionário completo está em [`governance/data_dictionary.md`](governance/data_dictionary.md).

### Finalidade e base legal

As finalidades são avaliar o risco de inadimplência, monitorar o modelo, investigar falhas e manter evidências de auditoria. A base legal principal proposta é a proteção do crédito, prevista no art. 7º, X, da LGPD. Procedimentos preliminares relacionados a contrato podem também se aplicar conforme o art. 7º, V.

Consentimento não é adotado como fundamento genérico para toda análise de crédito. Usos secundários exigem avaliação própria de finalidade, necessidade e base legal.

### Privacy by Design e by Default

- minimização das features e ausência de identificadores diretos no modelo;
- privacidade como configuração padrão;
- separação entre identidade e atributos analíticos;
- criptografia em trânsito e em repouso;
- acesso pelo menor privilégio;
- logs sem PII, tokens ou segredos;
- rastreabilidade de dados, modelo, configuração e decisão;
- revisão periódica de necessidade, retenção, drift e vieses.

### Retenção proposta

| Categoria | Prazo proposto | Destinação |
|---|---:|---|
| cache do dataset bruto acadêmico | 90 dias | exclusão e novo download quando necessário |
| referência pseudonimizada | versão ativa + 180 dias | exclusão ou anonimização |
| inferências individuais | 180 dias | agregação e exclusão individual |
| logs técnicos sem PII | 12 meses | exclusão segura |
| relatórios agregados | 24 meses | arquivamento ou exclusão |
| modelo, métricas e model card | 5 anos | revisão e descarte controlado |

Esses prazos são uma política acadêmica proposta, não prazos legais universais. Uma implantação real deve conciliá-los com requisitos regulatórios e políticas corporativas.

### Direitos e decisões automatizadas

O processo deve permitir confirmação de tratamento, acesso, correção, informação sobre compartilhamento e demais direitos aplicáveis. Decisões exclusivamente automatizadas devem admitir solicitação de revisão e explicação clara dos critérios e procedimentos utilizados.

Uma contestação deve gerar protocolo, preservar as evidências da decisão e ser encaminhada a uma pessoa qualificada. O fluxo completo está em [`governance/lgpd_privacy_plan.md`](governance/lgpd_privacy_plan.md).

### Governança e vieses

- avaliar desempenho por grupos quando houver base legal e controles adequados;
- investigar features que possam funcionar como proxies de grupos protegidos;
- exigir revisão humana em decisões contestadas;
- impedir retreinamento e promoção automáticos;
- comparar e aprovar formalmente cada nova versão;
- manter possibilidade de rollback;
- registrar incidentes, causas e ações preventivas.

Consulte o [`model card`](governance/model_card.md) e a [`política de monitoramento`](governance/monitoring_policy.md).

### Análise causal

PSI e Kolmogorov-Smirnov evidenciam mudança de distribuição, mas não demonstram causalidade. A queda de recall ocorreu junto com alterações em renda, taxa de juros e comprometimento da renda no cenário simulado. Em produção real, inflação, desemprego, política de concessão, seleção de clientes e mix de produtos poderiam atuar como causas ou confundidores.

Portanto, o projeto não afirma que uma variável econômica específica causou a degradação. A investigação recomendada utiliza coortes temporais, análise por segmentos, registro da política de crédito e testes de robustez. Consulte [`governance/causal_analysis.md`](governance/causal_analysis.md).

### Vídeo STAR

O roteiro de até cinco minutos está em [`governance/video_star_script.md`](governance/video_star_script.md) e cobre:

- **Situation:** risco da degradação silenciosa;
- **Task:** qualidade, drift, observabilidade e compliance;
- **Action:** Pandera, Evidently, PSI, KS, logs, alertas e governança;
- **Result:** lote bloqueado, drift detectado, queda de recall e controles LGPD.

Link do vídeo: **a adicionar após a gravação**.

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

### Etapa 4

```text
governance/causal_analysis.md
governance/data_dictionary.md
governance/lgpd_privacy_plan.md
governance/model_card.md
governance/monitoring_policy.md
governance/video_star_script.md
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
- `v0.3.0`: observabilidade, logs estruturados, métricas Prometheus, alertas e dashboard.
- `v0.4.0` (próxima versão): governança, LGPD, causalidade e roteiro STAR.

## Próximas etapas

- gravar e publicar o vídeo STAR de até cinco minutos;
- substituir o campo provisório pelo link do vídeo;
- executar a revisão final dos entregáveis;
- publicar a versão `v0.4.0` após o aceite da documentação.

## Autor

**Rafael Alves da Costa**

GitHub: [RafaExMachina](https://github.com/RafaExMachina)
