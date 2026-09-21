# Credit Risk Observability - Etapa 1

Pipeline de validação e classificação binária financeira utilizando o **Credit
Risk Dataset do OpenML, ID 43454**. O download é público e não exige conta ou
token do Kaggle.

## Fluxo

```text
OpenML 43454 -> limpeza determinística -> contrato Pandera -> baseline
                                         -> bloqueio se inválido
```

O dataset possui aproximadamente 32 mil registros e o alvo `loan_status`:

- `0`: não inadimplente;
- `1`: inadimplente.

## Execução

```bash
uv sync --locked
uv run python -m home_credit_observability.cli download
uv run python -m home_credit_observability.cli validate
uv run python -m home_credit_observability.cli validate-invalid
uv run python -m home_credit_observability.cli train
```

## Qualidade

```bash
uv run pytest -v
uv run ruff check .
```

## Artefatos gerados

- `data/raw/credit_risk_openml_43454.csv`
- `data/reference/reference.parquet`
- `data/samples/invalid_batch.csv`
- `reports/data_quality/invalid_batch_failures.csv`
- `artifacts/models/baseline_pipeline.joblib`
- `artifacts/metrics/baseline_metrics.json`
