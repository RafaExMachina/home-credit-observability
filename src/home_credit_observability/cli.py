from __future__ import annotations

import argparse
import json

from home_credit_observability.config import SAMPLES_DIR
from home_credit_observability.data.download import download_credit_risk
from home_credit_observability.data.load import load_application_data
from home_credit_observability.data.prepare import (
    clean_reference_data,
    select_model_columns,
)
from home_credit_observability.pipelines.train_pipeline import run_training_pipeline
from home_credit_observability.validation.schema import make_invalid_batch
from home_credit_observability.validation.validator import (
    DataQualityError,
    validate_credit_risk,
)


def _reference_data():
    path = download_credit_risk()
    raw = load_application_data(path)
    return clean_reference_data(select_model_columns(raw))


def _download(force: bool) -> int:
    print(download_credit_risk(force=force))
    return 0


def _validate() -> int:
    validated = validate_credit_risk(_reference_data())
    print(f"Contrato aprovado para {len(validated):,} registros.")
    return 0


def _validate_invalid() -> int:
    invalid = make_invalid_batch(_reference_data())
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    invalid_path = SAMPLES_DIR / "invalid_batch.csv"
    invalid.to_csv(invalid_path, index=False)
    try:
        validate_credit_risk(
            invalid, min_rows=1, report_name="invalid_batch_failures.csv"
        )
    except DataQualityError as exc:
        print(exc)
        print("Resultado esperado: o lote inválido foi bloqueado.")
        return 0
    print("ERRO: o lote inválido foi aceito.")
    return 1


def _train(force: bool) -> int:
    result = run_training_pipeline(force_download=force)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Credit Risk Observability")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "validate", "validate-invalid", "train"):
        subparser = subparsers.add_parser(name)
        if name in {"download", "train"}:
            subparser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "download":
        return _download(args.force)
    if args.command == "validate":
        return _validate()
    if args.command == "validate-invalid":
        return _validate_invalid()
    if args.command == "train":
        return _train(args.force)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
