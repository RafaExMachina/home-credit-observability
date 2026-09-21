from home_credit_observability.data.download import download_credit_risk
from home_credit_observability.data.load import load_application_data
from home_credit_observability.data.prepare import (
    clean_reference_data,
    select_model_columns,
)
from home_credit_observability.validation.validator import validate_credit_risk

if __name__ == "__main__":
    path = download_credit_risk()
    data = select_model_columns(load_application_data(path))
    validated = validate_credit_risk(clean_reference_data(data))
    print(f"Contrato aprovado para {len(validated):,} registros.")
