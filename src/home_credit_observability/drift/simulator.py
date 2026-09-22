"""Simulação determinística de um lote de produção com data drift."""

from __future__ import annotations

import pandas as pd


class ProductionDataSimulator:
    """Cria dados atuais e altera variáveis financeiras de forma controlada."""

    def __init__(
        self,
        *,
        sample_size: int = 8_000,
        random_state: int = 42,
        income_multiplier: float = 1.35,
        interest_rate_shift: float = 3.0,
    ) -> None:
        """Configura tamanho, semente e intensidade da simulação."""
        if sample_size < 1:
            raise ValueError("sample_size deve ser positivo")
        if income_multiplier <= 0:
            raise ValueError("income_multiplier deve ser positivo")
        self.sample_size = sample_size
        self.random_state = random_state
        self.income_multiplier = income_multiplier
        self.interest_rate_shift = interest_rate_shift

    def simulate(self, reference: pd.DataFrame) -> pd.DataFrame:
        """Amostra registros e aplica drift em renda, juros e comprometimento."""
        if reference.empty:
            raise ValueError("o dataset de referência não pode ser vazio")
        current = reference.sample(
            n=self.sample_size,
            replace=self.sample_size > len(reference),
            random_state=self.random_state,
        ).reset_index(drop=True)
        current["person_income"] = current["person_income"] * self.income_multiplier
        current["loan_int_rate"] = current["loan_int_rate"] + self.interest_rate_shift
        current["loan_percent_income"] = (
            current["loan_amnt"] / current["person_income"]
        ).clip(0, 1)
        return current
