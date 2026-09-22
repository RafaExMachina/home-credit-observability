"""Testes da simulação do dataset de produção."""

import pandas as pd

from home_credit_observability.drift.simulator import ProductionDataSimulator


def _reference() -> pd.DataFrame:
    """Cria uma referência mínima para testes."""
    return pd.DataFrame(
        {
            "person_income": [10_000.0, 20_000.0, 30_000.0],
            "loan_int_rate": [8.0, 9.0, 10.0],
            "loan_amnt": [1_000.0, 2_000.0, 3_000.0],
            "loan_percent_income": [0.1, 0.1, 0.1],
        }
    )


def test_simulator_is_reproducible_and_preserves_row_count() -> None:
    """A mesma semente deve gerar exatamente o mesmo lote."""
    simulator = ProductionDataSimulator(sample_size=10, random_state=7)
    first = simulator.simulate(_reference())
    second = simulator.simulate(_reference())
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 10


def test_simulator_changes_financial_distributions() -> None:
    """A simulação deve elevar renda e taxa de juros."""
    simulator = ProductionDataSimulator(sample_size=3, random_state=2)
    output = simulator.simulate(_reference())
    sampled_original = _reference().sample(n=3, random_state=2).reset_index(drop=True)
    assert output["person_income"].mean() > sampled_original["person_income"].mean()
    assert output["loan_int_rate"].mean() > sampled_original["loan_int_rate"].mean()
