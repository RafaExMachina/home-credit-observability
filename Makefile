.PHONY: sync test lint check download validate invalid train

sync:
	uv sync --locked

test:
	uv run pytest -v

lint:
	uv run ruff check .

check: lint test

download:
	uv run python -m home_credit_observability.cli download

validate:
	uv run python -m home_credit_observability.cli validate

invalid:
	uv run python -m home_credit_observability.cli validate-invalid

train:
	uv run python -m home_credit_observability.cli train
