"""Testes do logger estruturado."""

import json

from home_credit_observability.observability.logger import JsonLinesEventLogger


def test_json_logger_writes_structured_event(tmp_path):
    """O logger deve persistir evento, nível e atributos em JSONL."""
    path = tmp_path / "events.jsonl"
    JsonLinesEventLogger(path).log("model_scored", rows=10)

    event = json.loads(path.read_text(encoding="utf-8"))

    assert event["event"] == "model_scored"
    assert event["level"] == "INFO"
    assert event["attributes"]["rows"] == 10
    assert event["timestamp_utc"].endswith("+00:00")
