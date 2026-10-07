"""Logger estruturado em JSON Lines para eventos do pipeline."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonLinesEventLogger:
    """Persiste um objeto JSON por linha para facilitar busca e agregação."""

    def __init__(self, output_path: Path) -> None:
        """Configura o arquivo que centralizará os eventos."""
        self.output_path = output_path

    def log(
        self,
        event: str,
        *,
        level: str = "INFO",
        **attributes: Any,
    ) -> None:
        """Acrescenta um evento com timestamp UTC ao arquivo JSONL."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "level": level.upper(),
            "event": event,
            "attributes": attributes,
        }
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
            stream.write("\n")
