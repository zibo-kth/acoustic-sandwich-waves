"""config_io.py

Load model/solver configuration from JSON or YAML.

- JSON: built-in
- YAML: requires PyYAML (optional)

The expected config shape is intentionally simple and flat.
See `examples/config.example.yml`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".json"}:
        return json.loads(path.read_text(encoding="utf-8"))

    if suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "YAML config requires PyYAML. Install with: python3 -m pip install pyyaml"
            ) from e
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    raise ValueError(f"Unsupported config extension: {suffix} (use .json/.yml/.yaml)")
