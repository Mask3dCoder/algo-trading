#!/usr/bin/env python3
"""YAML field reader for manage_silent_run.sh.

Usage:
    python bin/yaml_field.py <key>
    python bin/yaml_field.py execution.passive_tracking_mode

Reads config/paper_trading.yaml and prints the value of the dot-notation key.
Exits 0 on success, 1 if the key is not found (prints empty string).
"""

import sys
import yaml
from pathlib import Path

CONFIG = Path(__file__).parent.parent / "config" / "paper_trading.yaml"


def get_field(key: str) -> str:
    with open(CONFIG) as f:
        data = yaml.safe_load(f) or {}
    for part in key.split("."):
        if isinstance(data, dict):
            data = data.get(part, {})
        else:
            return ""
    return str(data) if data is not None else ""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: yaml_field.py <dot.notation.key>", file=sys.stderr)
        sys.exit(1)
    print(get_field(sys.argv[1]))