"""Dynaconf-based configuration loader with environment layering."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dynaconf import Dynaconf

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


@lru_cache(maxsize=1)
def get_config() -> Dynaconf:
    """Return the singleton Dynaconf settings object.

    Layering order (later wins):
      1. config/default.toml
      2. config/{QUANTCORE_ENV}.toml  (e.g. development.toml)
      3. Environment variables prefixed with QUANTCORE_  (e.g. QUANTCORE_DATABASE__HOST)
    """
    env = __import__("os").getenv("QUANTCORE_ENV", "development")
    return Dynaconf(
        env=env,
        root_path=str(_CONFIG_DIR),
        settings_files=["default.toml", f"{env}.toml"],
        envvar_prefix="QUANTCORE",
        merge_enabled=True,
        default_env="development",
    )
