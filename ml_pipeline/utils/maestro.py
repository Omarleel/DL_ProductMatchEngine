from __future__ import annotations

from pathlib import Path

import pandas as pd


MAESTRO_FILENAME = "maestro.csv"


def leer_maestro_csv(path: str | Path) -> pd.DataFrame:
    """Lectura estándar de maestro.csv para API y scripts."""
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        sep=None,
        engine="python",
    )
