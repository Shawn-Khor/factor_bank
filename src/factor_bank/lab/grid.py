"""Candidate grid: numeric base factors × the 8 transform registry keys."""
from __future__ import annotations

from factor_bank.engine.catalog import numeric_base_factors
from factor_bank.engine.factors import TRANSFORMS


def candidate_grid(include_custom: bool = True) -> list[str]:
    bases = list(numeric_base_factors())
    if include_custom:
        from factor_bank.data.custom import custom_names
        bases += custom_names()
    return [f"{base}__{t}" for base in bases for t in TRANSFORMS]
