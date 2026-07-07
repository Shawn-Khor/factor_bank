import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_market():
    """12 tickers × ~2 years. Ticker i drifts at daily rate r_i (+ small noise
    so the daily IC series isn't degenerate at exactly 1.0), and the 'pe'
    column equals r_i — a near-perfect, known-answer cross-sectional signal."""
    n = 500
    dates = pd.bdate_range("2018-06-01", periods=n)
    rates = np.linspace(-0.002, 0.002, 12)
    frames = []
    for i, r in enumerate(rates):
        rng = np.random.default_rng(100 + i)
        # Noise must be large enough to occasionally perturb the cross-sectional
        # rank of forward returns (else rank_ic is stuck at exactly 1.0 every
        # day -> std_ic == 0.0 -> t_stat is None -> verdict degenerates to
        # INSIGNIFICANT despite a near-perfect signal), yet small enough that
        # drift still dominates (rank_ic stays > 0.9).
        prices = 100.0 * np.cumprod(1.0 + r + rng.normal(0, 8e-4, n))
        frames.append(pd.DataFrame({
            "ticker": f"T{i:02d}",
            "date": dates,
            "sector": "Tech" if i % 2 == 0 else "Energy",
            "prev_close_price": prices,
            "pe": r,                       # factor == future return driver
            "marketcap": 1e9 * (i + 1),
            "ev": 1.2e9 * (i + 1),
            "evebitda": 10.0 + i,
            "pb": 2.0,
            "ps": 3.0,
            "evebit": 11.0,
        }))
    enriched = pd.concat(frames, ignore_index=True).sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)
    spells = pd.DataFrame({
        "ticker": [f"T{i:02d}" for i in range(12)],
        "start": pd.Timestamp("2018-01-01"),
        "end": pd.Timestamp.max,
    })
    return enriched, spells


@pytest.fixture
def market_with_noise(synthetic_market):
    """pe and evebitda are (redundant) true signals in the fixture; overwrite
    ps with pure noise so screening has something to rank last."""
    enriched, spells = synthetic_market
    enriched = enriched.copy()
    rng = np.random.default_rng(5)
    enriched["ps"] = rng.normal(0, 1, len(enriched))
    return enriched, spells
