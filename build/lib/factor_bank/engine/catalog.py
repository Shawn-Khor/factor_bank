"""Factor catalog — grouped registry of all known factor names."""
from __future__ import annotations

FACTOR_CATALOG: dict[str, dict[str, str]] = {
    # ─── Vendor groups (legacy — carried over from the prior dashboard) ─
    "Seeking Alpha": {
        "sa_quant_rating":        "SA Quant rating (1-5, higher = more bullish)",
        "sa_wall_street_rating":  "SA Wall Street analyst consensus",
        "sa_analyst_rating":      "SA author analyst rating",
        "sa_momentum":            "SA momentum grade",
        "sa_revisions":           "SA revisions grade",
        "sa_value":               "SA value grade",
        "sa_growth":              "SA growth grade",
        "sa_profitability":       "SA profitability grade",
    },
    "Zacks": {
        "zack_rating":   "Zacks rank, inverted (5=Strong Buy, 1=Strong Sell)",
        "zack_value":    "Zacks value score (higher = better)",
        "zack_growth":   "Zacks growth score",
        "zack_momentum": "Zacks momentum score",
        "zack_vgm":      "Zacks VGM composite",
    },
    "IBD": {
        "ibd_rs_rank":   "IBD Relative Strength rank (0-100)",
        "ibd_trend_ord": "Trend ordinal: BULL=2, MIXED_UP=1, MIXED_DOWN=-1, BEAR=-2",
        "ibd_state_ord": "State ordinal: RECOVERY=2, AVERAGE=1, DETERIORATION=-1, COLLAPSE=-2",
        "ibd_risk_ord":  "Risk ordinal: LOW=1, MED=0, HIGH=-1",
    },
    "Technicals": {
        "beta":               "Market beta",
        "kalman_beta":        "Kalman-filtered beta",
        "rsi_14":             "14-day RSI",
        "macd":               "MACD line",
        "macd_signal":        "MACD signal line",
        "macd_hist":          "MACD histogram (macd - signal)",
        "bb_pct":             "Bollinger %B",
        "bb_width":           "Bollinger band width %",
        "atr_pct":            "ATR(14) as % of close",
        "stoch_k":            "Stochastic %K (14d)",
        "stoch_d":            "Stochastic %D (3d SMA of %K)",
        "adx_14":             "14-day ADX (trend strength)",
        "volume_ratio":       "volume / 20d-avg-volume",
        "dist_sma_50":        "(close - SMA50) / SMA50",
        "dist_sma_200":       "(close - SMA200) / SMA200",
        "pct_from_52w_high":  "(close - 52w high) / 52w high",
        "pct_from_52w_low":   "(close - 52w low) / 52w low",
    },
    "Returns / Vol": {
        "ret_5d":          "5-day return",
        "ret_10d":         "10-day return",
        "ret_20d":         "20-day return",
        "ret_60d":         "60-day (quarterly) return",
        "ret_120d":        "120-day return",
        "ret_252d":        "252-day (annual) return",
        "volatility_20d":  "20-day annualised vol",
        "volatility_60d":  "60-day annualised vol",
    },
    # ─── New factor-family groups (Sharadar daily values) ────────────────
    "Value": {
        "pe":              "Price/Earnings (raw)",
        "pb":              "Price/Book (raw)",
        "ps":              "Price/Sales (raw)",
        "evebitda":        "EV/EBITDA (raw)",
        "earnings_yield":  "1/PE — loss-maker safe",
        "book_yield":      "1/PB",
        "sales_yield":     "1/PS",
        "ebitda_yield":    "1/EV/EBITDA — practitioner favorite",
    },
    "Capital Structure": {
        "ev_to_mcap":      "EV/MarketCap (>1=debt, <1=cash)",
    },
    "Size": {
        "log_marketcap":   "log(MarketCap)",
    },
    "Re-rating": {
        "pe_change_30d":   "PE - PE.shift(30) per ticker",
    },
    "Value × TS": {
        "pe_zscore_252d":              "Rolling 252d z-score of PE per ticker",
        "ebitda_yield_zscore_252d":    "Rolling 252d z-score of 1/EV/EBITDA",
        "pb_percentile_252d":          "Rolling 252d rank percentile of PB",
    },
    "Value × XS": {
        "evebitda_sector_relative":    "EV/EBITDA minus sector median",
    },
}


def all_factor_names() -> list[str]:
    return [f for group in FACTOR_CATALOG.values() for f in group]


def numeric_base_factors() -> list[str]:
    return [f for f in all_factor_names() if not f.startswith("ibd_") or f == "ibd_rs_rank"]
