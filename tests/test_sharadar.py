import numpy as np
import pandas as pd

from factor_bank.data.sharadar import ticker_to_permaticker


def test_permaticker_basic_and_related():
    df = pd.DataFrame({
        "table": ["SEP", "SEP", "SEP"],
        "ticker": ["META", "AAPL", "BADROW"],
        "permaticker": [111, 222, np.nan],
        "relatedtickers": ["FB", None, "X"],
    })
    m = ticker_to_permaticker(df)
    assert m["META"] == 111
    assert m["FB"] == 111        # historical alias maps to same id
    assert m["AAPL"] == 222
    assert "BADROW" not in m     # NaN permaticker dropped
    assert "X" not in m          # alias of dropped row not mapped


def test_permaticker_alias_does_not_clobber_primary():
    df = pd.DataFrame({
        "table": ["SEP", "SEP"],
        "ticker": ["A", "B"],
        "permaticker": [1, 2],
        "relatedtickers": [None, "A"],  # B lists A as related — A's own id wins
    })
    m = ticker_to_permaticker(df)
    assert m["A"] == 1
    assert m["B"] == 2


def test_permaticker_filters_non_sep_when_table_present():
    df = pd.DataFrame({
        "table": ["SFP", "SEP"],
        "ticker": ["FUND", "STK"],
        "permaticker": [9, 10],
        "relatedtickers": [None, None],
    })
    m = ticker_to_permaticker(df)
    assert "FUND" not in m and m["STK"] == 10
