from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ETH_BTC_MTF_LIVE import compute_atr, compute_rsi


def test_compute_rsi_matches_manual():
    closes = pd.Series([44, 47, 46, 48, 50], dtype=float)
    rsi = compute_rsi(closes, length=3)

    # Expected values manually derived using rolling average gains/losses over 3 periods
    assert rsi.iloc[3] == pytest.approx(83.3333, rel=1e-3)
    assert rsi.iloc[4] == pytest.approx(80.0, rel=1e-3)


def test_compute_atr_ewm_matches_manual():
    df = pd.DataFrame(
        {
            "high": [10, 12, 11],
            "low": [8, 9, 10],
            "close": [9, 11, 10],
        }
    )
    atr = compute_atr(df, length=3)
    # TR values: [2, 3, 1] with EWM(span=3, adjust=False)
    expected = pd.Series([2.0, 2.5, 1.75])
    assert atr.round(4).tolist() == pytest.approx(expected.round(4).tolist())
