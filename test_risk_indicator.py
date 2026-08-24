import numpy as np
import pandas as pd

from risk_indicator import continuous_absolute_momentum


def _prices(equity_monthly_return: float, cash_monthly_return: float) -> pd.DataFrame:
    index = pd.date_range("2010-01-31", periods=72, freq="ME")
    return pd.DataFrame(
        {
            "EQUITY": 100 * np.cumprod(np.full(len(index), 1 + equity_monthly_return)),
            "CASH": 100 * np.cumprod(np.full(len(index), 1 + cash_monthly_return)),
        },
        index=index,
    )


def test_strong_positive_trend_produces_high_target():
    result = continuous_absolute_momentum(_prices(0.02, 0.002), "EQUITY", "CASH")
    assert result["target_weight"].dropna().iloc[-1] >= 0.90


def test_negative_trend_produces_low_target():
    result = continuous_absolute_momentum(_prices(-0.01, 0.002), "EQUITY", "CASH")
    assert result["target_weight"].dropna().iloc[-1] <= 0.10


def test_weights_are_normalized_automatically():
    prices = _prices(0.01, 0.002)
    first = continuous_absolute_momentum(
        prices,
        "EQUITY",
        "CASH",
        lookback_weights=(2, 3, 5),
    )
    second = continuous_absolute_momentum(
        prices,
        "EQUITY",
        "CASH",
        lookback_weights=(0.2, 0.3, 0.5),
    )
    pd.testing.assert_series_equal(first["target_weight"], second["target_weight"])
