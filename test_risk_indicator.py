import unittest

import numpy as np
import pandas as pd

from risk_indicator import backtest_allocation, continuous_absolute_momentum


def _prices(equity_monthly_return: float, cash_monthly_return: float) -> pd.DataFrame:
    index = pd.date_range("2010-01-31", periods=72, freq="ME")
    return pd.DataFrame(
        {
            "EQUITY": 100 * np.cumprod(np.full(len(index), 1 + equity_monthly_return)),
            "CASH": 100 * np.cumprod(np.full(len(index), 1 + cash_monthly_return)),
        },
        index=index,
    )


class RiskIndicatorTests(unittest.TestCase):
    def test_strong_positive_trend_produces_high_target(self):
        result = continuous_absolute_momentum(
            _prices(0.02, 0.002), "EQUITY", "CASH"
        )
        self.assertGreaterEqual(result["target_weight"].dropna().iloc[-1], 0.90)

    def test_negative_trend_produces_low_target(self):
        result = continuous_absolute_momentum(
            _prices(-0.01, 0.002), "EQUITY", "CASH"
        )
        self.assertLessEqual(result["target_weight"].dropna().iloc[-1], 0.10)

    def test_weights_are_normalized_automatically(self):
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

    def test_target_is_lagged_one_month(self):
        index = pd.date_range("2020-01-31", periods=5, freq="ME")
        prices = pd.DataFrame(
            {
                "EQUITY": [100, 110, 121, 133.1, 146.41],
                "CASH": [100, 100, 100, 100, 100],
            },
            index=index,
        )
        target = pd.Series([np.nan, 0.0, 1.0, 1.0, 1.0], index=index)
        result = backtest_allocation(
            prices, "EQUITY", "CASH", target, transaction_cost_bps=0
        )
        self.assertEqual(result.loc[index[2], "target_weight"], 0.0)
        self.assertAlmostEqual(result.loc[index[2], "net_return"], 0.0)
        self.assertEqual(result.loc[index[3], "target_weight"], 1.0)
        self.assertAlmostEqual(result.loc[index[3], "net_return"], 0.10)

    def test_turnover_includes_drift_rebalancing(self):
        index = pd.date_range("2020-01-31", periods=4, freq="ME")
        prices = pd.DataFrame(
            {
                "EQUITY": [100, 110, 121, 133.1],
                "CASH": [100, 100, 100, 100],
            },
            index=index,
        )
        target = pd.Series(0.5, index=index)
        result = backtest_allocation(
            prices, "EQUITY", "CASH", target, transaction_cost_bps=0
        )
        self.assertAlmostEqual(result.loc[index[1], "turnover"], 0.5)
        self.assertGreater(result.loc[index[2], "turnover"], 0.0)


if __name__ == "__main__":
    unittest.main()
