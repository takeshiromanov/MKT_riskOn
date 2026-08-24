import unittest

import numpy as np
import pandas as pd

from risk_indicator import (
    backtest_allocation,
    continuous_absolute_momentum,
    continuous_absolute_momentum_recovery,
    continuous_absolute_momentum_recovery_one_shot,
)


def _prices(equity_monthly_return: float, cash_monthly_return: float) -> pd.DataFrame:
    index = pd.date_range("2010-01-31", periods=72, freq="ME")
    return pd.DataFrame(
        {
            "EQUITY": 100 * np.cumprod(np.full(len(index), 1 + equity_monthly_return)),
            "CASH": 100 * np.cumprod(np.full(len(index), 1 + cash_monthly_return)),
        },
        index=index,
    )


def _v_reversal_prices() -> pd.DataFrame:
    index = pd.date_range("2010-01-31", periods=100, freq="ME")
    equity_returns = np.r_[
        np.full(50, 0.012),
        np.full(8, -0.08),
        np.full(12, 0.07),
        np.full(30, 0.012),
    ]
    cash_returns = np.full(len(index), 0.0015)
    return pd.DataFrame(
        {
            "EQUITY": 100 * np.cumprod(1 + equity_returns),
            "CASH": 100 * np.cumprod(1 + cash_returns),
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

    def test_recovery_overlay_preserves_the_down_path(self):
        prices = _v_reversal_prices()
        baseline = continuous_absolute_momentum(
            prices, "EQUITY", "CASH"
        )["target_weight"]
        recovery = continuous_absolute_momentum_recovery(
            prices, "EQUITY", "CASH"
        )["target_weight"]
        pd.testing.assert_series_equal(
            baseline.iloc[50:58], recovery.iloc[50:58], check_names=False
        )

    def test_recovery_overlay_accelerates_the_rebound(self):
        prices = _v_reversal_prices()
        baseline = continuous_absolute_momentum(
            prices, "EQUITY", "CASH"
        )["target_weight"]
        recovery = continuous_absolute_momentum_recovery(
            prices, "EQUITY", "CASH"
        )
        rebound = prices.index[58:64]
        self.assertGreater(
            recovery.loc[rebound, "target_weight"].mean(),
            baseline.loc[rebound].mean(),
        )
        self.assertTrue(recovery.loc[rebound, "recovery_mode"].any())

    def test_recovery_signal_has_no_future_data_dependency(self):
        prices = _v_reversal_prices()
        cutoff = prices.index[70]
        altered = prices.copy()
        altered.loc[altered.index > cutoff, "EQUITY"] *= 3
        first = continuous_absolute_momentum_recovery(
            prices, "EQUITY", "CASH"
        )["target_weight"]
        second = continuous_absolute_momentum_recovery(
            altered, "EQUITY", "CASH"
        )["target_weight"]
        pd.testing.assert_series_equal(
            first.loc[:cutoff], second.loc[:cutoff], check_names=False
        )

    def test_one_shot_matches_core_before_its_first_trigger(self):
        prices = _v_reversal_prices()
        baseline = continuous_absolute_momentum(
            prices, "EQUITY", "CASH"
        )["target_weight"]
        one_shot = continuous_absolute_momentum_recovery_one_shot(
            prices, "EQUITY", "CASH"
        )
        trigger_dates = one_shot.index[one_shot["probe_active"]]
        self.assertEqual(len(trigger_dates), 1)
        before_trigger = one_shot.index < trigger_dates[0]
        pd.testing.assert_series_equal(
            baseline.loc[before_trigger],
            one_shot.loc[before_trigger, "target_weight"],
            check_names=False,
        )

    def test_one_shot_accelerates_rebound_without_rearming_same_episode(self):
        prices = _v_reversal_prices()
        baseline = continuous_absolute_momentum(
            prices, "EQUITY", "CASH"
        )["target_weight"]
        one_shot = continuous_absolute_momentum_recovery_one_shot(
            prices, "EQUITY", "CASH"
        )
        rebound = prices.index[58:64]
        self.assertGreater(
            one_shot.loc[rebound, "target_weight"].mean(),
            baseline.loc[rebound].mean(),
        )
        self.assertEqual(int(one_shot["probe_active"].sum()), 1)

    def test_one_shot_signal_has_no_future_data_dependency(self):
        prices = _v_reversal_prices()
        cutoff = prices.index[70]
        altered = prices.copy()
        altered.loc[altered.index > cutoff, "EQUITY"] *= 3
        first = continuous_absolute_momentum_recovery_one_shot(
            prices, "EQUITY", "CASH"
        )["target_weight"]
        second = continuous_absolute_momentum_recovery_one_shot(
            altered, "EQUITY", "CASH"
        )["target_weight"]
        pd.testing.assert_series_equal(
            first.loc[:cutoff], second.loc[:cutoff], check_names=False
        )


if __name__ == "__main__":
    unittest.main()
