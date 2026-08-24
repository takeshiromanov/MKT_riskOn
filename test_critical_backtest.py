import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from critical_backtest import (
    BacktestConfig,
    calibration_table,
    drawdown_episodes,
    execute_backtest,
)


def _synthetic_prices(months: int = 144) -> pd.DataFrame:
    index = pd.date_range("2010-01-31", periods=months, freq="ME")
    phase = np.arange(months)
    equity_returns = 0.008 + 0.035 * np.sin(phase / 5) - 0.02 * (
        (phase >= 55) & (phase <= 62)
    )
    cash_returns = np.full(months, 0.0015)
    return pd.DataFrame(
        {
            "EQUITY": 100 * np.cumprod(1 + equity_returns),
            "CASH": 100 * np.cumprod(1 + cash_returns),
        },
        index=index,
    )


class CriticalBacktestTests(unittest.TestCase):
    def test_drawdown_episode_dates(self):
        index = pd.date_range("2020-01-31", periods=6, freq="ME")
        prices = pd.Series([100, 110, 90, 95, 111, 115], index=index)
        episodes = drawdown_episodes(prices)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes.iloc[0]["peak_date"], index[1])
        self.assertEqual(episodes.iloc[0]["trough_date"], index[2])
        self.assertEqual(episodes.iloc[0]["recovery_date"], index[4])
        self.assertAlmostEqual(episodes.iloc[0]["drawdown"], 90 / 110 - 1)

    def test_calibration_uses_only_future_prices(self):
        prices = _synthetic_prices(48)
        signal = pd.Series(0.5, index=prices.index)
        calibration = calibration_table(prices, signal)
        middle = calibration.loc[calibration["exposure_bin"] == "40-60%"].iloc[0]
        expected = (
            (prices["EQUITY"].shift(-1) / prices["EQUITY"])
            / (prices["CASH"].shift(-1) / prices["CASH"])
            - 1
        ).mean()
        self.assertAlmostEqual(middle["mean_excess_1m"], expected)

    def test_end_to_end_writes_audit_outputs(self):
        prices = _synthetic_prices()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            metrics = execute_backtest(
                prices,
                "synthetic unit test",
                output,
                BacktestConfig(),
            )
            self.assertEqual(
                set(metrics.index),
                {"Buy & Hold", "Binary 12m", "Graduated 3/6/12", "Continuous L1"},
            )
            for filename in (
                "strategy_metrics.csv",
                "calibration.csv",
                "stress_events.csv",
                "robustness_grid.csv",
                "robustness_summary.csv",
                "manifest.json",
                "critical_backtest_summary.md",
                "backtest_diagnostics.png",
                "calibration.png",
            ):
                self.assertTrue((output / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
