"""Backtest critico e riproducibile del Layer 1.

L'obiettivo non e trovare i parametri con il CAGR migliore. Il programma
confronta il modello continuo con benchmark semplici e prova a falsificarlo
su cinque dimensioni: drawdown, ritardo del segnale, whipsaw, calibrazione
dell'esposizione e stabilita rispetto a parametri vicini.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from risk_indicator import (
    backtest_allocation,
    binary_signal,
    continuous_absolute_momentum,
    continuous_absolute_momentum_recovery,
    momentum_score,
)


@dataclass(frozen=True)
class BacktestConfig:
    equity: str = "URTH"
    cash: str = "BIL"
    start: str = "2012-01-01"
    transaction_cost_bps: float = 10.0
    lookback_periods: tuple[int, ...] = (3, 6, 12)
    lookback_weights: tuple[float, ...] = (0.20, 0.30, 0.50)
    volatility_window_months: int = 12
    transition_width: float = 0.75
    alpha_up: float = 0.30
    alpha_down: float = 0.60
    round_to: float = 0.05
    probe_size: float = 0.20
    recovery_alpha_up: float = 0.60
    recovery_step_cap: float = 0.20
    recovery_activation_ceiling: float = 0.50
    recovery_target_ceiling: float = 0.70


def load_prices(
    equity: str,
    cash: str,
    start: str,
    prices_csv: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """Carica prezzi adjusted mensili da CSV oppure da Yahoo Finance."""
    if prices_csv is not None:
        raw = pd.read_csv(prices_csv)
        columns = {column.upper(): column for column in raw.columns}
        required = {"DATE", "EQUITY", "CASH"}
        missing = required.difference(columns)
        if missing:
            raise ValueError(
                "Il CSV deve contenere DATE, EQUITY e CASH; mancano "
                f"{sorted(missing)}"
            )
        prices = raw[[columns["DATE"], columns["EQUITY"], columns["CASH"]]].copy()
        prices.columns = ["DATE", "EQUITY", "CASH"]
        prices["DATE"] = pd.to_datetime(prices["DATE"], errors="raise")
        prices = prices.set_index("DATE").sort_index()
        source = f"CSV locale: {prices_csv}"
    else:
        import yfinance as yf

        downloaded = yf.download(
            [equity, cash],
            start=start,
            auto_adjust=True,
            actions=False,
            progress=False,
            threads=False,
        )
        if downloaded.empty or "Close" not in downloaded:
            raise RuntimeError("Yahoo Finance non ha restituito prezzi utilizzabili")
        close = downloaded["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame()
        prices = close.rename(columns={equity: "EQUITY", cash: "CASH"})
        source = f"Yahoo Finance adjusted close: {equity}/{cash}"

    prices = prices[["EQUITY", "CASH"]].apply(pd.to_numeric, errors="coerce")
    prices = prices.resample("ME").last().dropna(how="any")
    today_utc = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    last_complete_month_end = today_utc.to_period("M").start_time - pd.Timedelta(days=1)
    prices = prices.loc[prices.index <= last_complete_month_end]
    if prices.index.has_duplicates:
        raise ValueError("Lo storico contiene date mensili duplicate")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Le date non sono ordinate")
    if (prices <= 0).any().any():
        raise ValueError("I prezzi devono essere strettamente positivi")
    if len(prices) < 36:
        raise ValueError("Servono almeno 36 osservazioni mensili comuni")
    return prices, source


def build_signal_frames(
    prices: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, pd.DataFrame]:
    baseline = continuous_absolute_momentum(
        prices,
        "EQUITY",
        "CASH",
        lookback_periods=config.lookback_periods,
        lookback_weights=config.lookback_weights,
        volatility_window_months=config.volatility_window_months,
        transition_width=config.transition_width,
        alpha_up=config.alpha_up,
        alpha_down=config.alpha_down,
        round_to=config.round_to,
    )
    recovery = continuous_absolute_momentum_recovery(
        prices,
        "EQUITY",
        "CASH",
        lookback_periods=config.lookback_periods,
        lookback_weights=config.lookback_weights,
        volatility_window_months=config.volatility_window_months,
        transition_width=config.transition_width,
        alpha_up=config.alpha_up,
        alpha_down=config.alpha_down,
        round_to=config.round_to,
        probe_size=config.probe_size,
        recovery_alpha_up=config.recovery_alpha_up,
        recovery_step_cap=config.recovery_step_cap,
        recovery_activation_ceiling=config.recovery_activation_ceiling,
        recovery_target_ceiling=config.recovery_target_ceiling,
    )
    return {"Continuous L1": baseline, "Recovery L1": recovery}


def build_targets(
    prices: pd.DataFrame,
    config: BacktestConfig,
    signals: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.Series]:
    if signals is None:
        signals = build_signal_frames(prices, config)
    return {
        "Buy & Hold": pd.Series(1.0, index=prices.index),
        "Binary 12m": binary_signal(prices, "EQUITY", "CASH", 12),
        "Graduated 3/6/12": momentum_score(
            prices, "EQUITY", "CASH", config.lookback_periods
        ),
        "Continuous L1": signals["Continuous L1"]["target_weight"],
        "Recovery L1": signals["Recovery L1"]["target_weight"],
    }


def run_strategies(
    prices: pd.DataFrame,
    targets: dict[str, pd.Series],
    transaction_cost_bps: float,
) -> tuple[dict[str, pd.DataFrame], pd.Index]:
    asset_returns = prices[["EQUITY", "CASH"]].pct_change(fill_method=None)
    applicable = pd.concat(targets, axis=1).shift(1).notna().all(axis=1)
    applicable &= asset_returns.notna().all(axis=1)
    if not applicable.any():
        raise ValueError("Nessuna data comune dopo il warm-up")
    common_start = applicable[applicable].index[0]
    start_position = prices.index.get_loc(common_start)
    if start_position == 0:
        raise ValueError("Serve un mese precedente alla prima data valutabile")

    # Tutte le strategie partono dallo stesso capitale cash un mese prima
    # della finestra valutata. Cosi anche il costo iniziale e confrontabile.
    evaluation_prices = prices.iloc[start_position - 1 :]
    results = {
        name: backtest_allocation(
            evaluation_prices,
            "EQUITY",
            "CASH",
            target.reindex(evaluation_prices.index),
            transaction_cost_bps=transaction_cost_bps,
        )
        for name, target in targets.items()
    }
    returns = pd.concat(
        {name: result["net_return"] for name, result in results.items()}, axis=1
    )
    common_index = returns.dropna(how="any").index
    if len(common_index) < 24:
        raise ValueError("Meno di 24 mesi confrontabili dopo il warm-up")
    return results, common_index


def _drawdown_duration(returns: pd.Series) -> int:
    equity = (1 + returns.dropna()).cumprod()
    underwater = equity < equity.cummax()
    longest = current = 0
    for value in underwater:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def _whipsaw_count(weight: pd.Series, window_months: int = 3) -> int:
    """Conta cambi di regime sopra/sotto 50% annullati entro pochi mesi."""
    regime = weight.dropna().ge(0.50).astype(int)
    switch_positions = np.flatnonzero(regime.ne(regime.shift(1)).to_numpy())[1:]
    if len(switch_positions) < 2:
        return 0
    return int(np.sum(np.diff(switch_positions) <= window_months))


def performance_metrics(
    result: pd.DataFrame,
    common_index: pd.Index,
    cash_returns: pd.Series,
) -> dict[str, float | int | str]:
    returns = result.loc[common_index, "net_return"].dropna()
    weights = result.loc[returns.index, "target_weight"]
    turnover = result.loc[returns.index, "turnover"]
    cash = cash_returns.loc[returns.index]
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    years = len(returns) / 12
    annual_vol = returns.std(ddof=1) * np.sqrt(12)
    excess = returns - cash
    excess_vol = excess.std(ddof=1) * np.sqrt(12)
    return {
        "start": returns.index[0].date().isoformat(),
        "end": returns.index[-1].date().isoformat(),
        "months": int(len(returns)),
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": float(equity.iloc[-1] ** (1 / years) - 1),
        "annualized_volatility": float(annual_vol),
        "excess_sharpe_vs_cash": float(
            excess.mean() * 12 / excess_vol if excess_vol > 0 else np.nan
        ),
        "max_drawdown": float(drawdown.min()),
        "max_drawdown_duration_months": _drawdown_duration(returns),
        "worst_month": float(returns.min()),
        "average_equity_weight": float(weights.mean()),
        "annual_turnover": float(turnover.sum() / years),
        "regime_switches": int(weights.ge(0.50).astype(int).diff().abs().sum()),
        "whipsaws_within_3m": _whipsaw_count(weights, 3),
    }


def calibration_table(
    prices: pd.DataFrame,
    signal: pd.Series,
    horizons: tuple[int, ...] = (1, 3, 6),
) -> pd.DataFrame:
    frame = pd.DataFrame({"target_weight": signal})
    for horizon in horizons:
        equity_growth = prices["EQUITY"].shift(-horizon) / prices["EQUITY"]
        cash_growth = prices["CASH"].shift(-horizon) / prices["CASH"]
        frame[f"forward_excess_{horizon}m"] = equity_growth / cash_growth - 1

    forward_drawdown = []
    horizon = max(horizons)
    equity = prices["EQUITY"]
    for position in range(len(prices)):
        future = equity.iloc[position + 1 : position + horizon + 1]
        if len(future) < horizon:
            forward_drawdown.append(np.nan)
        else:
            relative_path = future / equity.iloc[position] - 1
            forward_drawdown.append(min(0.0, float(relative_path.min())))
    frame[f"forward_drawdown_{horizon}m"] = forward_drawdown

    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    frame["exposure_bin"] = pd.cut(
        frame["target_weight"],
        bins=[-0.001, 0.20, 0.40, 0.60, 0.80, 1.001],
        labels=labels,
        include_lowest=True,
    )
    aggregations: dict[str, tuple[str, str]] = {
        "observations": (f"forward_excess_{max(horizons)}m", "count"),
        "mean_target_weight": ("target_weight", "mean"),
    }
    for horizon_value in horizons:
        column = f"forward_excess_{horizon_value}m"
        aggregations[f"mean_excess_{horizon_value}m"] = (column, "mean")
        aggregations[f"median_excess_{horizon_value}m"] = (column, "median")
    aggregations[f"mean_forward_drawdown_{horizon}m"] = (
        f"forward_drawdown_{horizon}m",
        "mean",
    )
    return (
        frame.dropna(subset=["exposure_bin"])
        .groupby("exposure_bin", observed=False)
        .agg(**aggregations)
        .reset_index()
    )


def drawdown_episodes(equity_prices: pd.Series) -> pd.DataFrame:
    """Estrae episodi picco-trough-recovery non sovrapposti."""
    prices = equity_prices.dropna()
    peak_date = prices.index[0]
    peak_value = float(prices.iloc[0])
    trough_date = peak_date
    trough_value = peak_value
    in_drawdown = False
    rows: list[dict[str, object]] = []

    for date, value_raw in prices.iloc[1:].items():
        value = float(value_raw)
        if value >= peak_value:
            if in_drawdown:
                rows.append(
                    {
                        "peak_date": peak_date,
                        "trough_date": trough_date,
                        "recovery_date": date,
                        "drawdown": trough_value / peak_value - 1,
                    }
                )
            peak_date = date
            peak_value = value
            trough_date = date
            trough_value = value
            in_drawdown = False
        else:
            in_drawdown = True
            if value < trough_value:
                trough_date = date
                trough_value = value

    if in_drawdown:
        rows.append(
            {
                "peak_date": peak_date,
                "trough_date": trough_date,
                "recovery_date": pd.NaT,
                "drawdown": trough_value / peak_value - 1,
            }
        )
    return pd.DataFrame(rows)


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return (end.year - start.year) * 12 + end.month - start.month


def stress_table(
    prices: pd.DataFrame,
    strategy_result: pd.DataFrame,
    strategy_name: str,
    top_n: int = 5,
) -> pd.DataFrame:
    episodes = drawdown_episodes(prices["EQUITY"])
    if episodes.empty:
        return episodes
    episodes = episodes.nsmallest(top_n, "drawdown").sort_values("peak_date")
    rows: list[dict[str, object]] = []
    for episode in episodes.itertuples(index=False):
        peak = pd.Timestamp(episode.peak_date)
        trough = pd.Timestamp(episode.trough_date)
        recovery = (
            pd.Timestamp(episode.recovery_date)
            if pd.notna(episode.recovery_date)
            else prices.index[-1]
        )
        event_returns = strategy_result.loc[
            (strategy_result.index > peak) & (strategy_result.index <= trough),
            "net_return",
        ].dropna()
        strategy_return = (1 + event_returns).prod() - 1 if len(event_returns) else np.nan
        event_weights = strategy_result.loc[
            (strategy_result.index >= peak) & (strategy_result.index <= trough),
            "target_weight",
        ].dropna()
        defensive = event_weights[event_weights < 0.50]
        de_risk_date = defensive.index[0] if len(defensive) else pd.NaT

        post_trough = strategy_result.loc[
            (strategy_result.index > trough) & (strategy_result.index <= recovery),
            "target_weight",
        ].dropna()
        risk_on = post_trough[post_trough >= 0.50]
        reentry_date = risk_on.index[0] if len(risk_on) else pd.NaT

        rebound_end_position = min(prices.index.get_loc(trough) + 3, len(prices) - 1)
        rebound_end = prices.index[rebound_end_position]
        rebound_returns = strategy_result.loc[
            (strategy_result.index > trough)
            & (strategy_result.index <= rebound_end),
            "net_return",
        ].dropna()
        rows.append(
            {
                "strategy": strategy_name,
                "peak_date": peak.date().isoformat(),
                "trough_date": trough.date().isoformat(),
                "recovery_date": (
                    pd.Timestamp(episode.recovery_date).date().isoformat()
                    if pd.notna(episode.recovery_date)
                    else "not_recovered"
                ),
                "benchmark_drawdown": float(episode.drawdown),
                "strategy_return_peak_to_trough": float(strategy_return),
                "weight_at_peak": float(event_weights.iloc[0]) if len(event_weights) else np.nan,
                "weight_at_trough": float(event_weights.iloc[-1]) if len(event_weights) else np.nan,
                "de_risk_lag_months": (
                    _months_between(peak, de_risk_date) if pd.notna(de_risk_date) else np.nan
                ),
                "reentry_lag_from_trough_months": (
                    _months_between(trough, reentry_date)
                    if pd.notna(reentry_date)
                    else np.nan
                ),
                "benchmark_rebound_3m": float(
                    prices.loc[rebound_end, "EQUITY"] / prices.loc[trough, "EQUITY"] - 1
                ),
                "strategy_rebound_3m": float(
                    (1 + rebound_returns).prod() - 1 if len(rebound_returns) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def robustness_grid(
    prices: pd.DataFrame,
    config: BacktestConfig,
    common_index: pd.Index,
) -> pd.DataFrame:
    cash_returns = prices["CASH"].pct_change(fill_method=None)
    profiles = {
        "equal": (1 / 3, 1 / 3, 1 / 3),
        "default": (0.20, 0.30, 0.50),
        "long_heavy": (0.10, 0.20, 0.70),
    }
    rows: list[dict[str, object]] = []
    for (profile_name, weights), transition, alpha_up, alpha_down in product(
        profiles.items(),
        (0.50, 0.75, 1.00),
        (0.20, 0.30, 0.40),
        (0.45, 0.60, 0.75),
    ):
        target = continuous_absolute_momentum(
            prices,
            "EQUITY",
            "CASH",
            lookback_periods=config.lookback_periods,
            lookback_weights=weights,
            volatility_window_months=config.volatility_window_months,
            transition_width=transition,
            alpha_up=alpha_up,
            alpha_down=alpha_down,
            round_to=config.round_to,
        )["target_weight"]
        result = backtest_allocation(
            prices,
            "EQUITY",
            "CASH",
            target,
            transaction_cost_bps=config.transaction_cost_bps,
        )
        metrics = performance_metrics(result, common_index, cash_returns)
        rows.append(
            {
                "weight_profile": profile_name,
                "transition_width": transition,
                "alpha_up": alpha_up,
                "alpha_down": alpha_down,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def recovery_robustness_grid(
    prices: pd.DataFrame,
    config: BacktestConfig,
    common_index: pd.Index,
) -> pd.DataFrame:
    """Griglia limitata e predefinita: verifica stabilita, non cerca l'ottimo."""
    cash_returns = prices["CASH"].pct_change(fill_method=None)
    rows: list[dict[str, object]] = []
    for probe_size, recovery_alpha_up, recovery_step_cap in product(
        (0.15, 0.20),
        (0.50, 0.60, 0.70),
        (0.15, 0.20),
    ):
        signal = continuous_absolute_momentum_recovery(
            prices,
            "EQUITY",
            "CASH",
            lookback_periods=config.lookback_periods,
            lookback_weights=config.lookback_weights,
            volatility_window_months=config.volatility_window_months,
            transition_width=config.transition_width,
            alpha_up=config.alpha_up,
            alpha_down=config.alpha_down,
            round_to=config.round_to,
            probe_size=probe_size,
            recovery_alpha_up=recovery_alpha_up,
            recovery_step_cap=recovery_step_cap,
            recovery_activation_ceiling=config.recovery_activation_ceiling,
            recovery_target_ceiling=config.recovery_target_ceiling,
        )
        result = backtest_allocation(
            prices,
            "EQUITY",
            "CASH",
            signal["target_weight"],
            transaction_cost_bps=config.transaction_cost_bps,
        )
        metrics = performance_metrics(result, common_index, cash_returns)
        rows.append(
            {
                "probe_size": probe_size,
                "recovery_alpha_up": recovery_alpha_up,
                "recovery_step_cap": recovery_step_cap,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def robustness_summary(grid: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "cagr",
        "max_drawdown",
        "average_equity_weight",
        "annual_turnover",
        "whipsaws_within_3m",
    ]
    return pd.DataFrame(
        {
            "p10": grid[columns].quantile(0.10),
            "median": grid[columns].median(),
            "p90": grid[columns].quantile(0.90),
            "min": grid[columns].min(),
            "max": grid[columns].max(),
        }
    ).rename_axis("metric").reset_index()


def plot_diagnostics(
    results: dict[str, pd.DataFrame],
    common_index: pd.Index,
    output_path: Path,
) -> None:
    """Multi-linea della crescita e pannello separato per l'esposizione."""
    colors = {
        "Buy & Hold": "#374151",
        "Binary 12m": "#6B7280",
        "Graduated 3/6/12": "#9CA3AF",
        "Continuous L1": "#2563EB",
        "Recovery L1": "#D97706",
    }
    styles = {
        "Buy & Hold": "--",
        "Binary 12m": ":",
        "Graduated 3/6/12": "-.",
        "Continuous L1": "-",
        "Recovery L1": "-",
    }
    fig, (wealth_ax, weight_ax) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for name, result in results.items():
        wealth = (1 + result.loc[common_index, "net_return"]).cumprod()
        wealth_ax.plot(
            wealth.index,
            wealth,
            label=name,
            color=colors[name],
            linestyle=styles[name],
            linewidth=2.2 if name == "Recovery L1" else (1.8 if name == "Continuous L1" else 1.3),
        )
    fig.suptitle(
        "Layer 1 - confronto fuori campione mensile",
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=14,
    )
    fig.text(
        0.08,
        0.947,
        "Segnale T applicato a T+1; stessa finestra; rendimenti netti dei costi",
        fontsize=9,
        color="#4B5563",
    )
    wealth_ax.set_ylabel("Crescita di 1")
    wealth_ax.set_yscale("log")
    wealth_ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    wealth_ax.legend(frameon=False, ncol=3)

    continuous_weight = results["Continuous L1"].loc[common_index, "target_weight"]
    weight_ax.step(
        continuous_weight.index,
        continuous_weight,
        where="mid",
        color="#2563EB",
        linewidth=1.8,
        label="Continuous L1",
    )
    recovery_weight = results["Recovery L1"].loc[common_index, "target_weight"]
    weight_ax.step(
        recovery_weight.index,
        recovery_weight,
        where="mid",
        color="#D97706",
        linewidth=1.8,
        label="Recovery L1",
    )
    weight_ax.axhline(0.50, color="#374151", linewidth=1, linestyle="--")
    weight_ax.set_ylim(-0.03, 1.03)
    weight_ax.set_ylabel("Esposizione")
    weight_ax.set_yticks([0, 0.5, 1], labels=["0%", "50%", "100%"])
    weight_ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    weight_ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_calibration(
    baseline: pd.DataFrame,
    recovery: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(baseline))
    width = 0.38
    baseline_values = baseline["mean_excess_6m"].to_numpy(dtype=float)
    recovery_values = recovery["mean_excess_6m"].to_numpy(dtype=float)
    baseline_bars = ax.bar(
        x - width / 2,
        baseline_values,
        width,
        label="Continuous L1",
        color="white",
        edgecolor="#2563EB",
        linewidth=1.5,
    )
    recovery_bars = ax.bar(
        x + width / 2,
        recovery_values,
        width,
        label="Recovery L1",
        color="#D97706",
        edgecolor="#1F2937",
        linewidth=0.7,
    )
    ax.axhline(0, color="#374151", linewidth=1)
    fig.suptitle(
        "Calibrazione del target: excess return azionario successivo",
        x=0.10,
        y=0.985,
        ha="left",
        fontsize=14,
    )
    fig.text(
        0.10,
        0.925,
        "Media a 6 mesi rispetto a BIL; osservazioni forward valide indicate sulle barre",
        fontsize=9,
        color="#4B5563",
    )
    ax.set_xticks(x, baseline["exposure_bin"].astype(str))
    ax.set_xlabel("Target azionario del Layer 1")
    ax.set_ylabel("Excess return medio a 6 mesi")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False)
    for bars, counts in (
        (baseline_bars, baseline["observations"]),
        (recovery_bars, recovery["observations"]),
    ):
        for bar, count in zip(bars, counts):
            y = bar.get_height()
            ax.annotate(
                f"n={int(count)}",
                (bar.get_x() + bar.get_width() / 2, y),
                xytext=(0, 4 if y >= 0 else -14),
                textcoords="offset points",
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=7,
            )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_rebound_comparison(stress: pd.DataFrame, output_path: Path) -> None:
    pivot = stress.pivot(
        index="trough_date",
        columns="strategy",
        values="strategy_rebound_3m",
    )
    benchmark = stress.groupby("trough_date")["benchmark_rebound_3m"].first()
    plot_data = pivot.join(benchmark.rename("Benchmark")).dropna(
        subset=["Continuous L1", "Recovery L1", "Benchmark"]
    )
    x = np.arange(len(plot_data))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        x - width,
        plot_data["Benchmark"],
        width,
        label="URTH",
        color="#D1D5DB",
        edgecolor="#374151",
    )
    ax.bar(
        x,
        plot_data["Continuous L1"],
        width,
        label="Continuous L1",
        color="white",
        edgecolor="#2563EB",
        linewidth=1.5,
    )
    ax.bar(
        x + width,
        plot_data["Recovery L1"],
        width,
        label="Recovery L1",
        color="#D97706",
        edgecolor="#1F2937",
    )
    fig.suptitle(
        "Partecipazione ai primi tre mesi dopo il minimo",
        x=0.10,
        y=0.985,
        ha="left",
        fontsize=14,
    )
    fig.text(
        0.10,
        0.925,
        "Rendimento netto nei maggiori drawdown URTH; segnale mensile applicato a T+1",
        fontsize=9,
        color="#4B5563",
    )
    ax.set_xticks(x, plot_data.index.astype(str))
    ax.set_xlabel("Mese del minimo")
    ax.set_ylabel("Rendimento nei 3 mesi successivi")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.axhline(0, color="#374151", linewidth=1)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def recovery_gate_table(metrics: pd.DataFrame, stress: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics.loc["Continuous L1"]
    recovery = metrics.loc["Recovery L1"]
    pivot_rebound = stress.pivot(
        index="trough_date",
        columns="strategy",
        values="strategy_rebound_3m",
    ).dropna(subset=["Continuous L1", "Recovery L1"])
    pivot_lag = stress.pivot(
        index="trough_date",
        columns="strategy",
        values="reentry_lag_from_trough_months",
    ).dropna(subset=["Continuous L1", "Recovery L1"])

    drawdown_deterioration = float(
        baseline["max_drawdown"] - recovery["max_drawdown"]
    )
    whipsaw_increase = int(
        recovery["whipsaws_within_3m"] - baseline["whipsaws_within_3m"]
    )
    rebound_gain = float(
        (pivot_rebound["Recovery L1"] - pivot_rebound["Continuous L1"]).mean()
    )
    lag_change = float(
        (pivot_lag["Recovery L1"] - pivot_lag["Continuous L1"]).median()
    )
    return pd.DataFrame(
        [
            {
                "gate": "max_drawdown_deterioration",
                "value": drawdown_deterioration,
                "rule": "<= 0.02",
                "passed": drawdown_deterioration <= 0.02,
            },
            {
                "gate": "whipsaw_increase",
                "value": whipsaw_increase,
                "rule": "<= 2",
                "passed": whipsaw_increase <= 2,
            },
            {
                "gate": "mean_3m_rebound_gain",
                "value": rebound_gain,
                "rule": "> 0",
                "passed": rebound_gain > 0,
            },
            {
                "gate": "median_reentry_lag_change_months",
                "value": lag_change,
                "rule": "<= 0",
                "passed": lag_change <= 0,
            },
        ]
    )


def write_summary(
    path: Path,
    source: str,
    prices: pd.DataFrame,
    metrics: pd.DataFrame,
    baseline_calibration: pd.DataFrame,
    recovery_calibration: pd.DataFrame,
    stress: pd.DataFrame,
    robustness: pd.DataFrame,
    recovery_robustness: pd.DataFrame,
    gates: pd.DataFrame,
    config: BacktestConfig,
) -> None:
    nonempty_bins = recovery_calibration.loc[recovery_calibration["observations"] > 0]
    monotonic_pairs = nonempty_bins["mean_excess_6m"].diff().dropna()
    violations = int((monotonic_pairs < 0).sum())
    continuous = metrics.loc["Continuous L1"]
    recovery = metrics.loc["Recovery L1"]
    binary = metrics.loc["Binary 12m"]
    gate_passed = bool(gates["passed"].all())
    stress_pivot = stress.pivot(
        index="trough_date",
        columns="strategy",
        values="strategy_rebound_3m",
    ).dropna(subset=["Continuous L1", "Recovery L1"])
    rebound_gain = (
        stress_pivot["Recovery L1"] - stress_pivot["Continuous L1"]
    ).mean()
    lines = [
        "# Layer 1 - sintesi del backtest critico",
        "",
        f"- Fonte: {source}",
        f"- Dati disponibili: {prices.index[0].date()} / {prices.index[-1].date()}",
        f"- Finestra comune valutata: {continuous['start']} / {continuous['end']}",
        f"- Costi: {config.transaction_cost_bps:.1f} bps per unita di turnover",
        "- Regola causale: il target calcolato a fine mese T e applicato a T+1.",
        "",
        "## Test principali",
        "",
        f"- Max drawdown continuo: {continuous['max_drawdown']:.1%}; binario: {binary['max_drawdown']:.1%}.",
        f"- Max drawdown Recovery: {recovery['max_drawdown']:.1%}.",
        f"- Esposizione media continua: {continuous['average_equity_weight']:.1%}.",
        f"- Esposizione media Recovery: {recovery['average_equity_weight']:.1%}.",
        f"- Whipsaw entro 3 mesi: continuo {int(continuous['whipsaws_within_3m'])}; binario {int(binary['whipsaws_within_3m'])}.",
        f"- Whipsaw Recovery: {int(recovery['whipsaws_within_3m'])}.",
        f"- Guadagno medio Recovery nei primi 3 mesi dai minimi: {rebound_gain:+.1%} rispetto al continuo.",
        f"- Violazioni della monotonicita tra bucket adiacenti: {violations} su {max(len(monotonic_pairs), 0)}.",
        f"- Robustness grid: {len(robustness)} configurazioni vicine, senza selezione dell'ottimo.",
        f"- Recovery grid: {len(recovery_robustness)} configurazioni predefinite.",
        f"- Gate sperimentali: {'SUPERATI' if gate_passed else 'NON SUPERATI'}.",
        "",
        "## Criterio di lettura",
        "",
        "Il modello non e promosso dal CAGR piu alto. E credibile solo se riduce la severita dei drawdown, non crea troppi falsi cambi di regime, assegna esposizioni maggiori a condizioni ex-post migliori e conserva risultati simili nella griglia di parametri vicini.",
        "La calibrazione usa finestre forward sovrapposte: e diagnostica descrittiva, non un test di significativita statistica.",
        "",
        "## Limite strutturale",
        "",
        "URTH/BIL offrono una finestra comune relativamente breve e non includono la bolla dot-com o la crisi 2008. Prima di congelare il Layer 1 occorre ripetere lo stesso test con una serie total-return storica del MSCI World e Treasury Bill USD, caricandola tramite --prices-csv.",
        "",
        "![Diagnostica](backtest_diagnostics.png)",
        "",
        "![Calibrazione](calibration.png)",
        "",
        "![Ripartenze](recovery_stress.png)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_outputs(
    output_dir: Path,
    source: str,
    prices: pd.DataFrame,
    results: dict[str, pd.DataFrame],
    common_index: pd.Index,
    metrics: pd.DataFrame,
    baseline_calibration: pd.DataFrame,
    recovery_calibration: pd.DataFrame,
    stress: pd.DataFrame,
    robustness: pd.DataFrame,
    recovery_robustness: pd.DataFrame,
    gates: pd.DataFrame,
    recovery_signal: pd.DataFrame,
    config: BacktestConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "strategy_metrics.csv")
    baseline_calibration.to_csv(output_dir / "calibration.csv", index=False)
    recovery_calibration.to_csv(
        output_dir / "calibration_recovery.csv", index=False
    )
    stress.to_csv(output_dir / "stress_events.csv", index=False)
    robustness.to_csv(output_dir / "robustness_grid.csv", index=False)
    robustness_summary(robustness).to_csv(
        output_dir / "robustness_summary.csv", index=False
    )
    recovery_robustness.to_csv(
        output_dir / "recovery_robustness_grid.csv", index=False
    )
    robustness_summary(recovery_robustness).to_csv(
        output_dir / "recovery_robustness_summary.csv", index=False
    )
    gates.to_csv(output_dir / "recovery_gates.csv", index=False)
    recovery_signal.to_csv(output_dir / "recovery_signal.csv")
    plot_diagnostics(results, common_index, output_dir / "backtest_diagnostics.png")
    plot_calibration(
        baseline_calibration,
        recovery_calibration,
        output_dir / "calibration.png",
    )
    plot_rebound_comparison(stress, output_dir / "recovery_stress.png")
    write_summary(
        output_dir / "critical_backtest_summary.md",
        source,
        prices,
        metrics,
        baseline_calibration,
        recovery_calibration,
        stress,
        robustness,
        recovery_robustness,
        gates,
        config,
    )
    manifest = {
        "source": source,
        "data_start": prices.index[0].date().isoformat(),
        "data_end": prices.index[-1].date().isoformat(),
        "price_currency": "USD",
        "frequency": "monthly",
        "signal_lag_months": 1,
        "models": ["L1-continuous-v1", "L1-recovery-v1-experimental"],
        "config": config.__dict__,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def execute_backtest(
    prices: pd.DataFrame,
    source: str,
    output_dir: Path,
    config: BacktestConfig,
) -> pd.DataFrame:
    signals = build_signal_frames(prices, config)
    targets = build_targets(prices, config, signals=signals)
    results, common_index = run_strategies(
        prices, targets, config.transaction_cost_bps
    )
    cash_returns = prices["CASH"].pct_change(fill_method=None)
    metrics = pd.DataFrame(
        {
            name: performance_metrics(result, common_index, cash_returns)
            for name, result in results.items()
        }
    ).T
    baseline_calibration = calibration_table(prices, targets["Continuous L1"])
    recovery_calibration = calibration_table(prices, targets["Recovery L1"])
    stress = pd.concat(
        [
            stress_table(prices, results["Continuous L1"], "Continuous L1"),
            stress_table(prices, results["Recovery L1"], "Recovery L1"),
        ],
        ignore_index=True,
    )
    robustness = robustness_grid(prices, config, common_index)
    recovery_robustness = recovery_robustness_grid(prices, config, common_index)
    gates = recovery_gate_table(metrics, stress)
    recovery_signal = signals["Recovery L1"][
        [
            "raw_score",
            "baseline_target_weight",
            "fast_excess_1m",
            "fast_excess_3m",
            "fast_excess_3m_change",
            "recovery_mode",
            "probe_active",
            "recovery_confirmed",
            "effective_alpha",
            "target_weight",
        ]
    ]
    save_outputs(
        output_dir,
        source,
        prices,
        results,
        common_index,
        metrics,
        baseline_calibration,
        recovery_calibration,
        stress,
        robustness,
        recovery_robustness,
        gates,
        recovery_signal,
        config,
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equity", default="URTH")
    parser.add_argument("--cash", default="BIL")
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--prices-csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/latest"))
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BacktestConfig(
        equity=args.equity,
        cash=args.cash,
        start=args.start,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    prices, source = load_prices(
        args.equity, args.cash, args.start, prices_csv=args.prices_csv
    )
    metrics = execute_backtest(prices, source, args.output, config)
    columns = [
        "cagr",
        "max_drawdown",
        "average_equity_weight",
        "annual_turnover",
        "whipsaws_within_3m",
    ]
    print(metrics[columns].to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nOutput: {args.output.resolve()}")


if __name__ == "__main__":
    main()
