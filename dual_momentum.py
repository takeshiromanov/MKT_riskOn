"""Funzioni condivise e motore Dual Momentum destinato al futuro Layer 2."""

import numpy as np
import pandas as pd


def compute_momentum(
    prices: pd.DataFrame,
    lookback_months: int = 12,
    skip_last_month: bool = False,
) -> pd.DataFrame:
    """Rendimento totale sul periodo di lookback per ciascun asset."""
    if skip_last_month:
        return prices.shift(1) / prices.shift(lookback_months + 1) - 1
    return prices / prices.shift(lookback_months) - 1


def dual_momentum_signal(
    prices: pd.DataFrame,
    risky_assets: list[str],
    defensive_asset: str,
    cash_asset: str,
    lookback_months: int = 12,
    skip_last_month: bool = False,
) -> pd.Series:
    """Seleziona il risky asset piu forte se batte il cash, altrimenti il difensivo."""
    momentum = compute_momentum(prices, lookback_months, skip_last_month)
    required = risky_assets + [cash_asset]
    signal = pd.Series(index=prices.index, dtype=object)
    for date in prices.index:
        row = momentum.loc[date, required]
        if row.isna().any():
            signal.loc[date] = np.nan
            continue
        winner = momentum.loc[date, risky_assets].idxmax()
        signal.loc[date] = (
            winner if momentum.loc[date, winner] > momentum.loc[date, cash_asset]
            else defensive_asset
        )
    return signal


def backtest_dual_momentum(
    prices: pd.DataFrame,
    risky_assets: list[str],
    defensive_asset: str,
    cash_asset: str,
    lookback_months: int = 12,
    skip_last_month: bool = False,
    transaction_cost_bps: float = 20.0,
) -> pd.DataFrame:
    """Backtest mensile del segnale Dual Momentum."""
    returns = prices.pct_change(fill_method=None)
    raw_signal = dual_momentum_signal(
        prices,
        risky_assets,
        defensive_asset,
        cash_asset,
        lookback_months,
        skip_last_month,
    )
    position = raw_signal.shift(1)
    gross_return = pd.Series(index=prices.index, dtype=float)
    for date in prices.index:
        asset = position.loc[date]
        gross_return.loc[date] = returns.loc[date, asset] if pd.notna(asset) else np.nan
    changed = position.ne(position.shift(1)) & position.notna() & position.shift(1).notna()
    cost = changed.astype(float) * transaction_cost_bps / 10_000
    net_return = gross_return - cost
    return pd.DataFrame(
        {
            "position": position,
            "gross_return": gross_return,
            "net_return": net_return,
            "equity": (1 + net_return.fillna(0)).cumprod(),
        }
    )


def performance_stats(returns: pd.Series, periods_per_year: int = 12) -> dict:
    """Statistiche annualizzate essenziali."""
    clean = returns.dropna()
    if clean.empty:
        return {
            "CAGR": np.nan,
            "Volatilita annua": np.nan,
            "Sharpe": np.nan,
            "Max Drawdown": np.nan,
            "% Mesi profittevoli": np.nan,
        }
    equity = (1 + clean).cumprod()
    years = len(clean) / periods_per_year
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    annual_volatility = clean.std() * np.sqrt(periods_per_year)
    sharpe = (
        clean.mean() * periods_per_year / annual_volatility
        if annual_volatility > 0 else np.nan
    )
    drawdown = equity / equity.cummax() - 1
    return {
        "CAGR": cagr,
        "Volatilita annua": annual_volatility,
        "Sharpe": sharpe,
        "Max Drawdown": drawdown.min(),
        "% Mesi profittevoli": (clean > 0).mean(),
    }
