"""Layer 1: indicatore continuo di avversione al rischio azionario.

Il motore confronta un benchmark azionario globale con un proxy cash in USD
su piu orizzonti. Il margine di absolute momentum viene normalizzato per il
rumore recente del mercato, trasformato in un punteggio morbido 0-100% e poi
smussato nel tempo in modo asimmetrico: le riduzioni sono piu rapide degli
aumenti di esposizione.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dual_momentum import compute_momentum


def binary_signal(
    prices: pd.DataFrame,
    equity_asset: str,
    cash_asset: str,
    lookback_months: int = 12,
) -> pd.Series:
    """Segnale absolute momentum binario, mantenuto come benchmark."""
    mom = compute_momentum(prices, lookback_months)
    signal = (mom[equity_asset] > mom[cash_asset]).astype(float)
    signal[mom[equity_asset].isna() | mom[cash_asset].isna()] = np.nan
    return signal


def momentum_score(
    prices: pd.DataFrame,
    equity_asset: str,
    cash_asset: str,
    lookback_periods: tuple[int, ...] = (3, 6, 12),
) -> pd.Series:
    """Quota di orizzonti in cui l'azionario batte il cash."""
    votes = pd.DataFrame(index=prices.index)
    for lookback in lookback_periods:
        mom = compute_momentum(prices, lookback)
        valid = mom[equity_asset].notna() & mom[cash_asset].notna()
        votes[lookback] = np.where(valid, mom[equity_asset] > mom[cash_asset], np.nan)
    return votes.mean(axis=1, skipna=False)


def _validate_configuration(
    lookback_periods: tuple[int, ...],
    lookback_weights: tuple[float, ...],
    transition_width: float,
    alpha_up: float,
    alpha_down: float,
    round_to: float,
) -> np.ndarray:
    if not lookback_periods or len(lookback_periods) != len(lookback_weights):
        raise ValueError("lookback_periods e lookback_weights devono avere la stessa lunghezza")
    if any(period <= 0 for period in lookback_periods):
        raise ValueError("i lookback devono essere interi positivi")
    weights = np.asarray(lookback_weights, dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("i pesi devono essere non negativi e avere somma positiva")
    if transition_width <= 0:
        raise ValueError("transition_width deve essere positivo")
    if not 0 < alpha_up <= 1 or not 0 < alpha_down <= 1:
        raise ValueError("alpha_up e alpha_down devono essere compresi tra 0 e 1")
    if round_to < 0 or round_to > 1:
        raise ValueError("round_to deve essere compreso tra 0 e 1")
    return weights / weights.sum()


def continuous_absolute_momentum(
    prices: pd.DataFrame,
    equity_asset: str,
    cash_asset: str,
    lookback_periods: tuple[int, ...] = (3, 6, 12),
    lookback_weights: tuple[float, ...] = (0.20, 0.30, 0.50),
    volatility_window_months: int = 12,
    transition_width: float = 0.75,
    alpha_up: float = 0.30,
    alpha_down: float = 0.60,
    round_to: float = 0.05,
) -> pd.DataFrame:
    """Calcola esposizione azionaria continua e diagnostica del segnale.

    Per ogni orizzonte viene calcolato il rendimento relativo composto del
    benchmark azionario rispetto al cash. Il margine viene diviso per il
    movimento atteso sull'orizzonte, stimato dalla volatilita annualizzata
    recente del benchmark. La funzione tanh trasforma il rapporto in un
    punteggio morbido compreso tra 0 e 1:

        score = 0.5 * (1 + tanh(strength / transition_width))

    Il punteggio grezzo e la media pesata degli orizzonti. Lo smussamento e
    asimmetrico: alpha_down governa le riduzioni e alpha_up gli incrementi.
    ``target_weight`` e infine arrotondato al passo ``round_to``.
    """
    missing = {equity_asset, cash_asset}.difference(prices.columns)
    if missing:
        raise KeyError(f"colonne mancanti: {sorted(missing)}")
    if volatility_window_months < 2:
        raise ValueError("volatility_window_months deve essere almeno 2")

    weights = _validate_configuration(
        lookback_periods,
        lookback_weights,
        transition_width,
        alpha_up,
        alpha_down,
        round_to,
    )

    result = pd.DataFrame(index=prices.index)
    monthly_returns = prices[equity_asset].pct_change(fill_method=None)
    annualized_vol = monthly_returns.rolling(
        volatility_window_months,
        min_periods=volatility_window_months,
    ).std() * np.sqrt(12)
    result["annualized_volatility"] = annualized_vol

    scores = pd.DataFrame(index=prices.index)
    for lookback in lookback_periods:
        equity_growth = prices[equity_asset] / prices[equity_asset].shift(lookback)
        cash_growth = prices[cash_asset] / prices[cash_asset].shift(lookback)
        relative_return = equity_growth / cash_growth - 1

        expected_move = annualized_vol * np.sqrt(lookback / 12)
        strength = relative_return / expected_move.replace(0, np.nan)
        soft_score = 0.5 * (1 + np.tanh(strength / transition_width))

        result[f"excess_{lookback}m"] = relative_return
        result[f"strength_{lookback}m"] = strength
        result[f"score_{lookback}m"] = soft_score
        scores[lookback] = soft_score

    raw_score = scores.mul(weights, axis=1).sum(axis=1, min_count=len(lookback_periods))
    result["raw_score"] = raw_score.clip(0, 1)

    smoothed = pd.Series(np.nan, index=prices.index, dtype=float)
    previous = np.nan
    for date, raw_value in result["raw_score"].items():
        if pd.isna(raw_value):
            continue
        if pd.isna(previous):
            current = float(raw_value)
        else:
            alpha = alpha_up if raw_value > previous else alpha_down
            current = previous + alpha * (float(raw_value) - previous)
        smoothed.loc[date] = current
        previous = current

    result["smoothed_score"] = smoothed.clip(0, 1)
    if round_to > 0:
        rounded = np.floor(result["smoothed_score"] / round_to + 0.5) * round_to
        result["target_weight"] = rounded.clip(0, 1)
    else:
        result["target_weight"] = result["smoothed_score"]
    result["cash_weight"] = 1 - result["target_weight"]
    return result


def backtest_allocation(
    prices: pd.DataFrame,
    equity_asset: str,
    cash_asset: str,
    target_weight: pd.Series,
    transaction_cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Applica il target del mese T ai rendimenti del mese T+1.

    Il turnover e calcolato rispetto al peso azionario effettivo dopo il
    rendimento del mese precedente. In questo modo il costo include anche il
    ribilanciamento necessario per riportare il portafoglio al target, non solo
    le variazioni esplicite del target stesso. Il portafoglio parte in cash.
    """
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps non puo essere negativo")

    returns = prices[[equity_asset, cash_asset]].pct_change(fill_method=None)
    weight = target_weight.shift(1)
    turnover = pd.Series(np.nan, index=prices.index, dtype=float)
    gross_return = pd.Series(np.nan, index=prices.index, dtype=float)
    net_return = pd.Series(np.nan, index=prices.index, dtype=float)

    pre_trade_equity_weight = 0.0
    for date in prices.index:
        desired_weight = weight.loc[date]
        equity_return = returns.loc[date, equity_asset]
        cash_return = returns.loc[date, cash_asset]
        if pd.isna(desired_weight) or pd.isna(equity_return) or pd.isna(cash_return):
            continue

        desired_weight = float(desired_weight)
        period_turnover = abs(desired_weight - pre_trade_equity_weight)
        period_gross_return = (
            desired_weight * float(equity_return)
            + (1 - desired_weight) * float(cash_return)
        )
        period_cost = period_turnover * (transaction_cost_bps / 10_000)

        turnover.loc[date] = period_turnover
        gross_return.loc[date] = period_gross_return
        net_return.loc[date] = period_gross_return - period_cost

        gross_growth = 1 + period_gross_return
        if gross_growth <= 0:
            pre_trade_equity_weight = desired_weight
        else:
            pre_trade_equity_weight = (
                desired_weight * (1 + float(equity_return)) / gross_growth
            )

    equity_curve = (1 + net_return.fillna(0)).cumprod()
    return pd.DataFrame(
        {
            "target_weight": weight,
            "turnover": turnover,
            "gross_return": gross_return,
            "net_return": net_return,
            "equity": equity_curve,
        }
    )
