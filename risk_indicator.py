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


MODEL_VERSION = "L1-continuous-v1"
RECOVERY_MODEL_VERSION = "L1-recovery-v1-experimental"


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


def continuous_absolute_momentum_recovery(
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
    probe_size: float = 0.20,
    recovery_alpha_up: float = 0.60,
    recovery_step_cap: float = 0.20,
    recovery_activation_ceiling: float = 0.50,
    recovery_target_ceiling: float = 0.70,
) -> pd.DataFrame:
    """Aggiunge al Layer 1 un overlay sperimentale di rientro dai minimi.

    Il motore lento resta invariato. L'overlay si arma soltanto sotto il 50%
    quando l'excess momentum a un mese e positivo, quello a tre mesi migliora
    e il punteggio grezzo sale. Inizialmente apre una posizione pilota; quando
    anche l'excess momentum a tre mesi diventa positivo, converge piu in fretta
    verso il 70%. Un nuovo deterioramento rimuove l'overlay con ``alpha_down``.

    Tutte le condizioni usano esclusivamente dati disponibili alla data del
    segnale. Come nel modello base, il target di T va applicato soltanto a T+1.
    """
    recovery_parameters = {
        "probe_size": probe_size,
        "recovery_alpha_up": recovery_alpha_up,
        "recovery_step_cap": recovery_step_cap,
        "recovery_activation_ceiling": recovery_activation_ceiling,
        "recovery_target_ceiling": recovery_target_ceiling,
    }
    if any(not np.isfinite(value) for value in recovery_parameters.values()):
        raise ValueError("i parametri recovery devono essere finiti")
    if not 0 <= probe_size <= 1 or not 0 < recovery_step_cap <= 1:
        raise ValueError("probe_size e recovery_step_cap devono essere tra 0 e 1")
    if not 0 < recovery_alpha_up <= 1:
        raise ValueError("recovery_alpha_up deve essere compreso tra 0 e 1")
    if not 0 < recovery_activation_ceiling < recovery_target_ceiling <= 1:
        raise ValueError(
            "le soglie recovery devono essere crescenti e comprese tra 0 e 1"
        )

    result = continuous_absolute_momentum(
        prices,
        equity_asset,
        cash_asset,
        lookback_periods=lookback_periods,
        lookback_weights=lookback_weights,
        volatility_window_months=volatility_window_months,
        transition_width=transition_width,
        alpha_up=alpha_up,
        alpha_down=alpha_down,
        round_to=0,
    )
    result["baseline_target_weight"] = result["target_weight"]

    equity_growth_1m = prices[equity_asset] / prices[equity_asset].shift(1)
    cash_growth_1m = prices[cash_asset] / prices[cash_asset].shift(1)
    result["fast_excess_1m"] = equity_growth_1m / cash_growth_1m - 1

    equity_growth_3m = prices[equity_asset] / prices[equity_asset].shift(3)
    cash_growth_3m = prices[cash_asset] / prices[cash_asset].shift(3)
    result["fast_excess_3m"] = equity_growth_3m / cash_growth_3m - 1
    result["fast_excess_3m_change"] = result["fast_excess_3m"].diff()
    result["raw_score_change"] = result["raw_score"].diff()

    recovery_target = pd.Series(np.nan, index=prices.index, dtype=float)
    recovery_mode = pd.Series(False, index=prices.index, dtype=bool)
    probe_active = pd.Series(False, index=prices.index, dtype=bool)
    recovery_confirmed = pd.Series(False, index=prices.index, dtype=bool)
    effective_alpha = pd.Series(np.nan, index=prices.index, dtype=float)

    previous_target = np.nan
    active = False
    for date, row in result.iterrows():
        base_target = row["baseline_target_weight"]
        if pd.isna(base_target):
            continue
        if pd.isna(previous_target):
            current = float(base_target)
            recovery_target.loc[date] = current
            previous_target = current
            continue

        fast_positive = pd.notna(row["fast_excess_1m"]) and row["fast_excess_1m"] > 0
        three_month_improving = (
            pd.notna(row["fast_excess_3m_change"])
            and row["fast_excess_3m_change"] > 0
        )
        raw_rising = pd.notna(row["raw_score_change"]) and row["raw_score_change"] > 0
        was_active = active
        pilot_trigger = (
            not active
            and previous_target < recovery_activation_ceiling
            and fast_positive
            and three_month_improving
            and raw_rising
        )

        if active and (not fast_positive or not raw_rising):
            active = False
        if pilot_trigger:
            active = True

        confirmed = (
            active
            and pd.notna(row["fast_excess_3m"])
            and row["fast_excess_3m"] > 0
            and raw_rising
        )

        if pilot_trigger and not confirmed:
            desired = max(
                float(base_target),
                min(previous_target + probe_size, recovery_activation_ceiling),
            )
            current = min(desired, previous_target + recovery_step_cap)
            alpha_used = 1.0
        elif confirmed:
            desired = max(float(base_target), recovery_target_ceiling)
            candidate = previous_target + recovery_alpha_up * (
                desired - previous_target
            )
            current = min(candidate, previous_target + recovery_step_cap)
            current = max(current, float(base_target))
            alpha_used = recovery_alpha_up
        elif active:
            current = max(float(base_target), previous_target)
            alpha_used = 0.0
        elif was_active:
            current = previous_target + alpha_down * (
                float(base_target) - previous_target
            )
            current = max(current, float(base_target))
            alpha_used = alpha_down
        else:
            current = float(base_target)
            alpha_used = np.nan

        current = float(np.clip(current, 0, 1))
        if current >= recovery_target_ceiling or base_target >= recovery_target_ceiling:
            active = False

        recovery_target.loc[date] = current
        recovery_mode.loc[date] = active or pilot_trigger or confirmed
        probe_active.loc[date] = pilot_trigger
        recovery_confirmed.loc[date] = confirmed
        effective_alpha.loc[date] = alpha_used
        previous_target = current

    result["recovery_score"] = recovery_target
    result["recovery_mode"] = recovery_mode
    result["probe_active"] = probe_active
    result["recovery_confirmed"] = recovery_confirmed
    result["effective_alpha"] = effective_alpha
    if round_to > 0:
        rounded = np.floor(recovery_target / round_to + 0.5) * round_to
        result["target_weight"] = rounded.clip(0, 1)
    else:
        result["target_weight"] = recovery_target
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
