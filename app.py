"""Streamlit app per il Layer 1 di avversione globale al rischio."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yfinance as yf

from dual_momentum import performance_stats
from risk_indicator import (
    backtest_allocation,
    binary_signal,
    continuous_absolute_momentum,
)


st.set_page_config(page_title="Layer 1 - Global Risk", layout="wide")
st.title("Layer 1 - Esposizione azionaria globale")
st.caption(
    "Absolute momentum continuo in USD. URTH misura l'azionario sviluppato globale; "
    "BIL rappresenta il rendimento dei Treasury Bill USA a 1-3 mesi. Il risultato "
    "indica quanta esposizione azionaria mantenere, non quali titoli acquistare."
)

st.sidebar.header("Mercato")
equity_ticker = st.sidebar.text_input("Benchmark azionario", value="URTH")
cash_ticker = st.sidebar.text_input("Hurdle cash USD", value="BIL")
start_date = st.sidebar.date_input("Inizio storico", value=pd.Timestamp("2012-01-01"))

st.sidebar.header("Absolute momentum")
short_lb = st.sidebar.number_input("Orizzonte breve (mesi)", 1, 24, 3)
medium_lb = st.sidebar.number_input("Orizzonte medio (mesi)", 2, 36, 6)
long_lb = st.sidebar.number_input("Orizzonte lungo (mesi)", 3, 60, 12)

short_weight = st.sidebar.number_input("Peso breve", 0.0, 1.0, 0.20, 0.05)
medium_weight = st.sidebar.number_input("Peso medio", 0.0, 1.0, 0.30, 0.05)
long_weight = st.sidebar.number_input("Peso lungo", 0.0, 1.0, 0.50, 0.05)

with st.sidebar.expander("Smussamento e costi"):
    volatility_window = st.number_input("Finestra volatilita (mesi)", 6, 36, 12)
    transition_width = st.slider(
        "Ampiezza della transizione",
        min_value=0.25,
        max_value=2.00,
        value=0.75,
        step=0.05,
        help="Valori maggiori rendono piu graduale il passaggio tra risk-off e risk-on.",
    )
    alpha_up = st.slider("Velocita aumento esposizione", 0.05, 1.00, 0.30, 0.05)
    alpha_down = st.slider("Velocita riduzione esposizione", 0.05, 1.00, 0.60, 0.05)
    round_step_pct = st.select_slider(
        "Passo operativo",
        options=[0, 1, 2, 5, 10],
        value=5,
        format_func=lambda value: "Nessuno" if value == 0 else f"{value}%",
    )
    transaction_cost_bps = st.slider("Costo sul turnover (bps)", 0, 50, 10)

run_button = st.sidebar.button("Calcola", type="primary")


@st.cache_data(ttl=3600, show_spinner="Scarico i dati di mercato...")
def load_prices(equity: str, cash: str, start) -> pd.DataFrame:
    close = yf.download(
        [equity, cash],
        start=start,
        auto_adjust=True,
        progress=False,
    )["Close"]
    monthly = close.rename(columns={equity: "EQUITY", cash: "CASH"})
    return monthly.resample("ME").last().dropna(how="any")


if run_button:
    periods = (int(short_lb), int(medium_lb), int(long_lb))
    weights = (float(short_weight), float(medium_weight), float(long_weight))
    if len(set(periods)) != 3:
        st.error("I tre orizzonti devono essere diversi.")
        st.stop()
    if sum(weights) <= 0:
        st.error("La somma dei pesi deve essere positiva.")
        st.stop()

    try:
        prices = load_prices(equity_ticker, cash_ticker, start_date)
        signal = continuous_absolute_momentum(
            prices,
            "EQUITY",
            "CASH",
            lookback_periods=periods,
            lookback_weights=weights,
            volatility_window_months=int(volatility_window),
            transition_width=transition_width,
            alpha_up=alpha_up,
            alpha_down=alpha_down,
            round_to=round_step_pct / 100,
        )
    except Exception as exc:
        st.error(f"Calcolo non riuscito: {exc}")
        st.stop()

    valid_signal = signal.dropna(subset=["target_weight"])
    if valid_signal.empty:
        st.error("Storico insufficiente per i parametri selezionati.")
        st.stop()

    latest_date = valid_signal.index[-1]
    latest = valid_signal.iloc[-1]
    target = latest["target_weight"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Azionario target", f"{target:.0%}")
    col2.metric("Liquidita target", f"{1 - target:.0%}")
    col3.metric("Segnale prima dello smussamento", f"{latest['raw_score']:.1%}")
    col4.metric("Ultimo dato", latest_date.strftime("%d %b %Y"))

    st.info(
        "Il target e un limite operativo: se i layer successivi non trovano "
        "abbastanza acquisti validi, l'esposizione effettiva puo essere inferiore."
    )

    details = []
    normalized_weights = [weight / sum(weights) for weight in weights]
    for period, weight in zip(periods, normalized_weights):
        details.append(
            {
                "Orizzonte": f"{period} mesi",
                "Peso": weight,
                "Excess momentum vs BIL": latest[f"excess_{period}m"],
                "Forza / rumore": latest[f"strength_{period}m"],
                "Punteggio risk-on": latest[f"score_{period}m"],
            }
        )
    details_df = pd.DataFrame(details)
    st.subheader("Composizione del segnale")
    st.dataframe(
        details_df.style.format(
            {
                "Peso": "{:.0%}",
                "Excess momentum vs BIL": "{:.2%}",
                "Forza / rumore": "{:.2f}",
                "Punteggio risk-on": "{:.1%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    continuous_bt = backtest_allocation(
        prices,
        "EQUITY",
        "CASH",
        signal["target_weight"],
        transaction_cost_bps=transaction_cost_bps,
    )
    binary = binary_signal(prices, "EQUITY", "CASH", lookback_months=12)
    binary_bt = backtest_allocation(
        prices,
        "EQUITY",
        "CASH",
        binary,
        transaction_cost_bps=transaction_cost_bps,
    )
    buy_hold_returns = prices["EQUITY"].pct_change(fill_method=None)
    buy_hold_equity = (1 + buy_hold_returns.fillna(0)).cumprod()

    st.subheader("Storico")
    figure, (equity_ax, signal_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    equity_ax.plot(buy_hold_equity, label="URTH Buy & Hold", color="gray", alpha=0.65)
    equity_ax.plot(continuous_bt["equity"], label="Layer 1 continuo", color="steelblue")
    equity_ax.plot(binary_bt["equity"], label="Absolute momentum binario 12m", color="darkred", alpha=0.65)
    equity_ax.set_yscale("log")
    equity_ax.set_ylabel("Crescita di 1")
    equity_ax.legend()

    signal_ax.plot(signal["raw_score"], label="Segnale grezzo", color="lightgray")
    signal_ax.step(
        signal["target_weight"].index,
        signal["target_weight"],
        where="mid",
        label="Esposizione target",
        color="steelblue",
    )
    signal_ax.set_ylim(0, 1)
    signal_ax.set_ylabel("Azionario")
    signal_ax.legend()
    figure.tight_layout()
    st.pyplot(figure)

    stats = pd.DataFrame(
        {
            "Layer 1 continuo": performance_stats(continuous_bt["net_return"]),
            "Binario 12 mesi": performance_stats(binary_bt["net_return"]),
            "Buy & Hold": performance_stats(buy_hold_returns),
        }
    ).T
    st.subheader("Statistiche di confronto")
    st.dataframe(
        stats.style.format(
            {
                "CAGR": "{:.2%}",
                "Volatilita annua": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max Drawdown": "{:.2%}",
                "% Mesi profittevoli": "{:.1%}",
            }
        ),
        width="stretch",
    )

    with st.expander("Ultimi 24 segnali"):
        history = signal[["raw_score", "smoothed_score", "target_weight", "cash_weight"]].tail(24)
        st.dataframe(history.style.format("{:.1%}"), width="stretch")

    st.caption(
        "Segnale e backtest calcolati in USD su prezzi adjusted. Il segnale di fine mese T "
        "viene applicato ai rendimenti del mese T+1. Le performance reali del portafoglio "
        "saranno misurate in EUR nei layer finali."
    )
else:
    st.info("Configura il modello nella barra laterale e premi Calcola.")
