# Layer 1 - Global Risk Indicator

Applicazione Streamlit che stima quanta esposizione azionaria globale
mantenere e quanta liquidita conservare.

## Mandato

Il Layer 1 misura l'avversione al rischio del mercato globale. Non seleziona
titoli e non costruisce il portafoglio. Il target e anche un tetto: se i layer
successivi trovano pochi acquisti validi, l'esposizione effettiva puo restare
inferiore.

- Benchmark azionario predefinito: `URTH` (MSCI World, USD)
- Hurdle predefinito: `BIL` (Treasury Bill USA 1-3 mesi, USD)
- Orizzonti predefiniti: 3, 6 e 12 mesi
- Pesi predefiniti: 20%, 30% e 50%
- Frequenza: mensile

Il cambio EUR/USD non entra nel Layer 1. Le performance effettive del
portafoglio saranno misurate in EUR nei layer finali.

## Segnale continuo

Per ciascun orizzonte viene calcolato il rendimento composto di URTH rispetto
a BIL. Il margine viene rapportato alla volatilita recente di URTH e convertito
con una funzione `tanh` in un punteggio compreso tra 0% e 100%.

I punteggi dei tre orizzonti vengono mediati con i pesi configurati. Infine il
segnale viene smussato in modo asimmetrico:

- riduzione dell'esposizione piu rapida (`alpha_down=0.60`);
- aumento dell'esposizione piu graduale (`alpha_up=0.30`);
- target operativo arrotondato, per default, al 5%.

Il segnale calcolato a fine mese T viene applicato al rendimento del mese T+1.

## File

- `app.py`: interfaccia Streamlit
- `risk_indicator.py`: motore del Layer 1
- `dual_momentum.py`: funzioni condivise e base per il futuro Layer 2
- `requirements.txt`: dipendenze

## Avvio

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Avvertenza

Strumento di ricerca e supporto decisionale. I backtest storici non
garantiscono risultati futuri.
