# Layer 1 - Global Risk Indicator

[![Critical Layer 1 backtest](https://github.com/takeshiromanov/MKT_riskOn/actions/workflows/critical-backtest.yml/badge.svg)](https://github.com/takeshiromanov/MKT_riskOn/actions/workflows/critical-backtest.yml)

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

## Overlay recovery sperimentale

L'app permette di attivare la variante one-shot per studiare un rientro piu
rapido dopo i minimi. Il percorso risk-off del modello core prima del trigger
non viene modificato. Quando il target e sotto il 50%, l'overlay apre una
posizione pilota se sono contemporaneamente positivi il momentum relativo a 1
mese, il miglioramento del momentum a 3 mesi e la variazione del punteggio
grezzo. Con la conferma del momentum relativo a 3 mesi converge piu rapidamente
verso il 70% di esposizione.

La posizione pilota puo scattare una sola volta nello stesso episodio risk-off
e resta attiva almeno due mesi, salvo un deterioramento forte. L'overlay si
riarma soltanto dopo due mesi del core sopra il 70%. Il backtest conserva anche
la prima recovery ripetibile come benchmark: non e esposta nell'app perche
produceva troppi whipsaw.

I parametri sono deliberatamente predefiniti, non scelti cercando il massimo
rendimento storico. L'overlay resta sperimentale finche non supera i gate su
drawdown, whipsaw, rendimento nei tre mesi dopo i minimi e ritardo di rientro.

## File

- `app.py`: interfaccia Streamlit
- `risk_indicator.py`: motore del Layer 1
- `critical_backtest.py`: test di falsificazione e robustezza del Layer 1
- `dual_momentum.py`: funzioni condivise e base per il futuro Layer 2
- `test_*.py`: test automatici, inclusi lag del segnale e turnover
- `requirements.txt`: dipendenze

## Avvio

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Backtest critico

Il backtest non cerca il set di parametri con il rendimento piu alto. Confronta
sei regole sulla stessa finestra e con il medesimo capitale iniziale:

1. buy & hold azionario;
2. absolute momentum binario a 12 mesi;
3. voto graduato a 3/6/12 mesi;
4. Layer 1 continuo;
5. prima recovery ripetibile;
6. recovery one-shot anti-whipsaw.

Il segnale di fine mese T viene sempre applicato a T+1. Il turnover comprende
anche il ribilanciamento dovuto alla deriva dei pesi. Gli output misurano
drawdown e relativa durata, ritardo negli episodi di stress, whipsaw, turnover,
calibrazione dei bucket di esposizione, una griglia di 81 configurazioni
vicine per il core e 12 varianti predefinite dell'overlay recovery. Le griglie
servono a scoprire fragilita, non a selezionare l'ottimo.
Il mese di calendario ancora in corso viene escluso automaticamente.

```bash
python critical_backtest.py --output reports/latest
python -m unittest discover -v
```

Vengono generati CSV di audit, tre grafici, un manifest con fonte e parametri e
una sintesi Markdown. La cartella `reports/` non viene versionata.

### Storico esteso

La storia comune di URTH/BIL parte solo nel 2012 circa: e insufficiente per
validare definitivamente un filtro di regime perche esclude dot-com e 2008.
Per ripetere esattamente lo stesso test con serie storiche piu lunghe:

```bash
python critical_backtest.py \
  --prices-csv data/msci_world_tbill_monthly.csv \
  --output reports/extended
```

Il CSV deve avere tre colonne: `DATE`, `EQUITY` e `CASH`; le ultime due devono
essere indici total-return o prezzi adjusted, entrambi in USD e a frequenza
mensile (o ricampionabile a fine mese).

## Avvertenza

Strumento di ricerca e supporto decisionale. I backtest storici non
garantiscono risultati futuri.
