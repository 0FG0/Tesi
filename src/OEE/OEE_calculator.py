"""
L'OEE (Overall Equipment Effectiveness) è un indicatore usato nella produzione industriale
per misurare quanto una macchina o linea produttiva è efficiente.
Serve per capire quanto della capacità produttiva teorica viene realmente sfruttata.

L'OEE è il prodotto di tre fattori:

1) Disponibilità -> quanto tempo la macchina è realmente operativa rispetto al tempo programmato.  
    TEMPO PROGRAMMATO - FERMI / TEMPO PROGRAMMATO

2) Performance -> quanto la macchina lavora vicino alla sua velocità teorica.
    TEMPO TEORICO / TEMPO EFFETTIVO

3) Qualità -> percentuale di pezzi buoni rispetto ai pezzi prodotti.
    PEZZI BUONI / PEZZI TOTALI

OEE = Disponibilità * Performance * Qualità

Viene usato per individuare le inefficienze di produzione, monitorare le prestazioni
delle macchine e migliorare processi e pianificazione della produzione.

sono stati fatti tre files:

OEE CALCULATOR:
calcola l'OEE reale per ogni riga del dataset, dopo di che calcola l'OEE finale come 
prodotto delle tre componenti disponibilità, performance e qualità e classifica l'OEE in:
Critico, Accettabile o Ottimo. inoltre, genera un alert se l'OEE è sotto la soglia 
accettabile o se una delle componenti è sotto una soglia critica.

OEE FEATURE ENGINEERING:
Questo file prepara le feature da usare nel file di regressione.

OEE REGRESSION:
addestra i modelli per stimare prima che un ordine di lavoro venga eseguito quale sarà 
l'OEE previsto.
"""

import pandas as pd
import numpy as np

# standard OEE thresholds for manufacturing industry
SOGLIA_OTTIMO  = 0.85
SOGLIA_ACCETTABILE  = 0.65

# components
def calcola_disponibilita(df: pd.DataFrame) -> pd.Series:
    # Availability = (Tempo Programmato - Fermi) / Tempo Programmato
    tempo_prog = df["Tempo Attrezz. ORE"] + df["Tempo Lavoraz. ORE"]
    fermi = df["Durata Soste Ore"].fillna(0)
    tempo_op = (tempo_prog - fermi).clip(lower=0)

    disp = np.where(tempo_prog > 0, tempo_op / tempo_prog, np.nan)
    return pd.Series(disp, index=df.index, name="OEE_Disponibilita").clip(0, 1)


def calcola_performance(df: pd.DataFrame) -> pd.Series:
    # Performance = Tempo_Teorico / Tempo_Effettivo = 1 / Indice_Inefficienza
    # esempio:
    # 1.0  macchina perfettamente in linea col teorico
    # 0.67 macchina al 67% della velocità teorica
    if "Indice_Inefficienza" in df.columns:
        perf = 1.0 / df["Indice_Inefficienza"].replace(0, np.nan)
    else:
        perf = df["Tempo_Teorico_TOT_ORE"] / df["Tempo Lavoraz. ORE"].replace(0, np.nan)

    return perf.clip(0, 1).rename("OEE_Performance")


def calcola_qualita(df: pd.DataFrame) -> pd.Series:
    # Qualità = Pezzi Buoni / Tot Pezzi Conteggiati
    scarti = (
        df["Scarti Materiale"].fillna(0)
        + df["Scarti Lavoraz."].fillna(0)
        + df["Pezzi Ripassati"].fillna(0)
    )
    buoni = (df["Tot pezzi Contegg."] - scarti).clip(lower=0)

    qual = np.where(df["Tot pezzi Contegg."] > 0, buoni / df["Tot pezzi Contegg."], np.nan)
    return pd.Series(qual, index=df.index, name="OEE_Qualita").clip(0, 1)

# final OEE
def calcola_oee(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["OEE_Disponibilita"] = calcola_disponibilita(out)
    out["OEE_Performance"]   = calcola_performance(out)
    out["OEE_Qualita"]       = calcola_qualita(out)
    out["OEE"]               = out["OEE_Disponibilita"] * out["OEE_Performance"] * out["OEE_Qualita"]

    out["OEE_Classe"] = pd.cut(
        out["OEE"],
        bins=[-np.inf, SOGLIA_ACCETTABILE, SOGLIA_OTTIMO, np.inf],
        labels=["Critico", "Accettabile", "Ottimo"]
    )

    return out

# alert
def genera_alert(df_oee: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (df_oee["OEE"] < SOGLIA_ACCETTABILE)
        | (df_oee["OEE_Performance"] < 0.70)
        | (df_oee["OEE_Disponibilita"] < 0.80)
        | (df_oee["OEE_Qualita"] < 0.95)
    )
    alert = df_oee[mask].copy()
    alert["Alert_Motivo"] = ""

    alert.loc[alert["OEE"] < SOGLIA_ACCETTABILE, "Alert_Motivo"] += "OEE_CRITICO "
    alert.loc[alert["OEE_Performance"] < 0.70, "Alert_Motivo"] += "PERFORMANCE_BASSA "
    alert.loc[alert["OEE_Disponibilita"] < 0.80, "Alert_Motivo"] += "DISPONIBILITA_BASSA "
    alert.loc[alert["OEE_Qualita"] < 0.95, "Alert_Motivo"] += "QUALITA_BASSA "

    alert["Alert_Motivo"] = alert["Alert_Motivo"].str.strip()
    return alert