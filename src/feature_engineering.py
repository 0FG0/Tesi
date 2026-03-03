import pandas as pd

# transforming time in numbers (es. 0 = monday)
# in order to see if there are patterns in a certain timeframe
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Data_Ora_Fine"] = pd.to_datetime(df["Data_Ora_Fine"])
    df["giorno_settimana"] = df["Data_Ora_Fine"].dt.weekday        
    df["mese"] = df["Data_Ora_Fine"].dt.month
    df["settimana_anno"] = df["Data_Ora_Fine"].dt.isocalendar().week.astype(int)
    df["anno"] = df["Data_Ora_Fine"].dt.year
    return df

# represents relationships between quantities and working times.
def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    # ratio of rejected pieces to total, index of process quality
    tot = df["Tot pezzi Contegg."].replace(0, pd.NA)
    df["ratio_scarti"] = (df["Scarti Materiale"] + df["Scarti Lavoraz."]) / tot
    
    # Ratio of setup time to AS400 machine time
    tempo_macc = df["Tempo Macc AS400 ORE"].replace(0, pd.NA)
    df["ratio_attr_macc"] = df["Tempo Attr AS400 ORE"] / tempo_macc
    return df

# rolling mean and rolling std 
# in order to see if in the last n processes there's been a trend
# lag features = how important was the inefficiency in the previous processing
def _add_rolling_and_lag(df: pd.DataFrame, colonna_base: str, window: int = 10, lags: list = [1, 2, 3]) -> pd.DataFrame:
    df = df.sort_values("Data_Ora_Fine").copy()
    past = df[colonna_base].shift(1)

    df[f"rolling_mean_{window}"] = past.rolling(window=window, min_periods=1).mean()
    df[f"rolling_std_{window}"]  = past.rolling(window=window, min_periods=1).std()

    for lag in lags:
        df[f"lag_{lag}"] = df[colonna_base].shift(lag)
    return df

# PIPELINE 1 - target: regression Indice_Inefficienza
def pipeline_inefficienza(df: pd.DataFrame, window: int = 10, lags: list = [1, 2, 3]) -> pd.DataFrame:
    df = add_time_features(df)
    df = _add_rolling_and_lag(df, colonna_base="Indice_Inefficienza", window=window, lags=lags)
    df = add_ratio_features(df)
    df = df.dropna(subset=[f"lag_{l}" for l in lags])
    return df

# PIPELINE 2 - target: classification Indice_Inefficienza
def pipeline_classificazione(df: pd.DataFrame, window: int = 10, lags: list = [1, 2, 3]) -> pd.DataFrame:
    df = add_time_features(df)
    df = _add_rolling_and_lag(df, colonna_base="Indice_Inefficienza", window=window, lags=lags)
    df = add_ratio_features(df)
    df = df.dropna(subset=[f"lag_{l}" for l in lags])
    return df

# PIPELINE 3 - target: Tempo Lavoraz. ORE
def pipeline_tempo(df: pd.DataFrame, window: int = 10, lags: list = [1, 2, 3]) -> pd.DataFrame:
    df = add_time_features(df)
    df = _add_rolling_and_lag(df, colonna_base="Tempo Lavoraz. ORE", window=window, lags=lags)
    df = add_ratio_features(df)
    df = df.dropna(subset=[f"lag_{l}" for l in lags])
    return df


# ****** APPUNTI ******
# Il feature engineering è il processo di trasformazione dei dati grezzi
# in variabili (feature) che rendono gli algoritmi di machine learning più efficaci. 
# I suoi obiettivi principali sono due: 
# Rendere i dati compatibili: gli algoritmi non accettano dati testuali o date
# dunque con il feature engineering trasformiamo i dati in formati leggibili dal modello.
# Migliorare le prestazioni: aiuta il modello a cogliere sfumature e relazioni nascoste
# nei dati che altrimenti passerebbero inosservate. 
# bisogna stare attenti ad alcune cose:
# non usare mai la colonna target (Indice_Inefficienza) come input diretto.
# le feature "lag" e "rolling" devono usare solo valori passati (shift(1)),
# mai il valore corrente della riga che stiamo cercando di predire.
# Questo file va applicato prima dello split train/test.
# QUESTO FILE SUPPORTA DUE USE CASE:
# 1. target = "Indice_Inefficienza"  → pipeline_inefficienza()
# 2. target = "Tempo Lavoraz. ORE"   → pipeline_tempo()
# La differenza è solo nella colonna usata per lag e rolling features,
# perché Indice_Inefficienza è calcolato da Tempo Lavoraz. ORE:
# usare l'uno come base quando l'altro è il target sarebbe leakage indiretto.