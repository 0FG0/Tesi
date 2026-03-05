# Main.py 
# loads saved models and it produces prediction on new datas

# how to use it:
# with python main.py it execute the datas from the training so data/processed/koepfer_160_2.csv and 
# produce an output in the standard file: outputs/predizioni.csv

# with python main.py --input (*path of new datas*) it will predict the new datas and produce an output
# in the standard file: outputs/predizioni.csv

# with python main.py --input (*path of new datas*) --output (*path of wanted outputs*)  it will predict 
# the new datas and produce an output where we want the outputs to be

# Used models:
# 1. Regressione Indice_Inefficienza      ->  models/regression/best_regressione_inefficienza.pkl
# 2. Regressione Tempo Lavoraz. ORE       ->  models/regression/best_regressione_time.pkl
# 3. Classificazione anomaly-oriented     ->  models/classification/best_classificazione_anomaly.pkl
# 4. Classificazione anomaly-oriented-BD  ->  models/classification/best_classificazione_anomaly_BD.pkl

# outputs structure:
# same columns of the data imported plus the four colums of models ouputs

# =============================== NOTE ===============================
# the file main is currently running with the same data that models did train with
# so obviously the results are extremely good, but they are false since 80% of the data 
# was used for training, and 20% for testing, but the whole dataset is the same, 
# so we are predicting on datas that the model already saw during training.

import pandas as pd
import numpy as np
import argparse
import sys
import os
import warnings
import joblib
warnings.filterwarnings("ignore")

# adding src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import pipeline_inefficienza, pipeline_tempo, pipeline_classificazione

# models path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

PATHS = {
    "regressione_inefficienza": {
        "model": os.path.join(MODELS_DIR, "regression", "best_regressione_inefficienza.pkl"),
        "params": os.path.join(MODELS_DIR, "regression", "parametri_prepocessing_regressione_inefficienza.pkl"),
    },
    "regressione_tempo": {
        "model": os.path.join(MODELS_DIR, "regression", "best_regressione_time.pkl"),
        "params": os.path.join(MODELS_DIR, "regression", "parametri_preprocessing_tempo.pkl"),
    },
    "classificazione_anomaly": {
        "model": os.path.join(MODELS_DIR, "classification", "best_classificazione_anomaly.pkl"),
        "params": os.path.join(MODELS_DIR, "classification", "parametri_classificazione_anomaly.pkl"),
    },
    "classificazione_anomaly_BD": {
        "model": os.path.join(MODELS_DIR, "classification", "best_classificazione_anomaly_BD.pkl"),
        "params": os.path.join(MODELS_DIR, "classification", "parametri_classificazione_anomaly_BD.pkl"),
    },

}

LABELS = {0: "NORMALE", 1: "ATTENZIONE", 2: "ANOMALIA"}
LABELS_BINARY = {0: "NORMALE", 1: "ANOMALIA"}

# columns to drop
COLS_TO_DROP = [
    "Indice_Inefficienza",
    "Tempo Lavoraz. ORE",
    "Tempo_Teorico_TOT_ORE",
    "WO",
    "ARTICOLO",
    "Descrizione Articolo",
    "ID DAD",
    "Descrizione Macchina",
    "C.d.L. Effett",
    "Data_Ora_Fine",
    "classe",
]

# support functions
# it loads the model and parameters
def carica_modello(nome: str):
    paths = PATHS[nome]
    if not os.path.exists(paths["model"]):
        raise FileNotFoundError(
            f"Modello '{nome}' non trovato in {paths['model']}.\n"
            f"Assicurati di aver eseguito prima il file di training corrispondente."
        )
    model = joblib.load(paths["model"])
    params = joblib.load(paths["params"])
    return model, params

# article encoding (ARTICOLO_grouped + OHE)
def applica_encoding_articolo(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    if "ARTICOLO" not in df.columns:
        return df

    if "articoli_top" in params:
        df["ARTICOLO"] = df["ARTICOLO"].fillna("MISSING_ARTICOLO").astype(str)
        articoli_top = set(map(str, params["articoli_top"]))
        df["ARTICOLO_grouped"] = df["ARTICOLO"].where(
            df["ARTICOLO"].isin(articoli_top),
            other="ALTRO"
        )
        df["ARTICOLO_grouped"] = df["ARTICOLO_grouped"].fillna("ALTRO").astype(str)
        return df

    return df

# removes columns
def prepara_X(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=COLS_TO_DROP, errors="ignore")


def normalizza_categoriche_inferenza(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = [
        "ARTICOLO_grouped",
        "FASE",
        "Cod CIC",
        "C.d.L. Prev",
        "Descrizione Centro di Lavoro previsto",
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("MISSING")
    return df

# functions prediction
def predici_inefficienza(df_raw: pd.DataFrame) -> pd.Series:
    # predicts indice_inefficienza (regression)
    # if values > 1 the machine took more time then it should have had 
    model, params = carica_modello("regressione_inefficienza")

    df = df_raw.copy()
    df = applica_encoding_articolo(df, params)
    df = pipeline_inefficienza(df)
    df = normalizza_categoriche_inferenza(df)
    X  = prepara_X(df)

    predizioni = model.predict(X)
    return pd.Series(predizioni, index=df.index, name="Indice_Inefficienza_Predetto")

# predicts time
def predici_tempo(df_raw: pd.DataFrame) -> pd.Series:
    model, params = carica_modello("regressione_tempo")

    df = df_raw.copy()
    df = applica_encoding_articolo(df, params)
    df = pipeline_tempo(df)
    df = normalizza_categoriche_inferenza(df)
    X  = prepara_X(df)

    predizioni = model.predict(X)
    return pd.Series(predizioni, index=df.index, name="Tempo_Predetto_ORE")

# classifies every machine processing with the anomaly oriented model (binary: NORMALE / ANOMALIA)
def predici_classe_anomaly(df_raw: pd.DataFrame) -> pd.Series:
    model, params = carica_modello("classificazione_anomaly")

    soglia_anomalia = params.get("soglia_proba_anomalia", 0.30)

    df = df_raw.copy()
    df = applica_encoding_articolo(df, params)
    df = pipeline_classificazione(df)
    df = normalizza_categoriche_inferenza(df)
    X  = prepara_X(df)

    proba = model.predict_proba(X)
    # Binary model: column 0 = NORMALE, column 1 = ANOMALIA
    classi = np.where(proba[:, 1] >= soglia_anomalia, 1, 0)
    return pd.Series([LABELS_BINARY[c] for c in classi], index=df.index, name="Classe_Anomaly_Oriented")

# classifies every machine processing with the anomaly oriented Big-Data model (3-class)
def predici_classe_anomaly_BD(df_raw: pd.DataFrame) -> pd.Series:
    model, params = carica_modello("classificazione_anomaly_BD")

    df = df_raw.copy()
    df = applica_encoding_articolo(df, params)
    df = pipeline_classificazione(df)
    df = normalizza_categoriche_inferenza(df)
    X  = prepara_X(df)

    classi = model.predict(X)
    return pd.Series([LABELS[c] for c in classi], index=df.index, name="Classe_Anomaly_Oriented_Big_Data")

# Main function
def main(path_input: str, path_output: str = None):

    # load data
    print(f"\nCaricamento dati da: {path_input}")
    if not os.path.exists(path_input):
        print(f"ERRORE: file non trovato → {path_input}")
        sys.exit(1)

    df = pd.read_csv(path_input)
    print(f"Righe caricate: {len(df)}")

    # model prediction
    print("\nApplicazione modelli...")

    try:
        pred_inefficienza = predici_inefficienza(df)
        print("  [OK] Regressione Indice_Inefficienza")
    except Exception as e:
        print(f"  [ERRORE] Regressione Indice_Inefficienza: {e}")
        pred_inefficienza = pd.Series([np.nan] * len(df), name="Indice_Inefficienza_Predetto")

    try:
        pred_tempo = predici_tempo(df)
        print("  [OK] Regressione Tempo Lavoraz. ORE")
    except Exception as e:
        print(f"  [ERRORE] Regressione Tempo Lavoraz. ORE: {e}")
        pred_tempo = pd.Series([np.nan] * len(df), name="Tempo_Predetto_ORE")

    try:
        pred_anomaly = predici_classe_anomaly(df)
        print("  [OK] Classificazione anomaly-oriented")
    except Exception as e:
        print(f"  [ERRORE] Classificazione anomaly-oriented: {e}")
        pred_anomaly = pd.Series(["ERRORE"] * len(df), name="Classe_Anomaly_Oriented")

    try:
        pred_anomaly_BD = predici_classe_anomaly_BD(df)
        print("  [OK] Classificazione anomaly-oriented Big Data")
    except Exception as e:
        print(f"  [ERRORE] Classificazione anomaly-oriented Big Data: {e}")
        pred_anomaly_BD = pd.Series(["ERRORE"] * len(df), name="Classe_Anomaly_Oriented_Big_Data")

    # results table
    id_cols = ["WO", "FASE", "ARTICOLO", "Data_Ora_Fine", "Descrizione Macchina", 
               "Tempo_Teorico_TOT_ORE", "Tempo Lavoraz. ORE", "Indice_Inefficienza"]
    id_cols = [c for c in id_cols if c in df.columns]

    risultati = df[id_cols].copy()
    risultati = risultati.join(pred_inefficienza)
    risultati = risultati.join(pred_tempo)
    risultati = risultati.join(pred_anomaly)
    risultati = risultati.join(pred_anomaly_BD)

    print("\n" + "="*70)
    print("RISULTATI PREDIZIONI")
    print("="*70)
    print(risultati.to_string(index=False))

    print("\n" + "="*70)
    print("RIEPILOGO")
    print("="*70)

    # Regression Indice_Inefficienza
    if pred_inefficienza.dtype != object and "Indice_Inefficienza" in df.columns:
        pred_ineff_reset = pred_inefficienza.dropna()
        real_ineff = df.loc[pred_ineff_reset.index, "Indice_Inefficienza"]
        print(f"\n{'Indice_Inefficienza_Predetto':<40} {'Indice_Inefficienza_Reale'}")
        print(f"{'Media:':<12} {pred_ineff_reset.mean():<27.4f} Media:       {real_ineff.mean():.4f}")
        print(f"{'Min:':<12} {pred_ineff_reset.min():<27.4f} Min:         {real_ineff.min():.4f}")
        print(f"{'Max:':<12} {pred_ineff_reset.max():<27.4f} Max:         {real_ineff.max():.4f}")

    # Classification summary tables
    def _stampa_tabella_classificazione(nome_col, serie_pred, df_raw, nome_modello):
        if "ERRORE" in serie_pred.astype(str).values:
            return
        if "Indice_Inefficienza" not in df_raw.columns:
            print(f"\n{nome_col}: colonna Indice_Inefficienza non disponibile, skip confronto reale.")
            return

        try:
            params_cls = PATHS[nome_modello]["params"]
            params = joblib.load(params_cls)
            soglia_anomalia = params["soglia_anomalia"]
            soglia_attenzione = params.get("soglia_attenzione", None)
        except Exception as e:
            print(f"\n{nome_col}: impossibile caricare soglie ({e})")
            return

        if soglia_attenzione is not None:
            label_order = ["NORMALE", "ATTENZIONE", "ANOMALIA"]
        else:
            label_order = ["NORMALE", "ANOMALIA"]

        pred_cls = serie_pred.dropna().astype(str)
        valid_idx = pred_cls[pred_cls.isin(label_order)].index
        if len(valid_idx) == 0:
            print(f"\n{nome_col}: nessuna predizione valida disponibile.")
            return

        ineff = df_raw.loc[valid_idx, "Indice_Inefficienza"]
        if soglia_attenzione is not None:
            real_cls = pd.Series(np.where(
                ineff <= soglia_attenzione, "NORMALE",
                np.where(ineff <= soglia_anomalia, "ATTENZIONE", "ANOMALIA")
            ), index=valid_idx)
        else:
            real_cls = pd.Series(
                np.where(ineff <= soglia_anomalia, "NORMALE", "ANOMALIA"),
                index=valid_idx
            )
        pred_cls = pred_cls.loc[valid_idx]

        print(f"\n{nome_col}:")
        header = f"{'Classe':<14} {'Reale':>7} {'Predetto':>9} {'Corretti':>9} {'Acc. classe':>12}"
        print(header)
        print("-" * (len(header) - 2))

        tot_reale = tot_pred = tot_corretti = 0
        for label in label_order:
            n_reale = (real_cls == label).sum()
            n_predetto = (pred_cls == label).sum()
            n_corretti = ((real_cls == label) & (pred_cls == label)).sum()
            acc = n_corretti / n_reale * 100 if n_reale > 0 else 0.0
            print(f"{label:<14} {n_reale:>7} {n_predetto:>9} {n_corretti:>9} {acc:>11.1f}%")
            tot_reale += n_reale
            tot_pred += n_predetto
            tot_corretti += n_corretti

        acc_tot = tot_corretti / tot_reale * 100 if tot_reale > 0 else 0.0
        print("-" * (len(header) - 2))
        print(f"{'TOTALE':<14} {tot_reale:>7} {tot_pred:>9} {tot_corretti:>9} {acc_tot:>11.1f}%")

    _stampa_tabella_classificazione("Classe_Anomaly_Oriented", pred_anomaly, df, "classificazione_anomaly")
    _stampa_tabella_classificazione("Classe_Anomaly_Oriented_Big_Data", pred_anomaly_BD, df, "classificazione_anomaly_BD")

    # Time regression
    if pred_tempo.dtype != object and "Tempo Lavoraz. ORE" in df.columns and "Tempo_Teorico_TOT_ORE" in df.columns:
        predetto = pred_tempo.dropna()
        if len(predetto) == 0:
            print(f"\nTempo_Predetto_ORE:")
            print("  Nessuna predizione valida disponibile dopo le feature lag.")
        else:
            effettivo = df.loc[predetto.index, "Tempo Lavoraz. ORE"]
            teorico = df.loc[predetto.index, "Tempo_Teorico_TOT_ORE"]

            mask = effettivo != 0

            err_as400 = (effettivo - teorico).abs()
            err_ml = (effettivo - predetto).abs()

            mape_as400 = (err_as400[mask] / effettivo[mask]).mean() * 100
            mape_ml = (err_ml[mask] / effettivo[mask]).mean() * 100
            riduzione_mape = (mape_as400 - mape_ml) / mape_as400 * 100 if mape_as400 != 0 else 0.0

            n_campioni = len(effettivo)
            n_ml_meglio = (err_ml < err_as400).sum()
            pct_ml_meglio = n_ml_meglio / n_campioni * 100 if n_campioni > 0 else 0.0

            print(f"\nTempo_Predetto_ORE:")
            print(f"Campioni nel test set:                    {n_campioni}")
            print(f"Casi in cui ML è più preciso di AS400:    {n_ml_meglio} / {n_campioni}  ({pct_ml_meglio:.1f}%)")
            print(f"")
            print(f"Percentuale di quanto la previsione si discosta dal valore reale:")
            print(f"MAPE Tempo Teorico AS400:                 {mape_as400:.2f}%")
            print(f"MAPE Modello ML:                          {mape_ml:.2f}%")
            print(f"Riduzione MAPE:                           {riduzione_mape:.1f}%")
    elif pred_tempo.dtype != object:
        print(f"\nTempo_Predetto_ORE:")
        print(f"  Media: {pred_tempo.mean():.4f} ore  |  Min: {pred_tempo.min():.4f}  |  Max: {pred_tempo.max():.4f}")
    for nome_col, serie in [
        ("Classe_Anomaly_Oriented", pred_anomaly),
        ("Classe_Anomaly_Oriented_Big_Data", pred_anomaly_BD),
    ]:
        serie_valida = serie.dropna().astype(str)
        if serie.dtype == object and len(serie_valida) > 0 and "ERRORE" not in serie_valida.values:
            print(f"\n{nome_col}:")
            dist = serie_valida.value_counts()
            for label in ["NORMALE", "ATTENZIONE", "ANOMALIA"]:
                cnt = dist.get(label, 0)
                pct = cnt / len(serie_valida) * 100
                print(f"  {label:<12}: {cnt:>4}  ({pct:.1f}%)")

    if path_output is None:
        output_dir = os.path.join(BASE_DIR, "..", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        path_output = os.path.join(output_dir, "predizioni.csv")

    risultati.to_csv(path_output, index=False)
    print(f"\nRisultati salvati in: {path_output}")

# input
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predizione efficienza macchina KOEPFER 160/2"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(BASE_DIR, "data", "processed", "koepfer_160_2.csv"),  
        help="Percorso del CSV con i dati di input"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(BASE_DIR, "outputs", "predizioni.csv"),
        help="Percorso del CSV di output (default: outputs/predizioni.csv)"
    )

    args = parser.parse_args()
    main(args.input, args.output)




