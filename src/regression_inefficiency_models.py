import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from feature_engineering import pipeline_inefficienza

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "koepfer_160_2.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "regression", "best_regressione_inefficienza.pkl")
PARAMS_PATH = os.path.join(PROJECT_ROOT, "models", "regression", "parametri_preprocessing_regressione_inefficienza.pkl")

# loading the clean datas of KOEPFER 160/2 machine
df = pd.read_csv(DATA_PATH)

if "ARTICOLO" in df.columns:
    df["ARTICOLO"] = df["ARTICOLO"].fillna("MISSING_ARTICOLO").astype(str)

# ARTICOLO_grouped + OHE
counts = df['ARTICOLO'].value_counts()
TOP_N_ARTICOLI = 20
articoli_top = counts.head(TOP_N_ARTICOLI).index
df['ARTICOLO_grouped'] = df['ARTICOLO'].where(
    df['ARTICOLO'].isin(articoli_top), 
    other='ALTRO'
)
df['ARTICOLO_grouped'] = df['ARTICOLO_grouped'].fillna('ALTRO').astype(str)

# applying feature engineering
df = pipeline_inefficienza(df)

cols_to_drop = [
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
]

y = df["Indice_Inefficienza"]
X = df.drop(columns=cols_to_drop, errors="ignore")

# find couples with high correlation
print("ANALISI CORRELAZIONI TRA FEATURES:")
corr_matrix = X.select_dtypes(include=[np.number]).corr().abs()

high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.7:
            high_corr.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

print("Features fortemente correlate (> 0.7):")
if high_corr:
    for feat1, feat2, corr in sorted(high_corr, key=lambda x: x[2], reverse=True):
        print(f"  {feat1} <-> {feat2}: {corr:.3f}")
else:
    print("  Nessuna coppia con correlazione > 0.7")

# train test split
if "Data_Ora_Fine" in df.columns:
    idx_sorted = df.sort_values("Data_Ora_Fine").index
    split_idx = int(len(idx_sorted) * 0.8)
    X_train, X_test = X.loc[idx_sorted[:split_idx]].copy(), X.loc[idx_sorted[split_idx:]].copy()
    y_train, y_test = y.loc[idx_sorted[:split_idx]].copy(), y.loc[idx_sorted[split_idx:]].copy()
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# cv strategy (consistent between grid search and cv esterna)
if "Data_Ora_Fine" in df.columns:
    cv_interna = TimeSeriesSplit(n_splits=5)
else:
    cv_interna = KFold(n_splits=5, shuffle=True, random_state=42)

# columns identification
categorical_cols = [
    col for col in [
        "ARTICOLO_grouped",
        "FASE",
        "Cod CIC",
        "C.d.L. Prev",
        "Descrizione Centro di Lavoro previsto"
    ]
    if col in X.columns
]

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

for col in categorical_cols:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype('string').fillna('MISSING')
    if col in X_test.columns:
        X_test[col] = X_test[col].astype('string').fillna('MISSING')

# preprocessing
# for linear models (scaling + one hot)
preprocessor_linear = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical_cols)
    ],
    remainder="drop" 
)

# for tree models (just one hot, no scaling)
preprocessor_tree = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="drop" 
)

# base models
base_models = {
    "Linear Regression": Pipeline([
        ("preprocessor", preprocessor_linear),
        ("model", LinearRegression())
    ]),
    "Ridge": Pipeline([
        ("preprocessor", preprocessor_linear),
        ("model", Ridge())
    ]),
    "Lasso": Pipeline([
        ("preprocessor", preprocessor_linear),
        ("model", Lasso())
    ]),
    "Decision Tree": Pipeline([
        ("preprocessor", preprocessor_tree),
        ("model", DecisionTreeRegressor(random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor_tree),
        ("model", RandomForestRegressor(random_state=42))
    ]),
    "XGBoost": Pipeline([
        ("preprocessor", preprocessor_tree),
        ("model", XGBRegressor(random_state=42, verbosity=0))
    ])
}

def valuta_modello_regressione(name, y_true, y_pred):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"R²   : {r2:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")

    return {
        "Model": name,
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
    }

# comparison of base models
results = []
trained_models = {}
print("\nBASE MODELS:")
for name, model in base_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = valuta_modello_regressione(name, y_test, y_pred)
    results.append(result)
    trained_models[name] = model 


# random forest grid search
print(" \nRANDOM FOREST GRID SEARCH")
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor_tree),
    ("model", RandomForestRegressor(random_state=42))
])

rf_param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [2, 5]
}

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=cv_interna,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train, y_train)

print("Miglior parametri RF:", rf_grid.best_params_)

best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)
result_rf = valuta_modello_regressione("Random Forest Ottimizzata", y_test, y_pred_rf)
y_pred_train_rf = best_rf.predict(X_train)
train_rmse_rf = np.sqrt(mean_squared_error(y_train, y_pred_train_rf))
test_rmse_rf = result_rf["RMSE"]
print(f"  Overfitting check -> Train RMSE: {train_rmse_rf:.4f} | Test RMSE: {test_rmse_rf:.4f} | Gap: {test_rmse_rf - train_rmse_rf:.4f}")
results.append(result_rf)
trained_models["Random Forest Ottimizzata"] = best_rf

# xgboost grid search
print("\nXGBOOST GRID SEARCH")
xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor_tree),
    ("model", XGBRegressor(
        random_state=42,
        objective="reg:squarederror",
        eval_metric="rmse"
    ))
])

xgb_param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [3, 6, 10],
    "model__learning_rate": [0.01, 0.1],
    "model__subsample": [0.8, 1.0]
}

xgb_grid = GridSearchCV(
    xgb_pipeline,
    xgb_param_grid,
    cv=cv_interna,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)
xgb_grid.fit(X_train, y_train)

print("Migliori parametri XGB:", xgb_grid.best_params_)

best_xgb = xgb_grid.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
result_xgb = valuta_modello_regressione("XGBoost Ottimizzata", y_test, y_pred_xgb)
y_pred_train_xgb = best_xgb.predict(X_train)
train_rmse_xgb = np.sqrt(mean_squared_error(y_train, y_pred_train_xgb))
test_rmse_xgb = result_xgb["RMSE"]
print(f"  Overfitting check -> Train RMSE: {train_rmse_xgb:.4f} | Test RMSE: {test_rmse_xgb:.4f} | Gap: {test_rmse_xgb - train_rmse_xgb:.4f}")
results.append(result_xgb)
trained_models["XGBoost Ottimizzata"] = best_xgb

# final comparison
results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print("\nCONFRONTO FINALE MODELLI:")
print(results_df.to_string(index=False))

#save model
best_model_name = min(
    results,
    key=lambda x: (x["RMSE"], -x["R2"])
)["Model"]

best_model = trained_models[best_model_name]
best_rmse = next(r["RMSE"] for r in results if r["Model"] == best_model_name)
print(f"\nModello migliore: {best_model_name}  (RMSE: {best_rmse:.4f})")

# cross-validation esterna (stima robusta delle performance)
if "Data_Ora_Fine" in df.columns and len(X) >= 10:
    cv_esterna = TimeSeriesSplit(n_splits=5)
    cv_descrizione = "TimeSeriesSplit 5-fold"
else:
    cv_esterna = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_descrizione = "KFold 5-fold"

cv_mae_neg = cross_val_score(best_model, X, y, cv=cv_esterna, scoring="neg_mean_absolute_error", n_jobs=-1)
cv_mse_neg = cross_val_score(best_model, X, y, cv=cv_esterna, scoring="neg_mean_squared_error", n_jobs=-1)
cv_r2 = cross_val_score(best_model, X, y, cv=cv_esterna, scoring="r2", n_jobs=-1)

cv_mae = -cv_mae_neg
cv_rmse = np.sqrt(-cv_mse_neg)

print(f"\nSTIMA ROBUSTA CV ESTERNA ({cv_descrizione}) - media ± std:")
print(f"  MAE:  {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")
print(f"  RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
print(f"  R²:   {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(best_model, MODEL_PATH)

parametri = {
    "articoli_top": articoli_top.tolist(),
    "top_n_articoli": TOP_N_ARTICOLI,
    "modello_scelto": best_model_name,
}
joblib.dump(parametri, PARAMS_PATH)

print(f"Modello salvato in {MODEL_PATH}")
print(f"Parametri salvati in {PARAMS_PATH}")

# ****** APPUNTI ******

# R² (Coefficiente di Determinazione) -> quanto il modello si avvicina al valore reale,
# MSE (Mean Squared Error) -> errore medio al quadrato, di quanto si discosta dal valore reale
# RMSE (Root Mean Squared Error) -> errore medio, di quanto si discosta dal valore reale

# esempio di R²:
# R² = 1 valore perfetto, 0 media, < 0 peggio della media 
# esempio di MSE: 
# Se il valore reale della variabile che voglio predire è 
# 1.2 e il modello prevede 1.5 -> errore = 0.3 -> MSE = 0.03
# esempio di RMSE: 
# Se il valore reale della variabile che voglio predire è 1.2 e il modello prevede 1.5 -> RMSE = 0.3 

# siccome nei dati ci sono gli articoli che hanno quasi tutti un numero diverso
# e questo nell'addestramento con onehotencoding andrebbe a creare centinaia di colonne
# ma allo stesso tempo non posso droppare l'intera colonna degli articoli in quanto
# potrebbe comunque essere che alcune anomalie di produzione si verifichino solo quando ci 
# sono determinati articoli per cui è comunque una colonna da cui il modello potrebbe imparare
# allora droppo la colonna degli articoli e ne creo un'altra che abbia solo gli articoli che 
# vengono ripetuti all'interno dei dati per più di 3 volte calcolandone poi la frequenza
# in modo tale che sia studiabile dal modello
# altre colonne che droppo oltre alla colonna che voglio predire sono
# le colonne tempo lavoro e tempo teorico in quanto sono le colonne che formano 
# la colonna di inefficienze che voglio predire, wo perchè non è utile in alcun modo 
# all'addestramento, ID DAD per lo stesso motivo, Descrizione Macchina in quanto stiamo 
# studiando sempre la stessa macchina, C.d.L. Effett, Data_Ora_Fine che viene
# già estratta come feature temporali

# nel preprocessing negli alberi non ho messo il drop="if_binary" in quanto 
# Per gli alberi è meglio non droppare mai la prima colonna. 
# Gli alberi traggono vantaggio dalla ridondanza perché possono scegliere di fare
# uno "split" su qualsiasi categoria in modo più diretto.
# per gli alberi non si usa lo StandardScaler() in quanto gli alberi di decisione 
# non variano in base alla scala. Non gli importa se un numero è 0.001 o 1.000.000,
# perché lavorano per soglie (es. Età > 30?). 
# Usando passthrough si risparmiano calcoli inutili.

# quando si runna il codice compare 
# UserWarning: Found unknown categories in columns [0] during transform. These unknown categories will be encoded as all zeros
# questo è dovuto dal fatto che 
# OneHotEncoder(drop="first", handle_unknown="ignore")
# con handle_unknown="ignore" le categorie sconosciute vengono trasformate tutte in zeri
# questo avviene perchè nella fase di training ovviamente non ci sono tutte le possibili variazioni
# dei dati dunque nel test appare una variazione che il modello non ha mai visto

# A differenza dell'MSE, che eleva al quadrato gli errori rendendone difficile l’interpretazione pratica, 
# l’RMSE fornisce una misura direttamente interpretabile: indica di quanto, in media, il modello 
# si discosta dal valore reale dell’indice. esempio l’RMSE ≈ 0.121 implica che l’errore medio di 
# previsione è pari a circa 0.12 punti di inefficienza. ipotizzando che l’Indice di Inefficienza 
# ha media ≈ 1.18 e deviazione standard ≈ 0.28, un errore medio di 0.12 rappresenta una 
# deviazione contenuta rispetto alla variabilità naturale del processo. 
# Questo confermerebbe la buona capacità predittiva del modello.

# -------------------------------------------------------------------------------------------

# Analisi dei risultati del modello di regressione sull’Indice di Inefficienza
 
# L’analisi dei risultati evidenzia una differenza molto marcata tra i modelli lineari e i modelli
# basati su alberi decisionali.
# I modelli lineari mostrano prestazioni estremamente scarse, mentre i modelli ad albero riescono 
# a catturare una parte significativa della variabilità dell’indice di inefficienza.
 
# Modelli lineari:
# I modelli lineari hanno prodotto risultati fortemente negativi:
# Linear Regression → R² ≈ -522.93
# Ridge → R² ≈ -528.31
# Lasso → R² ≈ -0.0007
 
# Un valore di R² negativo indica che il modello performa peggio di una semplice media dei dati, 
# dimostrando che la relazione tra le variabili operative e l’indice di inefficienza non è 
# descrivibile tramite una relazione lineare.
# In particolare, i valori estremamente negativi ottenuti da Linear Regression e Ridge suggeriscono 
# che la presenza di multicollinearità tra le variabili numeriche influisce fortemente sulla 
# stabilità del modello lineare. L’analisi delle correlazioni tra le feature ha infatti evidenziato 
# diverse coppie di variabili fortemente correlate, ad esempio:
 
# Pezzi da Avanzare ↔ Tot pezzi Contegg. → correlazione 1.000
# mese ↔ settimana_anno → 0.996
# Tot pezzi Contegg. ↔ Qta totale su AS/400 → 0.791
 
# Queste forti correlazioni rendono difficile per i modelli lineari attribuire correttamente il peso 
# alle variabili, causando instabilità nelle stime e prestazioni molto scarse.
# Il modello Lasso riesce a comportarsi leggermente meglio grazie al meccanismo di regolarizzazione L1, 
# che tende a ridurre il peso delle variabili meno rilevanti, ma le prestazioni restano comunque 
# molto limitate.
 
# Modelli ad albero:
# I modelli basati su alberi decisionali mostrano invece prestazioni significativamente migliori:
 
# Decision Tree → R² ≈ 0.356
# Random Forest → R² ≈ 0.588
# XGBoost → R² ≈ 0.576
 
# Questi risultati indicano che circa il 58% della variabilità dell’indice di inefficienza può essere 
# spiegato dalle variabili operative disponibili nel dataset.
# Questo comportamento suggerisce che il sistema produttivo presenta relazioni non lineari e interazioni
# tra variabili, che i modelli ad albero riescono a catturare meglio rispetto ai modelli lineari.
# Gli algoritmi ensemble come Random Forest e XGBoost riescono infatti a modellare interazioni complesse 
# tra le feature, migliorando la capacità predittiva.
 
# Ottimizzazione dei modelli:
# È stata successivamente applicata una procedura di Grid Search con cross-validation interna per 
# ottimizzare i parametri dei modelli Random Forest e XGBoost.
# Random Forest ottimizzata
# Parametri ottimali trovati:
 
# n_estimators = 400
# max_depth = 10
# min_samples_split = 2
# min_samples_leaf = 2
# Prestazioni:
# R² ≈ 0.569
# RMSE ≈ 0.215
# MAE ≈ 0.157
 
# Il confronto tra errore di training e test mostra:
# Train RMSE ≈ 0.064
# Test RMSE ≈ 0.215
 
# Questo evidenzia la presenza di un certo grado di overfitting, tipico dei modelli basati su alberi 
# quando il dataset non è molto grande.
# XGBoost ottimizzata
# Parametri ottimali:
 
# n_estimators = 400
# max_depth = 6
# learning_rate = 0.01
# subsample = 0.8
# Prestazioni:
# R² ≈ 0.565
# RMSE ≈ 0.216
# MAE ≈ 0.154
 
# Anche in questo caso si osserva un gap significativo tra training e test:
 
# Train RMSE ≈ 0.024
# Test RMSE ≈ 0.216
 
# Questo indica che il modello apprende molto bene il training set ma generalizza meno sui dati non visti.
 
# Confronto finale tra modelli
# Il confronto finale mostra che il miglior modello risulta essere Random Forest base, con:
 
# R² ≈ 0.588
# RMSE ≈ 0.210
# MAE ≈ 0.156
 
# Curiosamente, i modelli ottimizzati tramite Grid Search non hanno migliorato le prestazioni 
# rispetto alle versioni base. Questo fenomeno è abbastanza comune quando:
# il dataset è relativamente piccolo
# i parametri di default sono già vicini alla configurazione ottimale
# Validazione con Cross Validation esterna
# Per ottenere una stima più robusta delle prestazioni del modello è stata applicata una cross 
# validation esterna con TimeSeriesSplit a 5 fold, che mantiene l’ordine temporale dei dati.
 
# Le metriche medie ottenute sono:
 
# MAE ≈ 0.146 ± 0.036
# RMSE ≈ 0.190 ± 0.031
# R² ≈ 0.474 ± 0.112
 
# Questo significa che:
# il modello riesce mediamente a spiegare circa il 47% della variabilità dell’inefficienza su dati 
# non visti
# l’errore medio di previsione è circa 0.19 punti di inefficienza
# La deviazione standard relativamente elevata del valore di R² indica che le prestazioni possono 
# variare tra i diversi intervalli temporali del dataset, suggerendo che il processo produttivo 
# può presentare dinamiche variabili nel tempo.
 
# Interpretazione pratica dell'errore:
# Considerando che l’indice di inefficienza assume valori tipicamente attorno a 1, un errore medio di 
# circa 0.19 punti indica che il modello riesce a fornire una stima ragionevolmente vicina al valore reale.
# Questo suggerisce che l’inefficienza non è completamente casuale, ma dipende in misura significativa 
# dalle variabili operative disponibili, anche se rimane una parte di variabilità che non può essere 
# spiegata dal dataset attuale.