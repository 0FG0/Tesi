import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
import os
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from feature_engineering import pipeline_tempo
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "koepfer_160_2.csv")
OUTPUT_PLOT_PATH = os.path.join(PROJECT_ROOT, "outputs", "times_compared_scatter.png")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "regression", "best_regressione_time.pkl")
PARAMS_PATH = os.path.join(PROJECT_ROOT, "models", "regression", "parametri_preprocessing_tempo.pkl")

# load data 
df = pd.read_csv(DATA_PATH)

# ARTICOLO_grouped + OHE
if "ARTICOLO" in df.columns:
    df["ARTICOLO"] = df["ARTICOLO"].fillna("MISSING_ARTICOLO").astype(str)

counts = df['ARTICOLO'].value_counts()
TOP_N_ARTICOLI = 20
articoli_top = counts.head(TOP_N_ARTICOLI).index
df['ARTICOLO_grouped'] = df['ARTICOLO'].where(
    df['ARTICOLO'].isin(articoli_top),
    other='ALTRO'
)
df['ARTICOLO_grouped'] = df['ARTICOLO_grouped'].fillna('ALTRO').astype(str)

# feature
df = pipeline_tempo(df)

# save real datas to compare them with models datas
confronto_cols = df[["Tempo Lavoraz. ORE", "Tempo_Teorico_TOT_ORE"]].copy()

cols_to_drop = [
    "Tempo Lavoraz. ORE",           
    "Indice_Inefficienza",          
    "Tempo_Teorico_TOT_ORE",        
    "WO",
    "ARTICOLO",
    "Descrizione Articolo",
    "ID DAD",
    "Descrizione Macchina",
    "C.d.L. Effett",
    "Data_Ora_Fine",
]

y = df["Tempo Lavoraz. ORE"]
X = df.drop(columns=cols_to_drop, errors="ignore")

# find couples with high correlation
print("ANALISI CORRELAZIONI TRA FEATURES (soglia > 0.7):")
corr_matrix = X.select_dtypes(include=[np.number]).corr().abs()
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.7:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

if high_corr:
    for f1, f2, c in sorted(high_corr, key=lambda x: x[2], reverse=True):
        print(f"  {f1} <-> {f2}: {c:.3f}")
else:
    print("  Nessuna coppia con correlazione > 0.7")

# train test split
# per evitare dataleakage, per evitare che il modello possa vedere lavorazioni 
# del giorno X nel training e lavorazioni del giorno X-1 nel test: 
if "Data_Ora_Fine" in df.columns:
    idx_sorted = df.sort_values("Data_Ora_Fine").index
    split_idx = int(len(idx_sorted) * 0.8)
    X_train, X_test = X.loc[idx_sorted[:split_idx]].copy(), X.loc[idx_sorted[split_idx:]].copy()
    y_train, y_test = y.loc[idx_sorted[:split_idx]].copy(), y.loc[idx_sorted[split_idx:]].copy()
    confronto_test = confronto_cols.loc[idx_sorted[split_idx:]].copy()
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    confronto_test = confronto_cols.loc[X_test.index].copy()

# se si volesse invece che i dati siano semplicemente random allora:
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# cv strategy (consistent between grid search and cv esterna)
if "Data_Ora_Fine" in df.columns:
    cv_interna = TimeSeriesSplit(n_splits=5)
else:
    cv_interna = KFold(n_splits=5, shuffle=True, random_state=42)

# columns identification
categorical_cols = [
    col for col in ["ARTICOLO_grouped", "FASE", "Cod CIC", "C.d.L. Prev", "Descrizione Centro di Lavoro previsto"]
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

# mape
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

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

# comparison of base models
results = []
trained_models = {}
print("\nBASE MODELS:")
for name, model in base_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape_val = mape(y_test.values, y_pred)

    results.append({"Model": name, "R2": r2, "MAE": mae, "RMSE": rmse, "MAPE%": mape_val})
    trained_models[name] = model

    print(f"\n{name}")
    print(f"  R²:    {r2:.4f}")
    print(f"  MAE:   {mae:.4f} ore")
    print(f"  RMSE:  {rmse:.4f} ore")
    print(f"  MAPE:  {mape_val:.2f}%")

# random forest grid search
print("\nRANDOM FOREST - GRID SEARCH")
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
print("Migliori parametri RF:", rf_grid.best_params_)

best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mape_rf = mape(y_test.values, y_pred_rf)

trained_models["Random Forest Ottimizzata"] = best_rf

mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f"\nRandom Forest Ottimizzata  ->  R²: {r2_rf:.4f}  |  MAE: {mae_rf:.4f} ore  |  RMSE: {rmse_rf:.4f} ore  |  MAPE: {mape_rf:.2f}%")
y_pred_train_rf = best_rf.predict(X_train)
train_rmse_rf = np.sqrt(mean_squared_error(y_train, y_pred_train_rf))
print(f"  Overfitting check -> Train RMSE: {train_rmse_rf:.4f} | Test RMSE: {rmse_rf:.4f} | Gap: {rmse_rf - train_rmse_rf:.4f}")
results.append({"Model": "Random Forest Ottimizzata", "R2": r2_rf, "MAE": mae_rf, "RMSE": rmse_rf, "MAPE%": mape_rf})

# xgboost grid search
print("\nXGBOOST - GRID SEARCH")
xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor_tree),
    ("model", XGBRegressor(
        random_state=42,
        objective="reg:squarederror",
        eval_metric="rmse",
        verbosity=0
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
r2_xgb = r2_score(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mape_xgb = mape(y_test.values, y_pred_xgb)

trained_models["XGBoost Ottimizzata"] = best_xgb

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
print(f"\nXGBoost Ottimizzata  ->  R²: {r2_xgb:.4f}  |  MAE: {mae_xgb:.4f} ore  |  RMSE: {rmse_xgb:.4f} ore  |  MAPE: {mape_xgb:.2f}%")
y_pred_train_xgb = best_xgb.predict(X_train)
train_rmse_xgb = np.sqrt(mean_squared_error(y_train, y_pred_train_xgb))
print(f"  Overfitting check -> Train RMSE: {train_rmse_xgb:.4f} | Test RMSE: {rmse_xgb:.4f} | Gap: {rmse_xgb - train_rmse_xgb:.4f}")
results.append({"Model": "XGBoost Ottimizzata", "R2": r2_xgb, "MAE": mae_xgb, "RMSE": rmse_xgb, "MAPE%": mape_xgb})

# final comparison
results_df = pd.DataFrame(results).sort_values("MAPE%", ascending=True)
print("\nCONFRONTO FINALE MODELLI:")
print(results_df.to_string(index=False))

# table to compare times (time predicted by the best model VS real teoric time in datas )
best_model_name = min(results, key=lambda x: x["MAPE%"])["Model"]
best_model = trained_models[best_model_name]

y_pred_best = best_model.predict(X_test)
nome_best = best_model_name

confronto_test = confronto_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

tabella_confronto = pd.DataFrame({
    "Tempo_Effettivo_ORE": y_test_reset,
    "Tempo_Teorico_AS400_ORE": confronto_test["Tempo_Teorico_TOT_ORE"].values,
    "Tempo_Predetto_ML_ORE": y_pred_best,
})

# absolute errors 
tabella_confronto["Errore_AS400_ORE"] = (
    tabella_confronto["Tempo_Effettivo_ORE"] - tabella_confronto["Tempo_Teorico_AS400_ORE"]
).abs()

tabella_confronto["Errore_ML_ORE"] = (
    tabella_confronto["Tempo_Effettivo_ORE"] - tabella_confronto["Tempo_Predetto_ML_ORE"]
).abs()

tabella_confronto["ML_migliora"] = (
    tabella_confronto["Errore_ML_ORE"] < tabella_confronto["Errore_AS400_ORE"]
)

print(f"\n\nTABELLA DI CONFRONTO TEMPI (modello: {nome_best}):")
print(tabella_confronto.round(4).to_string(index=False))

# how many times does the model perform better then real datas
n_test = len(tabella_confronto)
n_migliora = tabella_confronto["ML_migliora"].sum()
pct_migliora = n_migliora / n_test * 100

rmse_as400 = np.sqrt(np.mean(tabella_confronto["Errore_AS400_ORE"] ** 2))
rmse_ml = np.sqrt(np.mean(tabella_confronto["Errore_ML_ORE"] ** 2))

mape_as400_val = mape(tabella_confronto["Tempo_Effettivo_ORE"].values, tabella_confronto["Tempo_Teorico_AS400_ORE"].values)
mape_ml_val = mape(tabella_confronto["Tempo_Effettivo_ORE"].values, tabella_confronto["Tempo_Predetto_ML_ORE"].values)

riduzione_rmse = (rmse_as400 - rmse_ml) / rmse_as400 * 100
riduzione_mape = (mape_as400_val - mape_ml_val) / mape_as400_val * 100

print(f"\n\nSTATISTICHE DI CONFRONTO - {nome_best} vs Tempo Teorico AS400:")
print(f"  Campioni nel test set:                    {n_test}")
print(f"  Casi in cui ML è più preciso di AS400:    {n_migliora} / {n_test}  ({pct_migliora:.1f}%)")
print(f"")
print(f"  RMSE Tempo Teorico AS400:                 {rmse_as400:.4f} ore")
print(f"  RMSE Modello ML:                          {rmse_ml:.4f} ore")
print(f"  Riduzione RMSE:                           {riduzione_rmse:.1f}%")
print(f"")
print(f"  MAPE Tempo Teorico AS400:                 {mape_as400_val:.2f}%")
print(f"  MAPE Modello ML:                          {mape_ml_val:.2f}%")
print(f"  Riduzione MAPE:                           {riduzione_mape:.1f}%")

# graph comparing teoric time predicted vs real teoric time
fig, ax = plt.subplots(figsize=(10, 5))
idx = range(n_test)

ax.scatter(idx, tabella_confronto["Tempo_Effettivo_ORE"].values, label="Effettivo", color="orangered", s=30, alpha=0.7, zorder=3)
ax.scatter(idx, tabella_confronto["Tempo_Teorico_AS400_ORE"].values, label="Teorico AS400", color="lime", s=30, alpha=0.7)
ax.scatter(idx, tabella_confronto["Tempo_Predetto_ML_ORE"].values, label=f"Predetto ML ({nome_best})", color="royalblue", s=30, alpha=0.7)

ax.grid(True, axis='both', which='major', color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)
ax.grid(True, axis='x', which='minor', color='lightgray', linestyle=':', linewidth=0.3, alpha=0.5)

ax.tick_params(axis='y', labelsize=10)
ax.tick_params(axis='x', labelsize=5.5)

ax.set_xticks(idx[::1]) 
ax.set_xticks(idx, minor=True)  

ax.set_title("Confronto: Tempo Effettivo vs Teorico AS400 vs Predetto ML", fontsize=13, fontweight="bold")
ax.set_xlabel("Campioni (test set, ordinati per indice)")
ax.set_ylabel("Tempo (ore)")
ax.legend()
plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PLOT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PLOT_PATH, dpi=150)
plt.show()
print(f"\nGrafico salvato in {OUTPUT_PLOT_PATH}")

# saves model
best_mape = min(r["MAPE%"] for r in results)

print(f"\nModello migliore: {best_model_name}  (MAPE: {best_mape:.2f}%)")

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
cv_mape_neg = cross_val_score(best_model, X, y, cv=cv_esterna, scoring="neg_mean_absolute_percentage_error", n_jobs=-1)

cv_mae = -cv_mae_neg
cv_rmse = np.sqrt(-cv_mse_neg)
cv_mape_cv = -cv_mape_neg * 100 

print(f"\nSTIMA ROBUSTA CV ESTERNA ({cv_descrizione}) - media ± std:")
print(f"  MAE:  {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")
print(f"  RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
print(f"  MAPE: {cv_mape_cv.mean():.2f}% ± {cv_mape_cv.std():.2f}%")
print(f"  R²:   {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(best_model, MODEL_PATH)
parametri_tempo = {
    "articoli_top": articoli_top.tolist(),
    "top_n_articoli": TOP_N_ARTICOLI,
}
joblib.dump(parametri_tempo, PARAMS_PATH)

print(f"\nModello salvato in {MODEL_PATH}")
print(f"Parametri salvati in {PARAMS_PATH}")


# ****** APPUNTI ******
# Mean Absolute Percentage Error risponde a: "il modello sbaglia mediamente del X% nel predire il tempo?"
# il MAPE misura dunque l'errore percentuale medio tra i valori previsti e quelli effettivi.

# I tempi teroici calcolati da AS400 sbagliano il 17.35% sul tempo di questa macchina.
# Il miglior modello sbaglia il 30.17% delle volte. Questo è dovuto principalmente dalla mancanza di dati 
# mancanza di dati sufficienti per i modelli per imparere realmente le ragioni delle tempistiche, 
# con solo 387 righe di dati i modelli non sono in grado di produrre dei risultati sufficientemente
# performanti. inoltre c'è da dire che al momento i tempi teorici della KOEPFER 160/2 sono già 
# ben calibrati nel sistema AS400 e diventa difficile batterli con ML perché comunque  
# non c'è troppo margine di miglioramento.
 
 
 
 
# Analisi delle correlazioni tra le variabili

# Prima dell’addestramento dei modelli è stata eseguita un’analisi di correlazione tra le feature 
# del dataset, considerando come significative le correlazioni con valore assoluto superiore a 0.7.
# Dall’analisi emergono alcune relazioni molto forti tra variabili. In particolare:
 
# “Pezzi da Avanzare” e “Tot pezzi Conteggiati” risultano perfettamente correlati (coefficiente pari 
# a 1.000), indicando che una delle due variabili potrebbe essere ridondante.
# “mese” e “settimana_anno” presentano una correlazione estremamente elevata (0.996), suggerendo che 
# entrambe rappresentano in parte la stessa informazione temporale.
# Le variabili rolling_mean_10 e rolling_std_10 mostrano una correlazione elevata (0.888), essendo 
# entrambe derivate da una finestra temporale di aggregazione.
# “Tot pezzi Conteggiati” e “Qta totale su AS/400” presentano una correlazione significativa (0.791), 
# indicando che le quantità registrate dal sistema gestionale risultano coerenti con quelle effettivamente conteggiate.
# La variabile “Scarti lavorazione” è fortemente correlata con ratio_scarti (0.703), poiché quest’ultima 
# deriva direttamente dalla prima.
 
# Queste correlazioni indicano la presenza di alcune variabili potenzialmente ridondanti; tuttavia, 
# i modelli basati su alberi (come Random Forest e XGBoost) sono generalmente robusti rispetto alla 
# multicollinearità, motivo per cui si è scelto di mantenere tutte le feature nella fase iniziale di 
# sperimentazione.

# Confronto tra i modelli di Machine Learning
# Sono stati testati diversi algoritmi di regressione per la previsione del tempo effettivo di 
# lavorazione, tra cui:
 
# Linear Regression
# Ridge
# Lasso
# Decision Tree
# Random Forest
# XGBoost
 
# Le prestazioni sono state valutate utilizzando quattro metriche principali:
# R² (coefficiente di determinazione)
# MAE (Mean Absolute Error)
# RMSE (Root Mean Squared Error)
# MAPE (Mean Absolute Percentage Error)
 
# Modelli lineari
# I modelli lineari tradizionali (Linear Regression e Ridge) hanno mostrato prestazioni molto scarse, 
# con valori di R² negativi (-0.14 e -0.19). Questo indica che i modelli sono meno accurati rispetto 
# a una semplice previsione basata sulla media del target.
# L’errore medio assoluto risulta inoltre elevato (circa 5 ore), mentre il MAPE supera il 150%, 
# evidenziando una scarsa capacità di catturare le relazioni tra le variabili.
# Il modello Lasso mostra invece un miglioramento significativo:
 
# R² = 0.69
# MAE = 3.48 ore
# RMSE = 5.47 ore
 
# Questo suggerisce che una parte della relazione tra variabili può essere descritta con un modello 
# lineare regolarizzato, ma con prestazioni ancora inferiori rispetto ai modelli non lineari.
# Modelli basati su alberi decisionali
# I modelli basati su alberi hanno mostrato prestazioni decisamente migliori.
 
# Decision Tree
# Il modello Decision Tree raggiunge:
 
# R² = 0.708
# MAE = 2.96 ore
# RMSE = 5.31 ore
 
# Questo risultato evidenzia come la relazione tra le variabili sia fortemente non lineare, 
# caratteristica che i modelli ad albero riescono a catturare meglio rispetto ai modelli lineari.
 
# Random Forest
# Il modello Random Forest migliora ulteriormente le prestazioni:
 
# R² = 0.813
# MAE = 2.33 ore
# RMSE = 4.25 ore
# MAPE = 35.15%
 
# Questo modello sfrutta l’aggregazione di più alberi decisionali, riducendo la varianza e 
# migliorando la capacità di generalizzazione.
 
# XGBoost
# Il modello XGBoost risulta il migliore tra i modelli base:
 
# R² = 0.829
# MAE = 2.07 ore
# RMSE = 4.06 ore
# MAPE = 30.17%
 
# Questo algoritmo di boosting costruisce gli alberi in modo sequenziale, correggendo progressivamente 
# gli errori del modello precedente, permettendo così di catturare relazioni più complesse nel dataset.
 
# Ottimizzazione degli iperparametri
# Per migliorare ulteriormente le prestazioni dei modelli sono state effettuate procedure di Grid Search 
# con validazione incrociata a 5 fold.
 
# Random Forest ottimizzata
# I parametri ottimali individuati sono:
 
# n_estimators = 200
# max_depth = 10
# min_samples_split = 2
# min_samples_leaf = 5
 
# Tuttavia, il modello ottimizzato ha mostrato prestazioni peggiori rispetto alla versione base:
 
# R² = 0.67
# RMSE = 5.63 ore
 
# Questo suggerisce che la configurazione iniziale del modello risultava già 
# ben bilanciata per il dataset utilizzato.
 
# XGBoost ottimizzato
# Per il modello XGBoost i parametri migliori sono risultati:
 
# learning_rate = 0.01
# max_depth = 3
# n_estimators = 400
# subsample = 0.8
 
# Con questa configurazione il modello ha ottenuto le migliori prestazioni complessive:
 
# R² = 0.877
# MAE = 1.79 ore
# RMSE = 3.44 ore
 
# Nonostante il miglioramento, il controllo di overfitting mostra una differenza tra errore su 
# training e test (gap RMSE ≈ 2.34 ore), indicando una leggera tendenza del modello ad adattarsi 
# eccessivamente ai dati di addestramento.
 
# Confronto con i tempi teorici AS400
# Uno degli obiettivi principali dell’analisi era verificare se il modello di Machine Learning 
# potesse fornire stime dei tempi più accurate rispetto ai tempi teorici presenti nel sistema 
# gestionale AS400.
# L’analisi sul test set mostra che:
 
# il modello ML è più preciso in 33 casi su 78
# pari al 42.3% dei casi
 
# Tuttavia, considerando le metriche globali di errore:
 
# Modello	RMSE
# Tempo teorico AS400	2.50 ore
# Modello ML	4.06 ore
 
# Anche il MAPE mostra una situazione analoga:
 
# Modello	MAPE
# Tempo teorico AS400	17.35%
# Modello ML	30.17%
 
# Nel complesso, quindi, il sistema AS400 continua a fornire stime mediamente più accurate 
# rispetto al modello di Machine Learning.
# Validazione robusta con Time Series Cross Validation
# Per verificare la stabilità del modello nel tempo è stata effettuata una TimeSeriesSplit a 5 fold, 
# ottenendo i seguenti risultati medi:
 
# MAE = 2.20 ± 0.45 ore
# RMSE = 4.58 ± 0.45 ore
# MAPE = 49.81% ± 20.11%
# R² = 0.64 ± 0.08
 
# La deviazione standard relativamente elevata del MAPE indica che le prestazioni del modello 
# possono variare significativamente tra diversi intervalli temporali, suggerendo una certa 
# instabilità nelle previsioni.
 
# Considerazioni finali
# L’analisi sperimentale ha evidenziato che:
# i modelli lineari non sono adeguati a descrivere il problema;
# i modelli basati su alberi decisionali offrono prestazioni significativamente migliori;
# XGBoost ottimizzato rappresenta il modello con la migliore capacità predittiva sul dataset.
# Tuttavia, confrontando i risultati con i tempi teorici del sistema AS400, emerge che il modello 
# di Machine Learning non riesce ancora a superare l’accuratezza del sistema esistente.
 
# Questo risultato suggerisce che:
# i tempi teorici AS400 sono già ben calibrati,
# oppure che il dataset disponibile non contiene tutte le variabili che influenzano il tempo reale 
# di lavorazione.