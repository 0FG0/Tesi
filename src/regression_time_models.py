import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
import os
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")
from feature_engineering import pipeline_tempo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "koepfer_160_2.csv")
OUTPUT_PLOT_PATH = os.path.join(PROJECT_ROOT, "outputs", "times_compared_scatter.png")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "regression", "best_regressione_time.pkl")
PARAMS_PATH = os.path.join(PROJECT_ROOT, "models", "regression", "parametri_preprocessing_tempo.pkl")

# load data 
df = pd.read_csv(DATA_PATH)

#frequency encoding
counts = df['ARTICOLO'].value_counts()
threshold = 3
df['ARTICOLO_grouped'] = df['ARTICOLO'].where(
    df['ARTICOLO'].isin(counts[counts >= threshold].index),
    other='ALTRO'
)
freq_map = df['ARTICOLO_grouped'].value_counts(normalize=True)
df['ARTICOLO_freq'] = df['ARTICOLO_grouped'].map(freq_map)

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
    "ARTICOLO_grouped",
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# save new time datas to compare them with real datas
confronto_test = confronto_cols.loc[X_test.index].copy()

# columns identification
categorical_cols = [
    col for col in ["FASE", "Cod CIC", "C.d.L. Prev", "Descrizione Centro di Lavoro previsto"]
    if col in X.columns
]
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

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
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape_val = mape(y_test.values, y_pred)

    results.append({"Model": name, "R2": r2, "RMSE": rmse, "MAPE%": mape_val})
    trained_models[name] = model

    print(f"\n{name}")
    print(f"  R²:    {r2:.4f}")
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
    "model__min_samples_leaf":  [1, 3]
}
rf_grid = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
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

print(f"\nRandom Forest Ottimizzata  ->  R²: {r2_rf:.4f}  |  RMSE: {rmse_rf:.4f} ore  |  MAPE: {mape_rf:.2f}%")
results.append({"Model": "Random Forest Ottimizzata", "R2": r2_rf, "RMSE": rmse_rf, "MAPE%": mape_rf})

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
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
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

print(f"\nXGBoost Ottimizzata  ->  R²: {r2_xgb:.4f}  |  RMSE: {rmse_xgb:.4f} ore  |  MAPE: {mape_xgb:.2f}%")
results.append({"Model": "XGBoost Ottimizzata", "R2": r2_xgb, "RMSE": rmse_xgb, "MAPE%": mape_xgb})

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
    "Tempo_Effettivo_ORE":     y_test_reset,
    "Tempo_Teorico_AS400_ORE": confronto_test["Tempo_Teorico_TOT_ORE"].values,
    "Tempo_Predetto_ML_ORE":   y_pred_best,
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

mape_as400_val = np.mean(
    tabella_confronto["Errore_AS400_ORE"] / tabella_confronto["Tempo_Effettivo_ORE"].replace(0, pd.NA)
) * 100

mape_ml_val = np.mean(
    tabella_confronto["Errore_ML_ORE"] / tabella_confronto["Tempo_Effettivo_ORE"].replace(0, pd.NA)
) * 100

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

ax.scatter(idx, tabella_confronto["Tempo_Effettivo_ORE"].values,
           label="Effettivo", color="orangered", s=30, alpha=0.7, zorder=3)
ax.scatter(idx, tabella_confronto["Tempo_Teorico_AS400_ORE"].values,
           label="Teorico AS400", color="lime", s=30, alpha=0.7)
ax.scatter(idx, tabella_confronto["Tempo_Predetto_ML_ORE"].values,
           label=f"Predetto ML ({nome_best})", color="royalblue", s=30, alpha=0.7)

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

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(best_model, MODEL_PATH)
parametri_tempo = {
    "freq_map":           freq_map,
    "articoli_frequenti": counts[counts >= threshold].index.tolist(),
    "threshold":          threshold,
}
joblib.dump(parametri_tempo, PARAMS_PATH)

print(f"\nModello salvato in {MODEL_PATH}")
print(f"Parametri salvati in {PARAMS_PATH}")


# ****** APPUNTI ******
# Mean Absolute Percentage Error risponde a: "il modello sbaglia mediamente del X% nel predire il tempo?"
# il MAPE misura dunque l'errore percentuale medio tra i valori previsti e quelli effettivi.

# Il modello non sta performando bene sul task temporale:
# Solo 23/78 casi (29.5%) il ML batte AS400
# RMSE ML peggiore di AS400 (-7.2%)
# MAPE ML 29% contro 13% di AS400

# I tempi teroici calcolati da AS400 sbagliano solo il 13% sul tempo di questa macchina.
# Il miglior modello sbaglia il 29%. Questo è dovuto principalmente dalla mancanza di dati sufficienti per i modelli
# per imparere realmente le ragioni delle tempistiche, con solo 387 righe di dati i modelli non sono in grado di 
# produrre dei risultati sufficientemente performanti. inoltre c'è da dire che al momento 
# i tempi teorici della KOEPFER 160/2 sono già molto ben calibrati nel sistema AS400 
# e diventa comunque difficile batterli con ML perché non c'è molto margine di miglioramento.