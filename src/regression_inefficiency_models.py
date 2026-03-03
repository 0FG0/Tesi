import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from feature_engineering import pipeline_inefficienza

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "koepfer_160_2.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "regression", "best_regressione_inefficienza.pkl")
PARAMS_PATH = os.path.join(PROJECT_ROOT, "models", "regression", "parametri_prepocessing_regressione_inefficienza.pkl")

# loading the clean datas of KOEPFER 160/2 machine
df = pd.read_csv(DATA_PATH)

# frequency encoding 
counts = df['ARTICOLO'].value_counts()
threshold = 3 
df['ARTICOLO_grouped'] = df['ARTICOLO'].where(
    df['ARTICOLO'].isin(counts[counts >= threshold].index), 
    other='ALTRO'
)

freq_map = df['ARTICOLO_grouped'].value_counts(normalize=True)
df['ARTICOLO_freq'] = df['ARTICOLO_grouped'].map(freq_map)

# applying feature engineering
df = pipeline_inefficienza(df)

cols_to_drop = [
    "Indice_Inefficienza",          
    "Tempo Lavoraz. ORE",           
    "Tempo_Teorico_TOT_ORE",        
    "WO",                           
    "ARTICOLO",                     
    "Descrizione Articolo",
    'ARTICOLO_grouped',         
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# columns identification
categorical_cols = [
    col for col in [
        "FASE",
        "Cod CIC",
        "C.d.L. Prev",
        "Descrizione Centro di Lavoro previsto"
    ]
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
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"R²   : {r2:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")

    return {
        "Model": name,
        "R2": r2,
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
    "model__min_samples_leaf": [1, 3]
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

print("Miglior parametri RF:", rf_grid.best_params_)

best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)
result_rf = valuta_modello_regressione("Random Forest Ottimizzata", y_test, y_pred_rf)
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
    cv=KFold(n_splits=5, shuffle=True, random_state=42),  
    scoring="r2",
    n_jobs=-1,
    verbose=1
)
xgb_grid.fit(X_train, y_train)

print("Migliori parametri XGB:", xgb_grid.best_params_)

best_xgb = xgb_grid.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
result_xgb = valuta_modello_regressione("XGBoost Ottimizzata", y_test, y_pred_xgb)
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
best_rmse = min(r["RMSE"] for r in results)
print(f"\nModello migliore: {best_model_name}  (RMSE: {best_rmse:.4f})")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(best_model, MODEL_PATH)

parametri = {
    "freq_map":           freq_map,
    "articoli_frequenti": counts[counts >= threshold].index.tolist(),
    "threshold":          threshold,
    "modello_scelto":     best_model_name,
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
# Se il valore reale della variabile che voglio predire è 1.2 e il modello prevede 1.5 -> errore = 0.3 -> MSE = 0.03
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

# I risultati mostrano una netta differenza tra modelli lineari e modelli basati su alberi decisionali,
# questi ultimi infatti superano significativamente i modelli lineari.
# suggerendo la presenza di relazioni non lineari tra le variabili operative e l'indice di inefficienza.

# Modelli Lineari:
# Linear Regression → R² ≈ -0.20
# Ridge → R² ≈ -0.18
# Lasso → R² ≈ -0.05

# Tutti questi modelli hanno avuto risultati peggiori della media.
# Questo indica che la relazione tra variabili operative e inefficienza non è lineare, 
# sono presenti interazioni complesse tra le feature che i modelli lineari non sono in grado di cogliere. 
# Inoltre come ovviamente ci si aspettava dai dati, esiste multicollinearità (come evidenziato dall’analisi
# delle correlazioni). segno che i modelli lineari non sono in grado di studiare il peso delle variabili fortemente correlate.
 
# Modelli ad Albero:
# Decision Tree → R² ≈ 0.23
# Random Forest → R² ≈ 0.705
# Random Forest Ottimizzata → R² ≈ 0.715
# XGBoost → R² ≈ 0.783
# XGBoost Ottimizzata → R² ≈ 0.763  

# Questi modelli arrivano a spiegare circa il 78% della variabilità. 
# Questo suggerisce che il sistema produttivo presenta dinamiche non lineari e che 
# l’inefficienza dipende da combinazioni di variabili e non da effetti indipendenti.

# XGBoost è il modello che ha ottenuto il miglior risultato con:
# R² ≈ 0.783
# Dopo l'ottimizzazione si note che random forest è migliorata leggermente mentre XGBoost è peggiorato 
# passando da 0.783 a 0.763, questo significa che il dataset è già ben modellato con parametri standard, 
# non c’è forte overfitting nel modello base e che la struttura dei dati è relativamente stabile.
# il fatto che xgboost spieghi circa il 78% dell'inefficienza significa che l’Indice di Inefficienza 
# non è casuale ma è fortemente legato alle variabili operative e che quindi è possibile costruire un 
# sistema predittivo affidabile.

# A differenza dell'MSE, che eleva al quadrato gli errori rendendone difficile l’interpretazione pratica, 
# l’RMSE fornisce una misura direttamente interpretabile: indica di quanto, in media, il modello si discosta 
# dal valore reale dell’indice. Nel caso del modello XGBoost, l’RMSE ≈ 0.121 implica che l’errore medio di 
# previsione è pari a circa 0.12 punti di inefficienza. Considerando che l’Indice di Inefficienza ha media ≈ 1.18 
# e deviazione standard ≈ 0.28, un errore medio di 0.12 rappresenta una deviazione contenuta rispetto alla 
# variabilità naturale del processo. Questo conferma la buona capacità predittiva del modello.