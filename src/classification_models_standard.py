import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import warnings
import joblib
warnings.filterwarnings("ignore")
from feature_engineering import pipeline_classificazione

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "koepfer_160_2.csv")
OUTPUT_CM_PATH = os.path.join(PROJECT_ROOT, "outputs", "confusion_matrix_rf_standard.png")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "classification", "best_classificazione_standard.pkl")
PARAMS_PATH = os.path.join(PROJECT_ROOT, "models", "classification", "parametri_classificazione_standard.pkl")

# load datas
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

df = pipeline_classificazione(df)

# class definition (target)
media = df['Indice_Inefficienza'].mean()
std = df['Indice_Inefficienza'].std()
SOGLIA_ATTENZIONE = media
SOGLIA_ANOMALIA = media + std 

def classifica_inefficienza(valore):
    if valore <= SOGLIA_ATTENZIONE:
        return 0   
    elif valore <= SOGLIA_ANOMALIA:
        return 1   
    else:
        return 2   

df["classe"] = df["Indice_Inefficienza"].apply(classifica_inefficienza)

print("DISTRIBUZIONE CLASSI:")
classe_counts = df["classe"].value_counts().sort_index()
labels = {0: "NORMALE", 1: "ATTENZIONE", 2: "ANOMALIA"}
for cls, cnt in classe_counts.items():
    pct = cnt / len(df) * 100
    print(f"  Classe {cls} ({labels[cls]}): {cnt} campioni ({pct:.1f}%)")

# warning if classes are unbalanced
min_pct = (classe_counts / len(df) * 100).min()
if min_pct < 10:
    print(f"\nClasse minoritaria sotto il 10% ({min_pct:.1f}%).")
    print("Considera class_weight='balanced' nei modelli.")

# X and y def
cols_to_drop = [
    "classe",
    "Indice_Inefficienza",
    "Tempo Lavoraz. ORE",
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

y = df["classe"]
X = df.drop(columns=cols_to_drop, errors="ignore")

# train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  
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
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor_linear),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ]),
    "Decision Tree": Pipeline([
        ("preprocessor", preprocessor_tree),
        ("model", DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ))
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor_tree),
        ("model", RandomForestClassifier(
            class_weight="balanced",
            random_state=42
        ))
    ]),
    "XGBoost": Pipeline([
        ("preprocessor", preprocessor_tree),
        ("model", XGBClassifier(
            random_state=42,
            verbosity=0,
            eval_metric="mlogloss",
            use_label_encoder=False
        ))
    ]),
    "SVM": Pipeline([
        ("preprocessor", preprocessor_linear),
        ("model", SVC(
            class_weight="balanced",
            kernel="rbf",
            probability=True, 
            random_state=42
        ))
    ])
}

def valuta_modello(name, y_true, y_pred, y_proba=None):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred,
                                 target_names=["NORMALE", "ATTENZIONE", "ANOMALIA"],
                                 zero_division=0))
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            print(f"  ROC-AUC (weighted OvR): {auc:.4f}")
        except Exception:
            pass

# training and comparison of base models
results = []
trained_models = {}
print("\nBASE MODELS:")
for name, model in base_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_proba = model.predict_proba(X_test) if hasattr(model.named_steps["model"], "predict_proba") else None

    valuta_modello(name, y_test, y_pred, y_proba)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    results.append({"Model": name, "Accuracy": acc, "F1-macro": f1})
    trained_models[name] = model

# random forest grid search
print("\n\nRANDOM FOREST - GRID SEARCH")

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor_tree),
    ("model", RandomForestClassifier(class_weight="balanced", random_state=42))
])

rf_param_grid = {
    "model__n_estimators":    [200, 400],
    "model__max_depth":       [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf":  [1, 3]
}

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1_macro",  
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train, y_train)
print("Migliori parametri RF:", rf_grid.best_params_)

best_rf = rf_grid.best_estimator_
y_pred_rf    = best_rf.predict(X_test)
y_proba_rf   = best_rf.predict_proba(X_test)
valuta_modello("Random Forest Ottimizzata", y_test, y_pred_rf, y_proba_rf)

results.append({
    "Model":      "Random Forest Ottimizzata",
    "Accuracy":   accuracy_score(y_test, y_pred_rf),
    "F1-macro":   f1_score(y_test, y_pred_rf, average="macro", zero_division=0)
})
trained_models["Random Forest Ottimizzata"] = best_rf

# xgboost grid search
print("\n\nXGBOOST - GRID SEARCH")

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor_tree),
    ("model", XGBClassifier(
        random_state=42,
        verbosity=0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        use_label_encoder=False
    ))
])

xgb_param_grid = {
    "model__n_estimators":  [200, 400],
    "model__max_depth":     [3, 6, 10],
    "model__learning_rate": [0.01, 0.1],
    "model__subsample":     [0.8, 1.0]
}

xgb_grid = GridSearchCV(
    xgb_pipeline,
    xgb_param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1
)
xgb_grid.fit(X_train, y_train)
print("Migliori parametri XGB:", xgb_grid.best_params_)

best_xgb     = xgb_grid.best_estimator_
y_pred_xgb   = best_xgb.predict(X_test)
y_proba_xgb  = best_xgb.predict_proba(X_test)
valuta_modello("XGBoost Ottimizzata", y_test, y_pred_xgb, y_proba_xgb)

results.append({
    "Model":    "XGBoost Ottimizzata",
    "Accuracy": accuracy_score(y_test, y_pred_xgb),
    "F1-macro": f1_score(y_test, y_pred_xgb, average="macro", zero_division=0)
})
trained_models["XGBoost Ottimizzata"] = best_xgb

# final comparison 
results_df = pd.DataFrame(results).sort_values("F1-macro", ascending=False)
print("\n\nCONFRONTO FINALE MODELLI:")
print(results_df.to_string(index=False))

# confusion matrix
print("\nMatrice di confusione - Random Forest Ottimizzata:")
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["NORMALE", "ATTENZIONE", "ANOMALIA"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix - Random Forest Ottimizzata")
plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_CM_PATH), exist_ok=True)
plt.savefig(OUTPUT_CM_PATH, dpi=150)
plt.show()
print(f"  Matrice salvata in {OUTPUT_CM_PATH}")


# save model
best_model_name = max(results, key=lambda x: x["F1-macro"])["Model"]
best_model      = trained_models[best_model_name]
best_f1         = max(r["F1-macro"] for r in results)

print(f"\nModello migliore: {best_model_name}  (F1-macro: {best_f1:.4f})")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(best_model, MODEL_PATH)

parametri_standard = {
    "freq_map":           freq_map,
    "articoli_frequenti": counts[counts >= threshold].index.tolist(),
    "threshold":          threshold,
    "soglia_attenzione":  SOGLIA_ATTENZIONE,
    "soglia_anomalia":    SOGLIA_ANOMALIA,
    "tipo_soglie":        "mean_std",
    "modello_scelto":     best_model_name,
}
joblib.dump(parametri_standard, PARAMS_PATH)

print(f"Modello salvato in {MODEL_PATH}")
print(f"Parametri salvati in {PARAMS_PATH}")