import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay)
from sklearn.metrics import make_scorer, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
warnings.filterwarnings("ignore")
from feature_engineering import pipeline_classificazione

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "koepfer_160_2.csv")
OUTPUT_CM_PATH = os.path.join(PROJECT_ROOT, "outputs", "confusion_matrix_anomaly_BD.png")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "classification", "best_classificazione_anomaly_BD.pkl")
PARAMS_PATH = os.path.join(PROJECT_ROOT, "models", "classification", "parametri_classificazione_anomaly_BD.pkl")

# load datas
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

df = pipeline_classificazione(df)

# class definition (target)
p60 = df['Indice_Inefficienza'].quantile(0.60)
p85 = df['Indice_Inefficienza'].quantile(0.85)

print(f"Soglia ATTENZIONE (60° percentile): {p60:.4f}")
print(f"Soglia ANOMALIA   (85° percentile): {p85:.4f}")

SOGLIA_ATTENZIONE = p60
SOGLIA_ANOMALIA = p85

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
    "ID DAD",
    "Descrizione Macchina",
    "C.d.L. Effett",
    "Data_Ora_Fine",
]

y = df["classe"]
X = df.drop(columns=cols_to_drop, errors="ignore")

# train/test split 
if "Data_Ora_Fine" in df.columns:
    idx_sorted = df.sort_values("Data_Ora_Fine").index
    split_idx = int(len(idx_sorted) * 0.8)
    train_idx = idx_sorted[:split_idx]
    test_idx = idx_sorted[split_idx:]
    X_train, X_test = X.loc[train_idx].copy(), X.loc[test_idx].copy()
    y_train, y_test = y.loc[train_idx].copy(), y.loc[test_idx].copy()
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

# train/validation split per ottimizzare le soglie probabilistiche
if "Data_Ora_Fine" in df.columns and len(X_train) > 5:
    # se i dati sono temporali, manteniamo validazione temporale (niente shuffle)
    split_val_idx = int(len(X_train) * 0.8)
    split_val_idx = min(max(split_val_idx, 1), len(X_train) - 1)
    X_fit, X_val = X_train.iloc[:split_val_idx].copy(), X_train.iloc[split_val_idx:].copy()
    y_fit, y_val = y_train.iloc[:split_val_idx].copy(), y_train.iloc[split_val_idx:].copy()
else:
    if y_train.value_counts().min() >= 2 and len(X_train) >= 10:
        skf_train_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fit_idx, val_idx = next(skf_train_val.split(X_train, y_train))
        X_fit, X_val = X_train.iloc[fit_idx].copy(), X_train.iloc[val_idx].copy()
        y_fit, y_val = y_train.iloc[fit_idx].copy(), y_train.iloc[val_idx].copy()
    else:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
    
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

# convert null in -> 'MISSING'
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
            class_weight="balanced",
            random_state=42,
            verbosity=0,
            eval_metric="mlogloss",
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
    print(classification_report(y_true, y_pred, target_names=["NORMALE", "ATTENZIONE", "ANOMALIA"],
                                 zero_division=0))
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            print(f"  ROC-AUC (weighted OvR): {auc:.4f}")
        except Exception:
            pass

def valuta_generalizzazione(name, y_train_true, y_train_pred, y_test_true, y_test_pred, soglia_gap=0.05):
    train_acc = accuracy_score(y_train_true, y_train_pred)
    test_acc = accuracy_score(y_test_true, y_test_pred)
    gap = train_acc - test_acc

    if gap > soglia_gap:
        diagnosi = "OVERFITTING (training accuracy molto più alta della validation/test)"
    else:
        diagnosi = "GENERALIZZAZIONE BUONA (training ≈ validation/test)"

    print(f"\nGeneralizzazione - {name}")
    print(f"  Training Accuracy:   {train_acc:.4f}")
    print(f"  Validation Accuracy: {test_acc:.4f}")
    print(f"  Gap (train-test):    {gap:.4f}")
    print(f"  Diagnosi:            {diagnosi}")

# SOGLIE CUSTOM (commentate - soglia standard 0.5 usata):
# Per riabilitarle: decommentare predici_con_soglie e ottimizza_soglie,
# e sostituire i blocchi "SOGLIA STANDARD 0.5" nelle sezioni RF e XGB.
"""
def predici_con_soglie(proba, soglia_anomalia, soglia_attenzione):
    return np.where(
        proba[:, 2] >= soglia_anomalia,
        2,
        np.where(proba[:, 1] >= soglia_attenzione, 1, 0)
    )

def ottimizza_soglie(proba_val, y_val):
    migliori_soglie = None
    migliore_tuple = None

    soglie_anomalia = np.arange(0.15, 0.61, 0.05)
    soglie_attenzione = np.arange(0.20, 0.71, 0.05)

    for soglia_anomalia in soglie_anomalia:
        for soglia_attenzione in soglie_attenzione:
            if soglia_attenzione < soglia_anomalia:
                continue

            y_pred = predici_con_soglie(proba_val, soglia_anomalia, soglia_attenzione)
            recall_anomalia = recall_score(y_val, y_pred, labels=[2], average="macro", zero_division=0)
            f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
            cm_val = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
            errori_2_to_1 = int(cm_val[2, 1])

            criterio = (recall_anomalia, -errori_2_to_1, f1_macro)

            if migliore_tuple is None or criterio > migliore_tuple:
                migliore_tuple = criterio
                migliori_soglie = {
                    "soglia_anomalia": float(soglia_anomalia),
                    "soglia_attenzione": float(soglia_attenzione),
                    "recall_anomalia_val": float(recall_anomalia),
                    "f1_macro_val": float(f1_macro),
                    "errori_2_to_1_val": errori_2_to_1,
                }

    return migliori_soglie
"""

# training and comparison of base models
results = []
trained_models = {}
predizioni_test = {}
# soglie_modello = {}
print("\nBASE MODELS:")
for name, model in base_models.items():
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)

    y_proba = model.predict_proba(X_test) if hasattr(model.named_steps["model"], "predict_proba") else None

    valuta_modello(name, y_test, y_pred, y_proba)
    valuta_generalizzazione(name, y_train, y_pred_train, y_test, y_pred)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    results.append({"Model": name, "Accuracy": acc, "F1-macro": f1})
    trained_models[name] = model
    predizioni_test[name] = y_pred

# random forest grid search
print("\n\nRANDOM FOREST - GRID SEARCH")

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor_tree),
    ("model", RandomForestClassifier(class_weight="balanced", random_state=42))
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
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring = make_scorer(recall_score, labels=[2], average="macro", zero_division=0), 
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_fit, y_fit)
print("Migliori parametri RF:", rf_grid.best_params_)

best_rf_val = rf_grid.best_estimator_
# SOGLIE CUSTOM (commentate - soglia standard 0.5 usata):
# proba_val = best_rf_val.predict_proba(X_val)
# rf_soglie = ottimizza_soglie(proba_val, y_val)
# print(
#     f"Soglie RF ottimizzate su validation -> anomalia: {rf_soglie['soglia_anomalia']:.2f}, "
#     f"attenzione: {rf_soglie['soglia_attenzione']:.2f}, "
#     f"recall_anomalia: {rf_soglie['recall_anomalia_val']:.4f}, "
#     f"errori 2->1: {rf_soglie['errori_2_to_1_val']}"
# )

best_rf = clone(rf_pipeline).set_params(**rf_grid.best_params_)
best_rf.fit(X_train, y_train)

# SOGLIA STANDARD 0.5: usa predict() direttamente
# soglia_anomalia = rf_soglie["soglia_anomalia"]
# soglia_attenzione = rf_soglie["soglia_attenzione"]
# proba_train = best_rf.predict_proba(X_train)
# y_pred_train_custom = predici_con_soglie(proba_train, soglia_anomalia, soglia_attenzione)
# proba = best_rf.predict_proba(X_test)
# y_pred_custom = predici_con_soglie(proba, soglia_anomalia, soglia_attenzione)
# valuta_modello("Random Forest Ottimizzata (soglie custom)", y_test, y_pred_custom, proba)
# valuta_generalizzazione("Random Forest Ottimizzata (soglie custom)", y_train, y_pred_train_custom, y_test,
#                        y_pred_custom)
proba_train_rf = best_rf.predict_proba(X_train)
y_pred_train_rf = best_rf.predict(X_train)
proba_rf = best_rf.predict_proba(X_test)
y_pred_rf = best_rf.predict(X_test)
valuta_modello("Random Forest Ottimizzata (soglia 0.5)", y_test, y_pred_rf, proba_rf)
valuta_generalizzazione("Random Forest Ottimizzata (soglia 0.5)", y_train, y_pred_train_rf, y_test, y_pred_rf)
results.append({
    "Model": "Random Forest Ottimizzata",
    "Accuracy": accuracy_score(y_test, y_pred_rf),
    # "Accuracy": accuracy_score(y_test, y_pred_custom),
    # "F1-macro": f1_score(y_test, y_pred_custom, average="macro", zero_division=0)
    "F1-macro": f1_score(y_test, y_pred_rf, average="macro", zero_division=0)
})
trained_models["Random Forest Ottimizzata"] = best_rf
# predizioni_test["Random Forest Ottimizzata"] = y_pred_custom
# soglie_modello["Random Forest Ottimizzata"] = rf_soglie
predizioni_test["Random Forest Ottimizzata"] = y_pred_rf
# soglie_modello["Random Forest Ottimizzata"] = rf_soglie

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
    "model__n_estimators": [200, 400],
    "model__max_depth": [3, 6, 10],
    "model__learning_rate": [0.01, 0.1],
    "model__subsample": [0.8, 1.0]
}

xgb_grid = GridSearchCV(
    xgb_pipeline,
    xgb_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring = make_scorer(recall_score, labels=[2], average="macro", zero_division=0),
    n_jobs=-1,
    verbose=1
)
xgb_grid.fit(X_fit, y_fit)
print("Migliori parametri XGB:", xgb_grid.best_params_)

best_xgb_val = xgb_grid.best_estimator_
# SOGLIE CUSTOM (commentate - soglia standard 0.5 usata):
# proba_val = best_xgb_val.predict_proba(X_val)
# xgb_soglie = ottimizza_soglie(proba_val, y_val)
# print(
#     f"Soglie XGB ottimizzate su validation -> anomalia: {xgb_soglie['soglia_anomalia']:.2f}, "
#     f"attenzione: {xgb_soglie['soglia_attenzione']:.2f}, "
#     f"recall_anomalia: {xgb_soglie['recall_anomalia_val']:.4f}, "
#     f"errori 2->1: {xgb_soglie['errori_2_to_1_val']}"
# )

best_xgb = clone(xgb_pipeline).set_params(**xgb_grid.best_params_)
best_xgb.fit(X_train, y_train)

# SOGLIA STANDARD 0.5: usa predict() direttamente
# soglia_anomalia = xgb_soglie["soglia_anomalia"]
# soglia_attenzione = xgb_soglie["soglia_attenzione"]
# proba_train = best_xgb.predict_proba(X_train)
# y_pred_train_custom = predici_con_soglie(proba_train, soglia_anomalia, soglia_attenzione)
# proba = best_xgb.predict_proba(X_test)
# y_pred_custom = predici_con_soglie(proba, soglia_anomalia, soglia_attenzione)
# valuta_modello("XGBoost Ottimizzata (soglie custom)", y_test, y_pred_custom, proba)
# valuta_generalizzazione("XGBoost Ottimizzata (soglie custom)", y_train, y_pred_train_custom, y_test,
#                        y_pred_custom)
proba_train_xgb = best_xgb.predict_proba(X_train)
y_pred_train_xgb = best_xgb.predict(X_train)
proba_xgb = best_xgb.predict_proba(X_test)
y_pred_xgb = best_xgb.predict(X_test)
valuta_modello("XGBoost Ottimizzata (soglia 0.5)", y_test, y_pred_xgb, proba_xgb)
valuta_generalizzazione("XGBoost Ottimizzata (soglia 0.5)", y_train, y_pred_train_xgb, y_test, y_pred_xgb)

results.append({
    "Model": "XGBoost Ottimizzata",
    # "Accuracy": accuracy_score(y_test, y_pred_custom),
    # "F1-macro": f1_score(y_test, y_pred_custom, average="macro", zero_division=0)
    "Accuracy": accuracy_score(y_test, y_pred_xgb),
    "F1-macro": f1_score(y_test, y_pred_xgb, average="macro", zero_division=0)
})
trained_models["XGBoost Ottimizzata"] = best_xgb
# predizioni_test["XGBoost Ottimizzata"] = y_pred_custom
# soglie_modello["XGBoost Ottimizzata"] = xgb_soglie
predizioni_test["XGBoost Ottimizzata"] = y_pred_xgb
# soglie_modello["XGBoost Ottimizzata"] = xgb_soglie

# final comparison 
results_df = pd.DataFrame(results).sort_values("F1-macro", ascending=False)
print("\n\nCONFRONTO FINALE MODELLI:")
print(results_df.to_string(index=False))
best_model_name = max(results, key=lambda x: x["F1-macro"])["Model"]
best_model = trained_models[best_model_name]
best_f1 = max(r["F1-macro"] for r in results)
best_y_pred = predizioni_test[best_model_name]
recall_anomalia = recall_score(y_test, best_y_pred, labels=[2], average="macro", zero_division=0)
print(f"  Recall ANOMALIA: {recall_anomalia:.4f}")
print(f"\nModello migliore: {best_model_name}  (F1-macro: {best_f1:.4f})")

# cross-validation esterna
if "Data_Ora_Fine" in df.columns and len(X) >= 10:
    cv_esterna = TimeSeriesSplit(n_splits=5)
    cv_descrizione = "TimeSeriesSplit 5-fold"
else:
    cv_esterna = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_descrizione = "StratifiedKFold 5-fold"

scorer_recall_anomalia = make_scorer(recall_score, labels=[2], average="macro", zero_division=0)

cv_acc = cross_val_score(best_model, X, y, cv=cv_esterna, scoring="accuracy", n_jobs=-1)
cv_f1_macro = cross_val_score(best_model, X, y, cv=cv_esterna, scoring="f1_macro", n_jobs=-1)
cv_recall_anomalia = cross_val_score(best_model, X, y, cv=cv_esterna, scoring=scorer_recall_anomalia, n_jobs=-1)
cv_roc_auc = cross_val_score(best_model, X, y, cv=cv_esterna, scoring="roc_auc_ovr_weighted", n_jobs=-1)

print(f"\nSTIMA ROBUSTA CV ESTERNA ({cv_descrizione}) - media ± std:")
print(f"  Accuracy:        {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"  F1-macro:        {cv_f1_macro.mean():.4f} ± {cv_f1_macro.std():.4f}")
print(f"  Recall ANOMALIA: {cv_recall_anomalia.mean():.4f} ± {cv_recall_anomalia.std():.4f}")
print(f"  ROC-AUC OvR w.:  {cv_roc_auc.mean():.4f} ± {cv_roc_auc.std():.4f}")

# CODE TO USE IN CASE YOU WANT TO USE CUSTOM THRESHOLDS:
"""
def cv_con_soglia_custom(model, X, y, cv, soglia_anomalia, soglia_attenzione):
    metriche = {"accuracy": [], "recall_anomalia": [], "f1_macro": [], "roc_auc": []}
    
    for train_idx, test_idx in cv.split(X, y):
        X_cv_train = X.iloc[train_idx]
        y_cv_train = y.iloc[train_idx]
        X_cv_test  = X.iloc[test_idx]
        y_cv_test  = y.iloc[test_idx]
        
        m = clone(model)
        m.fit(X_cv_train, y_cv_train)
        proba = m.predict_proba(X_cv_test)
        y_pred = predici_con_soglie(proba, soglia_anomalia, soglia_attenzione)
        
        metriche["accuracy"].append(accuracy_score(y_cv_test, y_pred))
        metriche["recall_anomalia"].append(recall_score(y_cv_test, y_pred, labels=[2], average="macro", zero_division=0))
        metriche["f1_macro"].append(f1_score(y_cv_test, y_pred, average="macro", zero_division=0))
        try:
            metriche["roc_auc"].append(roc_auc_score(y_cv_test, proba, multi_class="ovr", average="weighted"))
        except:
            pass
    
    return {k: np.array(v) for k, v in metriche.items()}

best_soglie_cv = soglie_modello.get(best_model_name, {})
soglia_an  = best_soglie_cv.get("soglia_anomalia", 0.5)  if best_soglie_cv else 0.5
soglia_att = best_soglie_cv.get("soglia_attenzione", 0.5) if best_soglie_cv else 0.5

metriche_cv = cv_con_soglia_custom(best_model, X, y, cv_esterna, soglia_an, soglia_att)

print(f"\nSTIMA ROBUSTA CV ESTERNA ({cv_descrizione}) - soglie custom ({soglia_an:.2f}/{soglia_att:.2f}) - media ± std:")
print(f"  Accuracy:        {metriche_cv['accuracy'].mean():.4f} ± {metriche_cv['accuracy'].std():.4f}")
print(f"  Recall ANOMALIA: {metriche_cv['recall_anomalia'].mean():.4f} ± {metriche_cv['recall_anomalia'].std():.4f}")
print(f"  F1-macro:        {metriche_cv['f1_macro'].mean():.4f} ± {metriche_cv['f1_macro'].std():.4f}")
print(f"  ROC-AUC:         {metriche_cv['roc_auc'].mean():.4f} ± {metriche_cv['roc_auc'].std():.4f}")
"""

# confusion matrix
print(f"\nMatrice di confusione - {best_model_name}:")
cm = confusion_matrix(y_test, best_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NORMALE", "ATTENZIONE", "ANOMALIA"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_CM_PATH), exist_ok=True)
plt.savefig(OUTPUT_CM_PATH, dpi=150)
plt.show()
print(f"Matrice salvata in {OUTPUT_CM_PATH}")

#save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(best_model, MODEL_PATH)

# best_soglie = soglie_modello.get(best_model_name)

parametri_anomaly = {
    "articoli_top": articoli_top.tolist(),
    "top_n_articoli": TOP_N_ARTICOLI,
    "soglia_attenzione": SOGLIA_ATTENZIONE,
    "soglia_anomalia": SOGLIA_ANOMALIA,
    "tipo_soglie": "percentili_60_85",
    "soglia_proba_anomalia": best_soglie["soglia_anomalia"] if best_soglie else None,
    "soglia_proba_attenzione": best_soglie["soglia_attenzione"] if best_soglie else None,
    "soglie_ottimizzate_su_validation": bool(best_soglie),
}
joblib.dump(parametri_anomaly, PARAMS_PATH)

print(f"Modello salvato in {MODEL_PATH}")
print(f"Parametri salvati in {PARAMS_PATH}")

