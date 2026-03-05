import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay)
from sklearn.metrics import make_scorer, recall_score, f1_score, accuracy_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from xgboost import XGBClassifier
from feature_engineering import pipeline_classificazione
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "koepfer_160_2.csv")
OUTPUT_CM_PATH = os.path.join(PROJECT_ROOT, "outputs", "confusion_matrix_anomaly.png")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "classification", "best_classificazione_anomaly.pkl")
PARAMS_PATH = os.path.join(PROJECT_ROOT, "models", "classification", "parametri_classificazione_anomaly.pkl")

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

# class definition
p85 = df['Indice_Inefficienza'].quantile(0.85)

print(f"Soglia ANOMALIA (85° percentile): {p85:.4f}")

SOGLIA_ANOMALIA = p85

def classifica_inefficienza(valore):
    if valore > SOGLIA_ANOMALIA:
        return 1
    return 0

df["classe"] = df["Indice_Inefficienza"].apply(classifica_inefficienza)

print("DISTRIBUZIONE CLASSI:")
classe_counts = df["classe"].value_counts().sort_index()
labels = {0: "NORMALE", 1: "ANOMALIA"}
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

# train/validation split: temporale se disponibile, altrimenti stratificato
# divide il training set in: 
# X_fit -> dati per addestrare il modello
# X_val -> dati per ottimizzare le soglie e fare grid search
# perchè quando si fanno: GridSearchCV, ottimizzazione soglie probabilistiche custom, selezioni iperparametri
# sono operazioni che non devono mai essere fatte sul test set.

if "Data_Ora_Fine" in df.columns and len(X_train) > 5:
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

# peso positivo per XGBoost (classe anomalia = 1)
# conta le osservazioni nel fit test:
# n_neg → numero di campioni classe 0 (normale)
# n_pos → numero di campioni classe 1 (anomalia)
# scale_pos_weight ≈ normali / anomalie ≈ n%
# dice a xgboost: sbagliare un’anomalia è circa n volte più grave che sbagliare un normale.
# spw calcolato su y_fit: coerente con la grid search che gira su X_fit.
# Con split temporale y_fit (dati piu vecchi) puo avere meno anomalie
# di y_train, dando uno spw piu alto che aiuta XGBoost sui dati recenti.
n_neg = int((y_fit == 0).sum())
n_pos = int((y_fit == 1).sum())
spw = round(n_neg / max(n_pos, 1), 2)
print(f"scale_pos_weight (calcolato su y_fit): {spw:.2f}")

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
    if col in X_fit.columns:
        X_fit[col] = X_fit[col].astype('string').fillna('MISSING')
    if col in X_val.columns:
        X_val[col] = X_val[col].astype('string').fillna('MISSING')

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
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=8,
            random_state=42
        ))
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor_tree),
        ("model", RandomForestClassifier(
            class_weight="balanced",
            n_estimators=300,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=42
        ))
    ]),
    "XGBoost": Pipeline([
        ("preprocessor", preprocessor_tree),
        ("model", XGBClassifier(
            random_state=42,
            verbosity=0,
            objective="binary:logistic",
            eval_metric="logloss",
            max_depth=4,
            min_child_weight=5,
            reg_alpha=0.5,
            reg_lambda=3.0,
            gamma=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
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
    print(classification_report(y_true, y_pred, target_names=["NORMALE", "ANOMALIA"],
                                 zero_division=0))
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba[:, 1])
            print(f"  ROC-AUC: {auc:.4f}")
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

# variable threshold 
def predici_con_soglie(proba, soglia_anomalia):
    return (proba[:, 1] >= soglia_anomalia).astype(int)

# CUSTOM THRESHOLDS 
# finds the threshold that maximizes recall of anomalies while keeping precision above a certain floor
# generate a set of thresholds (0.05, 0.91, 0.01) for each one it calculates: 
# Recall, Precision, F1, Falsi negativi
# If the threshold with the best results is less than min_precision = 0.25, use that threshold.
# Otherwise, remove the constraint and maximize recall only.
"""
def ottimizza_soglie(proba_val, y_val):
    return ottimizza_soglie_con_floor(proba_val, y_val, min_precision=0.25)

def ottimizza_soglie_con_floor(proba_val, y_val, min_precision=0.25):
    migliori_soglie = None
    migliore_recall = -1.0

    # generates an array that starts at 0.05 and goes up to 0.90 (inclusive), incrementing by 0.01.
    soglie_anomalia = np.arange(0.05, 0.91, 0.01)

    # test each of these thresholds
    for soglia in soglie_anomalia:
        y_pred = predici_con_soglie(proba_val, soglia)

        if y_pred.sum() == 0:
            continue

        recall = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
        precision = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
        fn = int(confusion_matrix(y_val, y_pred, labels=[0, 1])[1, 0])

        if precision < min_precision:
            continue

        if migliori_soglie is None or recall > migliore_recall or (
            recall == migliore_recall and f1 > migliori_soglie["f1_anomalia_val"]
        ):
            migliore_recall = recall
            migliori_soglie = {
                "soglia_anomalia": float(soglia),
                "recall_anomalia_val": float(recall),
                "precision_anomalia_val": float(precision),
                "f1_anomalia_val": float(f1),
                "fn_anomalia_val": fn,
            }

    if migliori_soglie is None and min_precision > 0.0:
        print("  Nessuna soglia soddisfa il floor di precision, uso recall puro.")
        return ottimizza_soglie_con_floor(proba_val, y_val, min_precision=0.0)

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
    f1  = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    results.append({"Model": name, "Accuracy": acc, "F1-Anomalia": f1, "Recall-Anomalia": recall_score(y_test, y_pred, pos_label=1, zero_division=0)})
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
    "model__max_depth": [4, 6, 8],
    "model__min_samples_split": [10, 20],
    "model__min_samples_leaf": [5, 10],
    "model__max_features": ["sqrt", 0.7]
}

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=make_scorer(recall_score, pos_label=1, zero_division=0),
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
#     f"recall_anomalia: {rf_soglie['recall_anomalia_val']:.4f}, "
#     f"precision_anomalia: {rf_soglie['precision_anomalia_val']:.4f}, "
#     f"FN anomalia: {rf_soglie['fn_anomalia_val']}"
# )

best_rf = clone(rf_pipeline).set_params(**rf_grid.best_params_)
best_rf.fit(X_train, y_train)

# SOGLIA STANDARD 0.5: usa predict() direttamente
# soglia_anomalia = rf_soglie["soglia_anomalia"]
# proba_train = best_rf.predict_proba(X_train)
# y_pred_train_custom = predici_con_soglie(proba_train, soglia_anomalia)
# proba = best_rf.predict_proba(X_test)
# y_pred_custom = predici_con_soglie(proba, soglia_anomalia)
# valuta_modello("Random Forest Ottimizzata (soglie custom)", y_test, y_pred_custom, proba)
# valuta_generalizzazione("Random Forest Ottimizzata (soglie custom)", y_train, y_pred_train_custom, y_test, y_pred_custom)
proba_train_rf = best_rf.predict_proba(X_train)
y_pred_train_rf = best_rf.predict(X_train)
proba_rf = best_rf.predict_proba(X_test)
y_pred_rf = best_rf.predict(X_test)
valuta_modello("Random Forest Ottimizzata (soglia 0.5)", y_test, y_pred_rf, proba_rf)
valuta_generalizzazione("Random Forest Ottimizzata (soglia 0.5)", y_train, y_pred_train_rf, y_test, y_pred_rf)
results.append({
    "Model": "Random Forest Ottimizzata",
    # "Accuracy": accuracy_score(y_test, y_pred_custom),
    # "F1-macro": f1_score(y_test, y_pred_custom, average="macro", zero_division=0)
    "Accuracy": accuracy_score(y_test, y_pred_rf),
    "F1-Anomalia": f1_score(y_test, y_pred_rf, pos_label=1, zero_division=0),
    "Recall-Anomalia": recall_score(y_test, y_pred_rf, pos_label=1, zero_division=0)
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
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
    ))
])

xgb_param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [3, 6, 10],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8],
    "model__colsample_bytree": [0.8],
    "model__min_child_weight": [3, 5],
    "model__reg_alpha": [0.0, 0.5],
    "model__reg_lambda": [1.0, 3.0],
    "model__gamma": [0.0, 1.0]
    ,"model__scale_pos_weight": [spw, round(spw * 1.5, 2), round(spw * 2.0, 2)]
}

xgb_grid = GridSearchCV(
    xgb_pipeline,
    xgb_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=make_scorer(recall_score, pos_label=1, zero_division=0),
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
#     f"recall_anomalia: {xgb_soglie['recall_anomalia_val']:.4f}, "
#     f"precision_anomalia: {xgb_soglie['precision_anomalia_val']:.4f}, "
#     f"FN anomalia: {xgb_soglie['fn_anomalia_val']}"
# )

best_xgb = clone(xgb_pipeline).set_params(**xgb_grid.best_params_)
best_xgb.fit(X_train, y_train)

# SOGLIA STANDARD 0.5: usa predict() direttamente
# soglia_anomalia = xgb_soglie["soglia_anomalia"]
# proba_train = best_xgb.predict_proba(X_train)
# y_pred_train_custom = predici_con_soglie(proba_train, soglia_anomalia)
# proba = best_xgb.predict_proba(X_test)
# y_pred_custom = predici_con_soglie(proba, soglia_anomalia)
# valuta_modello("XGBoost Ottimizzata (soglie custom)", y_test, y_pred_custom, proba)
# valuta_generalizzazione("XGBoost Ottimizzata (soglie custom)", y_train, y_pred_train_custom, y_test, y_pred_custom)
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
    "F1-Anomalia": f1_score(y_test, y_pred_xgb, pos_label=1, zero_division=0),
    "Recall-Anomalia": recall_score(y_test, y_pred_xgb, pos_label=1, zero_division=0)
})
trained_models["XGBoost Ottimizzata"] = best_xgb
# predizioni_test["XGBoost Ottimizzata"] = y_pred_custom
# soglie_modello["XGBoost Ottimizzata"] = xgb_soglie
predizioni_test["XGBoost Ottimizzata"] = y_pred_xgb
# soglie_modello["XGBoost Ottimizzata"] = xgb_soglie

# final comparison 
results_df = pd.DataFrame(results).sort_values("Recall-Anomalia", ascending=False)
print("\n\nCONFRONTO FINALE MODELLI:")
print(results_df.to_string(index=False))
best_model_name = max(results, key=lambda x: x["Recall-Anomalia"])["Model"]
best_model = trained_models[best_model_name]
best_recall = next(r["Recall-Anomalia"] for r in results if r["Model"] == best_model_name)
best_f1     = next(r["F1-Anomalia"]     for r in results if r["Model"] == best_model_name)
best_y_pred = predizioni_test[best_model_name]
recall_anomalia = recall_score(y_test, best_y_pred, pos_label=1, zero_division=0)
print(f"  Recall ANOMALIA: {recall_anomalia:.4f}")
print(f"Modello migliore: {best_model_name}  (Recall: {best_recall:.4f}, F1: {best_f1:.4f})")

# cross-validation esterna 
# ti dice media ± deviazione standard
# quanto il modello mediamente è performante e di quanto può variare la performance
if "Data_Ora_Fine" in df.columns and len(X) >= 10:
    cv_esterna = TimeSeriesSplit(n_splits=5)
    cv_descrizione = "TimeSeriesSplit 5-fold"
else:
    cv_esterna = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_descrizione = "StratifiedKFold 5-fold"

cv_acc = cross_val_score(best_model, X, y, cv=cv_esterna, scoring="accuracy", n_jobs=-1)
cv_roc_auc = cross_val_score(best_model, X, y, cv=cv_esterna, scoring="roc_auc", n_jobs=-1)

print(f"\nSTIMA ROBUSTA CV ESTERNA ({cv_descrizione}) - media ± std:")
print(f"  Accuracy:       {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"  ROC-AUC:        {cv_roc_auc.mean():.4f} ± {cv_roc_auc.std():.4f}")

# CODE TO USE IN CASE YOU WANT TO USE CUSTOM THRESHOLDS:
"""
def cv_con_soglia_custom(model, X, y, cv, soglia):
    metriche = {"accuracy": [], "recall": [], "f1": [], "roc_auc": []}
    
    for train_idx, test_idx in cv.split(X, y):
        X_cv_train = X.iloc[train_idx]
        y_cv_train = y.iloc[train_idx]
        X_cv_test  = X.iloc[test_idx]
        y_cv_test  = y.iloc[test_idx]
        
        m = clone(model)
        m.fit(X_cv_train, y_cv_train)
        proba = m.predict_proba(X_cv_test)
        y_pred = predici_con_soglie(proba, soglia)
        
        metriche["accuracy"].append(accuracy_score(y_cv_test, y_pred))
        metriche["recall"].append(recall_score(y_cv_test, y_pred, pos_label=1, zero_division=0))
        metriche["f1"].append(f1_score(y_cv_test, y_pred, pos_label=1, zero_division=0))
        try:
            metriche["roc_auc"].append(roc_auc_score(y_cv_test, proba[:, 1]))
        except:
            pass
    
    return {k: np.array(v) for k, v in metriche.items()}
    
soglia_best = soglie_modello.get(best_model_name, {}).get("soglia_anomalia", 0.5)

metriche_cv = cv_con_soglia_custom(best_model, X, y, cv_esterna, soglia_best)

print(f"\nSTIMA ROBUSTA CV ESTERNA ({cv_descrizione}) - soglia={soglia_best:.2f} - media ± std:")
print(f"  Accuracy:        {metriche_cv['accuracy'].mean():.4f} ± {metriche_cv['accuracy'].std():.4f}")
print(f"  Recall ANOMALIA: {metriche_cv['recall'].mean():.4f} ± {metriche_cv['recall'].std():.4f}")
print(f"  F1 ANOMALIA:     {metriche_cv['f1'].mean():.4f} ± {metriche_cv['f1'].std():.4f}")
print(f"  ROC-AUC:         {metriche_cv['roc_auc'].mean():.4f} ± {metriche_cv['roc_auc'].std():.4f}")
"""

# confusion matrix
print(f"\nMatrice di confusione - {best_model_name}:")
cm = confusion_matrix(y_test, best_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NORMALE", "ANOMALIA"])
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
    "classificazione": "binaria_normale_vs_anomalia",
    "soglia_anomalia": SOGLIA_ANOMALIA,
    "tipo_soglie": "percentile_85",
    # "soglia_proba_anomalia": best_soglie["soglia_anomalia"] if best_soglie else None,
    # "soglie_ottimizzate_su_validation": bool(best_soglie),
}
joblib.dump(parametri_anomaly, PARAMS_PATH)

print(f"Modello salvato in {MODEL_PATH}")
print(f"Parametri salvati in {PARAMS_PATH}")

# ****** APPUNTI ******

# (i file di regressione non sono adattati ad un dataset di piccole dimensioni)

# i file classification_models_anomaly_oriented e classification_models_anomaly_bigdata hanno lo 
# stesso obbiettivo ovvero predire il livello di inefficienza della macchina a partire dalle 
# feature disponibili. La differenza non è nello scopo, ma nella strategia di modellazione, 
# che cambia in funzione della quantità di dati disponibile. infatti classification_models_anomaly_oriented
# è stato fatto in modo tale che i modelli abbiano delle buone prestazioni nonstante la poca quantità
# di dati su cui sono addestrati.

# in classification_models_anomaly_oriented si è eseguito un Approccio più RIGIDO (Small Data Oriented)
# il dataset su cui sono addestrati i modelli è di 387 righe circa
# Con questa numerosità si ha:
# Alta varianza del modello
# Alto rischio di overfitting
# Alta instabilità delle metriche (1 sola predizione può cambiare recall di ~8%)
# Per questo motivo i modelli sono stati progettati per essere Più regolarizzati, Meno complessi, Più conservativi

# questione importante delle soglie: 
# le soglie sono fondamentali in quanto sono quelle che decidono quando si avrà un anomalia o meno
# Un modello (come XGBoost, Random Forest ecc.) non predice direttamente “anomalia” o “normale”.
# Predice invece una probabilità in percentuale. bisogna dunque decidere quando questa percentuale
# è sufficientemente alta perchè possa essere considerata come un anomalia ovvero la soglia.
# con la soglia a 0.5 significa che quando il modello valuta che la probabilità che si verifichi una
# anomalia è > 50% allora la considera come tale. abbassando la soglia a 0.15 ad esempio significa che
# quando il modello prevede che la macchina ha una probabilità che è > 15% allora la considera come anomalia
# facendo così si trovano più anomalie ma aumentano i falsi allarmi.
# questa scelta deve dipendere dall'azienda. se avere un anomalia è un grande costo significa che sarà più
# adatto impostare una soglia bassa ad esempio con min_precision = 0.25 in questo modo significa che 
# almeno il 25% delle previsioni classificate come anomalia devono essere effettivamente anomalie.
# ciò significa che è accettabile avere un 75% di falsi allarmi.
# ciò va bene quando  per l'azienda è meglio avere falsi allarmi piuttosto di rischiare di perdere un anomalia. 
# per il progetto ho deciso di utilizzare una soglia minima di 0.5 che è la soglia standard
# dunque non utilizzare le soglie custom in quanto è una soglia che ti permette di non avere 
# così tanti falsi allarmi ma allo stesso modo non perdersi tante anomalie.
# stessa cosa è stata fatta anche per anomaly_BD

# DIFFERENZE rispetto ad anomaly_BD:

# 1) RIDUZIONE DEL PROBLEMA: classificazione binaria
# 
# p85 = df['Indice_Inefficienza'].quantile(0.85)
# 
# def classifica_inefficienza(valore):
#     if valore > SOGLIA_ANOMALIA:
#         return 1
#     return 0
# Classi:
# 
# 0 → NORMALE
# 1 → ANOMALIA
# 
# così a differenza di anomaly_db si è eliminata la classe "ATTENZIONE" in quanto # 
# Con pochi dati: ATTENZIONE è una classe intermedia di conseguenza ha pochi campioni ed 
# È statisticamente simile sia a NORMALE che ad ANOMALIA, ciò comporta ad avere del rumore decisionale
# Unificando ATTENZIONE dentro NORMALE si aumenta la numerosità della classe 0, migliorando
# così la separabilità e riducendo l'instabilità del modello in questo modo ci permette 
# di focalizzarsi solo sull'individuazione delle anomalie reali

# 2) MODELLI PIù RIGIDI (regularizzati)
# 
# Esempio XGBoost nel primo file:
# 
# XGBClassifier(
#     max_depth=4,
#     min_child_weight=5,
#     reg_alpha=0.5,
#     reg_lambda=3.0,
#     gamma=1.0,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     scale_pos_weight=spw
# )
# 
# Qui si nota che inseriamo una profondità limitata, Penalizzazioni di L1 e L2 elevate
# gamma > 0 -> crea meno split inutili, subsample < 1 -> riduce la varianza e scale_pos_weight calibrato

# 3) OTTIMIZZAZIONE ORIENTATA AL RECALL delle anomalie

# rf_grid = GridSearchCV(
#     ...
#     scoring=make_scorer(recall_score, pos_label=1)
# )
# 
# Viene massimizzato il recall della classe ANOMALIA
# Inoltre viene ottimizzata la soglia probabilistica: soglie_anomalia = np.arange(0.05, 0.91, 0.01)
# Questo permette di abbassare la soglia rispetto a 0.5, ridurre i falsi negativi e accettare qualche 
# falso positivo in più in qunto è meglio segnalare un’anomalia in più che perderne una reale.

# 4. SCALE_POS_WEIGHT
# spw = round(n_neg / max(n_pos, 1), 2)
# Serve per compensare lo sbilanciamento con pochi dati è fondamentale perché
# la classe anomalia è minoritaria ed il modello tenderebbe a ignorarla

# ANOMALY_BD
# è pensato per dataset con elevata numerosità dove il rischio di overfitting è molto più basso

# 1). CLASSIFICAZIONE A 3 CLASSI

# def classifica_inefficienza(valore):
#     if valore <= SOGLIA_ATTENZIONE:
#         return 0
#     elif valore <= SOGLIA_ANOMALIA:
#         return 1
#     else:
#         return 2
# 
# 0 → NORMALE
# 1 → ATTENZIONE
# 2 → ANOMALIA

# Qui si mantiene la struttura semantica completa del problema. 
# Questo è corretto quando ci sono abbastanza dati per ogni classe ed il modello può apprendere
# confini complessi in quanto la classe intermedia non crea instabilità

# 2) MODELLI MENO VINCOLATI
 
# Esempio XGBoost 
 
# XGBClassifier(
#     objective="multi:softprob",
#     ...
# )
 
# Grid search più ampia:
# "model__max_depth": [3, 6, 10],
# "model__learning_rate": [0.01, 0.1],
# "model__subsample": [0.8, 1.0]

# Qui: max_depth può arrivare a 10, C'è minore regolarizzazione esplicita, il modello 
# fa una ricerca più ampia nello spazio iperparametri
# Questo è sostenibile solo con molti dati.
 
# 3) OTTIMIZZAZIONE SOGLIE DOPPIA 

# predici_con_soglie(proba, soglia_anomalia, soglia_attenzione)
 
# Qui vengono ottimizzate due soglie la soglia probabilità ANOMALIA e la soglia probabilità ATTENZIONE
# Questo rende il modello più flessibile e più fine nella separazione, ciò lo rende 
# più instabile con pochi dati

# 4) METRICA DI OTTIMIZZAZIONE DIVERSA
 
# scoring = make_scorer(recall_score, labels=[2], average="macro")
 
# Si massimizza il recall solo della classe 2 (ANOMALIA), ma il modello resta multi-classe.
# a differenza di anomaly_oriented dove il problema è già binario.

# considerazione finale sulle differenze dei due file:
# in contesti dove si ha accesso a grandi dimensioni di dati il file anomaly_DB
# porterà sicuramente ad avere performance migliori dei modelli, ma nel caso in cui si abbia invece
# accesso ad una dimensione di dati limitata come quella che abbiamo noi il file anomaly_db
# porterebbe ad ancora più overfitting risultando così in dati poco affidabili in quanto falsati


# FUNZIONAMENTO DI CLASSFIFICATION MODELS ANOMALY ORIENTED E CLASSIFICATION MODELS STANDARD:

# prima di addestrare il modello devo droppare delle colonne per far si che l'addestramento
# sia sensato e privo di overfitting ovvero chiaramente togliendo prima di tutte la colonna 
# target ovvero i dati che vogliamo predire e poi tante altre colonne che creerebbero una 
# quantità di dati inutili dal momento che nel preprocessing utilizziamo onehotencoding
# che trasforma ogni variabile categorica in un formato numerico che il modello può comprendere
# così facendo però viene creata una colonna per ogni categoria distinta. ad esempio:
# La colonna ARTICOLO  può avere moltissimi valori diversi (alta cardinalità) con 
# articoli rari che compaiono solo 1-2 volte
# dunque con :

# TOP_N_ARTICOLI = 20
# articoli_top = counts.head(TOP_N_ARTICOLI).index
# 
# df['ARTICOLO_grouped'] = df['ARTICOLO'].where(
#     df['ARTICOLO'].isin(articoli_top),
#     other='ALTRO'
# )

# in questo modo mantengo solo i 20 articoli più frequenti, Tutti gli altri vengono trasformati in: ALTRO
# dal momemento poi che articolo_grouped è all'interno di categorical_cols 
# Nel preprocessing:
# ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
# Non sto creando una colonna per ogni articolo esistente ma 20 + 1 = 21 colonne
# in questo modo a differenza del frequency encoding non perdo l'identità dell'articolo
# come funziona frequency encoding invece: 

# senza frequency encoding: 

#    cliente  articolo  articolo_Mouse  articolo_Monitor  articolo_Tastiera
# 0    Mario     Mouse               1                 0                   0
# 1    Luigi  Tastiera               0                 0                   1
# 2 Giovanni     Mouse               1                 0                   0
# 3    Mario   Monitor               0                 1                   0
# 4     Anna  Tastiera               0                 0                   1

#con frequency encoding si avrà invece: 

#    cliente  articolo  articolo_freq
# 0    Mario     Mouse              2
# 1    Luigi  Tastiera              2
# 2 Giovanni     Mouse              2
# 3    Mario   Monitor              1
# 4     Anna  Tastiera              2

# il problema è che in questo modo si perde l'identità dell'articolo ad esempio nel nostro caso se
# una macchina è inefficiente ogni volta che deve produrre un mouse, il modello ora considera sia il mouse
# che la tastiera come lo stesso oggetto avendo la stessa frequenza in questo modo non è in grado di distinguere
# il fatto che magari la produzione del mouse è inefficiente mentre quella della tastiera si

# il campione di dati su cui vengono addestrati i modelli è molto limitato, poco meno di 400 righe per i 
# modelli è veramente poco per avere delle performance reali buone senza overfitting.
# i modelli che ho addestrato specialmente xgboost ottimizzato sono molto rigidi
# cercando di ridurre quanto possibile l'overfitting anche se come vediamo dall'output c'è sempre, data 
# la poca quantità di dati, inoltre a causa della rigidità di questi modelli le performance non sono
# eccellenti. per dati con quantità di righe maggiori, si potrebbe pensare di addestrare modelli 
# più complessi e meno regolarizzati avendo così in questo modo dei risultati molto più performanti
# ma con 387 righe si rischia di avere delle performance falsate.

# i risultati di anomaly oriented cercando di evitare l'overfitting con modelli più rigidi sono 
# peggiori o poco meglio su determinati valori rispetto a classiffication standard.

# classification_models_anomaly_bidata è il file di classification anomaly oriented ma 
# dove i modelli vengono addestrati in maniera più aggressiva senza preoccuparsi troppo 
# dell'overfitting, è dunque più adatto da usare in situazioni per le quali si hanno grandi
# quantitativi di dati in modo tale che i modelli possano essere addestrati correttemente 
# senza rischiare di inciampare nell'overfitting.

# COMMENTO DATI OTTENUTI 


# ===========================================================================================================================
# ========================================= VANNO AGGIORNATI ================================================================
# ===========================================================================================================================

# metriche calcolate:
# ACCURACY = calcola la percentuale di risposte corrette

# F1-SCORE = combina le metriche precisione e richiamo in un unico valore 
# Precisione (Precision): Indica quanti dei casi positivi predetti sono effettivamente positivi (precisione delle previsioni positive).
# Richiamo (Recall/Sensitivity): Indica quanti dei casi positivi identificati dal modello sono effettivamente reali (capacità di trovare tutti i casi positivi). 

# ROC-AUC = misura quanto un modello è in grado di distinguere tra le classi a diversi livelli di soglia.
# Curva ROC (Receiver Operating Characteristic): È un grafico che mostra la performance del modello tracciando due parametri
# al variare della soglia di classificazione:
# True Positive Rate (TPR) / Sensibilità / Recall: Indica la percentuale di casi positivi correttamente identificati.
# False Positive Rate (FPR): Indica la percentuale di casi negativi erroneamente classificati come positivi.
# AUC (Area Under the Curve): Rappresenta l'area geometrica sottesa alla curva ROC.

# F1-score (per classe) = Media armonica di precision e recall di quella singola classe
# F1-macro = Media semplice degli F1-score di tutte le classi (NORMALE + ATTENZIONE + ANOMALIA / 3), 
# senza pesare per quante osservazioni ha ogni classe
# F1-weighted = Media degli F1-score pesata per il numero di campioni (support) di ogni classe

# inizialmente bisogna decidere le soglie ovvero trasformare l'Indice_Inefficienza in classi discrete.
# in base alle quali classifichiamo come normale, attenzione e anomalia  i risultati che otteniamo. 
# soglie utilizzate nella classificazione standard basate sulla deviazione standard:
#   NORMALE     ≤ mean
#   ATTENZIONE  < mean < mean + 1std
#   ANOMALIA    > mean + 1std
# in base ai nostri dati:
#   NORMALE     ≤ 1.185 -> lavorazione nella norma
#   ATTENZIONE  1.185 - 1.47 -> lieve inefficienza, da monitorare
#   ANOMALIA    > 1.47 -> inefficienza significativa
# la media e l'std sono però più influenzati dalla skweness e non
# hanno risultato dare valori particolarmente performanti

# risultati allenando il modello in maniera standard:

# Distribuzione classi:
# NORMALE → 65.6%
# ATTENZIONE → 19.6%
# ANOMALIA → 14.7%
 
# Il migliore è:
 
# XGBoost
# Accuracy: 0.756
# F1-macro: 0.616
# ROC-AUC: 0.8921
# È coerente che batta Random Forest: con dataset piccoli e non lineari, XGBoost tende a essere più stabile.
 
# Logistic buono con (0.68 accuracy)
# Random Forest base collassa su ATTENZIONE
# XGBoost mantiene equilibrio sulle 3 classi
# Classe ATTENZIONE
# In tutti i modelli:
# Precision bassa (~0.42 XGB)
# Recall medio-bassa (~0.33)
# questo accade in quanto ATTENZIONE è una classe di confine statistico.
# È la zona grigia tra normale e anomalia.
# Il modello fa fatica perché i dati sono simili a NORMALE e le feature
# probabilmente non separano abbastanza bene.
# 
# come si può notare dalla matrice confusione i dati che vengono predetti meglio sono i dati
# relativi alla classe normale (47), mentre i peggiori sono della classe attenzione (5) e anomalie (7)

# risultati allenando il modello in maniera anomaly oriented:

# modificando il codice cambiando le soglie usando i quantili (60° percentile) (85° percentile) al posto 
# della media e dell'std che sono più influenzati dalla skweness in quanto i dati presentano una lunga coda 
# sulla destra e la maggiorparte dei dati sulla sinistra con il valore di 1.00 ovvero che la macchina performa 
# come dovrebbe e solo una piccola percentuale di volte risultano esserci degli outliers quando la macchina
# produce con un tempo maggiore o molto maggiore rispetto al tempo previsto, essendo influenzati da ciò
# non hanno risultato dare valori particolarmente performanti.

# soglie utilizzate nella classificazione standard basate sulla deviazione standard:
#   NORMALE     ≤ 60° percentile
#   ATTENZIONE  < 60° percentile < 85° percentile
#   ANOMALIA    > 85° percentile
# in base ai nostri dati:
#   NORMALE     ≤ 1.1466 -> lavorazione nella norma
#   ATTENZIONE  1.1466 - 1.4311 -> lieve inefficienza, da monitorare
#   ANOMALIA    > 1.4311 -> inefficienza significativa

# addestrando inoltre i modelli ottimizzati in modo tale che non 
# sia semplicemente una valutazione standard ma aumentando il recall delle anomalie ovvero:
# al posto di utilizzare y_pred = best_xgb.predict(X_test)
# che non permette di scegliere la soglia ma usa di default 0.5 
# rendendo il modello più bravo a non lasciarsi sfuggire i casi positivi reali, 
# accettando però il rischio di accusare qualche innocente per errore.

# nella classificazione anomaly oriented si è inoltre cambiato lo 
# scoring che è quel fattore che valuta i modelli basandoti solo su quanto sono bravi a identificare 
# correttamente gli esempi della classe ANOMALIE, ignorando le altre classi durante l'ottimizzazione
  
# risultati allenando il modello aumentando le recall delle anomalie:
# Distribuzione classi:
# NORMALE → (59.9%)
# ATTENZIONE → (25.1%)
# ANOMALIA → (15.0%)
# Il migliore è chiaramente:
# XGBoost Ottimizzata
# Accuracy: 0.794
# F1-macro: 0.759
# ROC-AUC: 0.9148
 
# come è possibile notare rispetto ai dati precedenti ogni metrica del modello che ha performato
# meglio (sempre XGBoost Ottimizzata) sono tutti migliorati
 
# Logistic è migliorato anch'esso da 0.68 a 0.74 di accuracy
  
# c'è stato un grosso miglioramento nei risultati di tutti i modelli, guardando la matrice di confusione
# però notiamo che i valori della classe normale sono diminuiti e sono ora (39), diminuiscono perché la
# soglia è scesa (60° percentile (1.1466) rispetto alla media (1.1855)), il miglioramento vero però è che
# ora ATTENZIONE e ANOMALIA hanno più campioni, i valori della classe attenzione ora sono (14), un netto
# miglioramento rispetto ai (5) precedente seppur ancora un dato abbastanza basso, (9) invece nella classe
# anomalia, piccolo miglioramente rispetto al (7) precedente. 
# un altro dato molto interessante è il ROC-AUC di XGBoost Ottimizzato che sale da 0.9037 a 0.9148. Questo 
# è particolarmente significativo perché ROC-AUC misura la capacità discriminativa del modello indipendentemente
# dalla soglia di classificazione, quindi il modello è genuinamente migliorato nella separazione delle classi, 
# non solo per effetto delle soglie percentili.
 
# a confermare ulteriormente i dati che abbiamo ottenuto nella matrice riguardo la classe ATTENZIONE nel 
# modello nuovo in XGBoost ottimizzato abbiamo notevoli miglioramenti di precision 
# (precision 0.42 -> precision 0.58) e di recall (recall 0.33 -> recall 0.74). 
# Rispetto al modello standard, il miglioramento è netto.
 
# c'è da fare una discriminante però avendo solo 387 righe totali e un test set di 78 campioni, 
# i risultati vanno letti con cautela statistica. 
# 1 predizione in più o in meno sulla classe ANOMALIA (12 campioni nel test) sposta 
# il recall dell'8%. I risultati sono promettenti ma andrebbero confermati su più dati.

# ===============================================================================================
# ========================================= OUTPUT ==============================================
# ===============================================================================================

# Soglia ANOMALIA (85° percentile): 1.4311
# DISTRIBUZIONE CLASSI:
#   Classe 0 (NORMALE): 329 campioni (85.0%)
#   Classe 1 (ANOMALIA): 58 campioni (15.0%)
# scale_pos_weight (calcolato su y_fit): 6.26
# 
# BASE MODELS:
# 
# ==================================================
#   Logistic Regression
# ==================================================
#               precision    recall  f1-score   support
# 
#      NORMALE       0.89      0.89      0.89        64
#     ANOMALIA       0.50      0.50      0.50        14
# 
#     accuracy                           0.82        78
#    macro avg       0.70      0.70      0.70        78
# weighted avg       0.82      0.82      0.82        78
# 
#   ROC-AUC: 0.7723
# 
# Generalizzazione - Logistic Regression
#   Training Accuracy:   0.8576
#   Validation Accuracy: 0.8205
#   Gap (train-test):    0.0371
#   Diagnosi:            GENERALIZZAZIONE BUONA (training ≈ validation/test)
# 
# ==================================================
#   Decision Tree
# ==================================================
#               precision    recall  f1-score   support
# 
#      NORMALE       0.88      0.94      0.91        64
#     ANOMALIA       0.60      0.43      0.50        14
# 
#     accuracy                           0.85        78
#    macro avg       0.74      0.68      0.70        78
# weighted avg       0.83      0.85      0.84        78
# 
#   ROC-AUC: 0.7009
# 
# Generalizzazione - Decision Tree
#   Training Accuracy:   0.8900
#   Validation Accuracy: 0.8462
#   Gap (train-test):    0.0438
#   Diagnosi:            GENERALIZZAZIONE BUONA (training ≈ validation/test)
# 
# ==================================================
#   Random Forest
# ==================================================
#               precision    recall  f1-score   support
# 
#      NORMALE       0.86      0.97      0.91        64
#     ANOMALIA       0.67      0.29      0.40        14
# 
#     accuracy                           0.85        78
#    macro avg       0.76      0.63      0.66        78
# weighted avg       0.83      0.85      0.82        78
# 
#   ROC-AUC: 0.7980
# 
# Generalizzazione - Random Forest
#   Training Accuracy:   0.9871
#   Validation Accuracy: 0.8462
#   Gap (train-test):    0.1409
#   Diagnosi:            OVERFITTING (training accuracy molto più alta della validation/test)
# 
# ==================================================
#   XGBoost
# ==================================================
#               precision    recall  f1-score   support
# 
#      NORMALE       0.91      0.95      0.93        64
#     ANOMALIA       0.73      0.57      0.64        14
# 
#     accuracy                           0.88        78
#    macro avg       0.82      0.76      0.79        78
# weighted avg       0.88      0.88      0.88        78
# 
#   ROC-AUC: 0.9152
# 
# Generalizzazione - XGBoost
#   Training Accuracy:   0.9935
#   Validation Accuracy: 0.8846
#   Gap (train-test):    0.1089
#   Diagnosi:            OVERFITTING (training accuracy molto più alta della validation/test)
# 
# ==================================================
#   SVM
# ==================================================
#               precision    recall  f1-score   support
# 
#      NORMALE       0.86      0.88      0.87        64
#     ANOMALIA       0.38      0.36      0.37        14
# 
#     accuracy                           0.78        78
#    macro avg       0.62      0.62      0.62        78
# weighted avg       0.78      0.78      0.78        78
# 
#   ROC-AUC: 0.7790
# 
# Generalizzazione - SVM
#   Training Accuracy:   0.8867
#   Validation Accuracy: 0.7821
#   Gap (train-test):    0.1047
#   Diagnosi:            OVERFITTING (training accuracy molto più alta della validation/test)
# 
# 
# RANDOM FOREST - GRID SEARCH
# Fitting 5 folds for each of 48 candidates, totalling 240 fits
# Migliori parametri RF: {'model__max_depth': 4, 'model__max_features': 0.7, 'model__min_samples_leaf': 10, 'model__min_samples_split': 10, 'model__n_estimators': 400}
# 
# ==================================================
#   Random Forest Ottimizzata (soglia 0.5)
# ==================================================
#               precision    recall  f1-score   support
# 
#      NORMALE       0.87      0.95      0.91        64
#     ANOMALIA       0.62      0.36      0.45        14
# 
#     accuracy                           0.85        78
#    macro avg       0.75      0.66      0.68        78
# weighted avg       0.83      0.85      0.83        78
# 
#   ROC-AUC: 0.8940
# 
# Generalizzazione - Random Forest Ottimizzata (soglia 0.5)
#   Training Accuracy:   0.9515
#   Validation Accuracy: 0.8462
#   Gap (train-test):    0.1053
#   Diagnosi:            OVERFITTING (training accuracy molto più alta della validation/test)
# 
# 
# XGBOOST - GRID SEARCH
# Fitting 5 folds for each of 576 candidates, totalling 2880 fits
# Migliori parametri XGB: {'model__colsample_bytree': 0.8, 'model__gamma': 1.0, 'model__learning_rate': 0.05, 'model__max_depth': 3, 'model__min_child_weight': 3, 'model__n_estimators': 200, 'model__reg_alpha': 0.0, 'model__reg_lambda': 3.0, 'model__scale_pos_weight': 9.39, 'model__subsample': 0.8}
# 
# ==================================================
#   XGBoost Ottimizzata (soglia 0.5)
# ==================================================
#               precision    recall  f1-score   support
# 
#      NORMALE       0.87      0.97      0.92        64
#     ANOMALIA       0.71      0.36      0.48        14
# 
#     accuracy                           0.86        78
#    macro avg       0.79      0.66      0.70        78
# weighted avg       0.84      0.86      0.84        78
# 
#   ROC-AUC: 0.8828
# 
# Generalizzazione - XGBoost Ottimizzata (soglia 0.5)
#   Training Accuracy:   0.9903
#   Validation Accuracy: 0.8590
#   Gap (train-test):    0.1313
#   Diagnosi:            OVERFITTING (training accuracy molto più alta della validation/test)
# 
# 
# CONFRONTO FINALE MODELLI:
#                     Model  Accuracy  F1-Anomalia  Recall-Anomalia
#                   XGBoost  0.884615     0.640000         0.571429
#       Logistic Regression  0.820513     0.500000         0.500000
#             Decision Tree  0.846154     0.500000         0.428571
# Random Forest Ottimizzata  0.846154     0.454545         0.357143
#                       SVM  0.782051     0.370370         0.357143
#       XGBoost Ottimizzata  0.858974     0.476190         0.357143
#             Random Forest  0.846154     0.400000         0.285714
#   Recall ANOMALIA: 0.5714
# Modello migliore: XGBoost  (Recall: 0.5714, F1: 0.6400)
# 
# STIMA ROBUSTA CV ESTERNA (TimeSeriesSplit 5-fold) - media ± std:
#   Accuracy:       0.8844 ± 0.0322
#   ROC-AUC:        0.8893 ± 0.0436