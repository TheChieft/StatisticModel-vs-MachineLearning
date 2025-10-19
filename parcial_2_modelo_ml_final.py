# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#   kernelspec:
#     name: python3
#     display_name: Python 3
# ---

# %% [markdown]
# # Parcial 2 — Notebook Final (Modelo ML de Mora)
# **Objetivo**: (1) Predecir probabilidad de mora, (2) optimizar *Recall* y controlar sobreajuste
# (gap = Recall_10k − Recall_test), (3) elegir umbral operativo **τ** tal que la mora esperada
# entre aprobados sea ≤ 2.5%, y (4) proyectar ingresos a 1 año al 20% E.A. sobre 50.000 solicitudes.
#
# **Datasets**
# - `base_modelo_40k.csv`: train (40k) con `CLIENTE_MORA` (prevalencia esperada ≈ 7.86%).
# - `base_prueba_10k_sin_mora.csv`: 10k para scoring (sin objetivo) y proyección.
#
# **Escenarios de modelado**
# - A) Todas las variables.
# - B) Sin posibles *leakers*: `ESTADO_MORA_FIN`, `ESTADO_MORA_REAL` (y cualquier otra que refleje estado
#   posterior al otorgamiento). Por defecto, usamos **B**.
#
# **Entregables que produce este notebook**
# - Métricas en TEST a t=0.5 y t*=F2.
# - Selección de **τ** operativo (mora esperada ≤ 2.5%).
# - Sensibilidades: τ vs % aprobados, τ vs mora esperada, τ vs ingreso 1 año.
# - SHAP (importancias globales y dependencias).
# - **CSV** `base_prueba_10k_predicciones.csv`: `ID, p_mora, y_pred_eval, y_aprobado`.
# - Bitácora de decisiones.

# %%
import os, math, random, warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, fbeta_score
)
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# Opcionales
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

SEEDS = [42, 2021, 7]

# %% [markdown]
# ## 0) Carga de datos y chequeos

# %%
TRAIN_PATH = "base_modelo_40k.csv"
TEST10_PATH = "base_prueba_10k_sin_mora.csv"
TARGET = "CLIENTE_MORA"
ID_COL = "ID"

# Carga
df = pd.read_csv(TRAIN_PATH)
df10 = pd.read_csv(TEST10_PATH)

# Esquema
assert TARGET in df.columns, "CLIENTE_MORA debe existir en train"
assert TARGET not in df10.columns, "CLIENTE_MORA NO debe existir en la base 10k"

print("Train shape:", df.shape)
print("Test10 shape:", df10.shape)
print("Prevalencia (train):", round(df[TARGET].mean(), 5))

schema = pd.DataFrame({
    "col": df.columns,
    "dtype": df.dtypes.astype(str),
    "n_nulls": df.isna().sum(),
    "pct_nulls": (df.isna().mean()*100).round(2)
})
schema

# %% [markdown]
# ## 1) Detección de leakage y definición de escenarios A/B

# %%
leak_suspects = [c for c in ["ESTADO_MORA_FIN", "ESTADO_MORA_REAL"] if c in df.columns]
features_A = [c for c in df.columns if c != TARGET]
features_B = [c for c in features_A if c not in leak_suspects]

print("Posibles leakers:", leak_suspects)
print("Nº features A (todas):", len(features_A))
print("Nº features B (sin leakers):", len(features_B))

# Escenario por defecto (B)
FEATURES = features_B

# %% [markdown]
# ## 2) Particiones estratificadas y *preprocessing*

# %%
X = df[FEATURES].copy()
y = df[TARGET].astype(int).copy()

# 70/15/15 estratificado
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_tmp, y_tmp, test_size=0.1764706, stratify=y_tmp, random_state=42)

# Columnas categóricas / numéricas
cat_cols = X_train.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]

# Preprocesamiento: imputación + OHE
pre = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ]), cat_cols)
    ],
    remainder="drop",
    verbose_feature_names_out=False
)
pre.fit(X_train)

Xtr = pre.transform(X_train)
Xva = pre.transform(X_valid)
Xte = pre.transform(X_test)
X10 = pre.transform(df10[FEATURES])

# Nombres transformados
feat_names = []
feat_names += num_cols
if cat_cols:
    ohe = pre.named_transformers_["cat"].named_steps["ohe"]
    feat_names += ohe.get_feature_names_out(cat_cols).tolist()

# Desbalance: sugerencia de scale_pos_weight
spw = (y_train==0).sum() / max((y_train==1).sum(), 1)
print(f"scale_pos_weight sugerido ≈ {spw:.2f}")

# %% [markdown]
# ## 3) Modelos y *tuning* rápido
# Principal: **XGBoost** (early stopping, `scale_pos_weight`). Alternativas: **LogisticRegression** y **LightGBM**.

# %%
def fit_xgb_quick(Xtr, ytr, Xva, yva, spw, seed=42):
    assert HAS_XGB, "XGBoost no disponible"
    mdl = XGBClassifier(
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
        n_jobs=0, random_state=seed, scale_pos_weight=spw,
        n_estimators=800, learning_rate=0.05, max_depth=5, min_child_weight=2,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=1.0, reg_lambda=5.0
    )
    mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False, early_stopping_rounds=50)
    return mdl

# Entrenamos modelos
models = {}
# Baseline lineal
models["logreg"] = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs").fit(Xtr, y_train)
# XGBoost principal (si disponible)
if HAS_XGB:
    models["xgb"] = fit_xgb_quick(Xtr, y_train, Xva, y_valid, spw, seed=42)
# LightGBM alternativo (si disponible)
if HAS_LGBM:
    lgbm = lgb.LGBMClassifier(objective="binary", n_estimators=1500, learning_rate=0.05,
                              subsample=0.85, colsample_bytree=0.85, reg_alpha=1.0, reg_lambda=5.0,
                              n_jobs=-1, random_state=42)
    lgbm.set_params(**{"scale_pos_weight": spw})
    lgbm.fit(Xtr, y_train, eval_set=[(Xva, y_valid)], verbose=-1)
    models["lgbm"] = lgbm

list(models.keys())

# %% [markdown]
# ## 4) Calibración, umbrales y métricas (t=0.5 y t*=F2)

# %%
def calibrate_prefit(model, Xva, yva, method="sigmoid"):
    calib = CalibratedClassifierCV(model, method=method, cv="prefit")
    calib.fit(Xva, yva)
    return calib

def metrics_at(y_true, p, t):
    yhat = (p >= t).astype(int)
    return {
        "threshold": float(t),
        "recall": float(recall_score(y_true, yhat, zero_division=0)),
        "precision": float(precision_score(y_true, yhat, zero_division=0)),
        "f1": float(f1_score(y_true, yhat, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "confusion_matrix": confusion_matrix(y_true, yhat).tolist(),
    }

def pick_t_star_f2(yva, pva):
    grid = np.linspace(0.01, 0.99, 99)
    f2s = [fbeta_score(yva, (pva>=t).astype(int), beta=2, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f2s))]), float(np.max(f2s))

def find_tau_under_limit(p, limit=0.025, min_approved=50):
    taus = np.unique(np.sort(p))
    tau_chosen=None; mean_chosen=None; n_apr=0
    for tau in taus:
        mask = p < tau
        if mask.sum() < min_approved:
            continue
        mean_m = p[mask].mean()
        if mean_m <= limit:
            tau_chosen=float(tau); mean_chosen=float(mean_m); n_apr=int(mask.sum())
    return tau_chosen, mean_chosen, n_apr

def policy_summary(p, tau):
    mask = p < tau
    return {
        "%approved": float(mask.mean()*100),
        "expected_mora_among_approved": float(p[mask].mean()) if mask.any() else None,
        "n_approved": int(mask.sum())
    }

results = {}
for name, mdl in models.items():
    calib = calibrate_prefit(mdl, Xva, y_valid)
    pva = calib.predict_proba(Xva)[:,1]
    pte = calib.predict_proba(Xte)[:,1]

    t_star, f2_best = pick_t_star_f2(y_valid, pva)
    m_test_05 = metrics_at(y_test, pte, 0.5)
    m_test_t  = metrics_at(y_test, pte, t_star)

    tau, mean_m_valid, n_apr_valid = find_tau_under_limit(pva, 0.025, 50)
    pol_valid = policy_summary(pva, tau) if tau is not None else None
    pol_test  = policy_summary(pte, tau) if tau is not None else None

    results[name] = {
        "t*": t_star,
        "f2_valid": f2_best,
        "metrics_test_t0.5": m_test_05,
        "metrics_test_t*": m_test_t,
        "tau": tau,
        "policy_valid": pol_valid,
        "policy_test": pol_test,
        "calib": calib,
    }

pd.DataFrame({k: {"Recall@0.5": v["metrics_test_t0.5"]["recall"],
                  "Recall@t*":  v["metrics_test_t*"]["recall"],
                  "PR-AUC":     v["metrics_test_t*"]["pr_auc"]}
              for k,v in results.items()}).T.sort_values("Recall@t*", ascending=False)

# %% [markdown]
# ## 5) Selección del modelo para scoring 10k

# %%
# Elegimos por Recall@t* en TEST (prioridad del negocio)
best_name, best_pack = sorted(results.items(), key=lambda kv: kv[1]["metrics_test_t*"]["recall"], reverse=True)[0]
print("Mejor modelo:", best_name)
print("t* (F2):", best_pack["t*"])
print("tau (≤2.5% mora esperada):", best_pack["tau"])

best_calib = best_pack["calib"]

# %% [markdown]
# ## 6) Curvas de sensibilidad (τ)

# %%
def tau_sensitivity(p, tickets=[2_000_000,5_000_000,10_000_000], ea_rate=0.20, n_apps=50_000):
    taus = np.quantile(p, np.linspace(0.01, 0.99, 99))
    rows=[]
    for t in np.unique(taus):
        mask = p < t
        if mask.sum() < 50:  # evitar valores con muy pocos aprobados
            continue
        pct = mask.mean()*100
        mora = p[mask].mean()
        for tk in tickets:
            principal = (pct/100)*n_apps*tk
            ingreso   = principal * ea_rate
            rows.append({"tau": float(t), "%aprob": float(pct), "mora_esp": float(mora),
                         "ticket": tk, "interes_1y": float(ingreso)})
    return pd.DataFrame(rows)

pva_best = best_calib.predict_proba(Xva)[:,1]
sens_df = tau_sensitivity(pva_best)
sens_df.head()

# %% [markdown]
# ## 7) Proyección financiera (50.000, 20% E.A.)

# %%
# Usamos TEST para evitar optimismo
pte_best = best_calib.predict_proba(Xte)[:,1]
mask_apr = pte_best < best_pack["tau"]
pct_aprob_test = mask_apr.mean()*100
mora_esperada_test = pte_best[mask_apr].mean()

print({"%aprob(TEST)": pct_aprob_test, "mora_esp(TEST)": mora_esperada_test})

def financial_projection(n_apps, pct_approved, expected_mora, ea_rate=0.20, tickets=[2_000_000,5_000_000,10_000_000], lgd_list=[0.3,0.6]):
    rows=[]
    approved = n_apps*(pct_approved/100)
    for tk in tickets:
        P = approved * tk
        I = P * ea_rate
        for lgd in lgd_list:
            loss = P * expected_mora * lgd
            net  = I - loss
            rows.append({
                "ticket": tk, "LGD": lgd, "approved_count": int(approved),
                "principal_approved": P, "interest_1y": I,
                "expected_mora": expected_mora, "expected_loss": loss,
                "net_interest_minus_loss": net
            })
    return pd.DataFrame(rows)

proj_df = financial_projection(50_000, pct_aprob_test, mora_esperada_test, 0.20, [2_000_000,5_000_000,10_000_000], [0.3,0.6])
proj_df

# %% [markdown]
# ## 8) SHAP — explicabilidad y estabilidad

# %%
if HAS_SHAP and ("xgb" in models):
    explainer = shap.TreeExplainer(models["xgb"])
    shap_values = explainer.shap_values(Xte)
    abs_mean = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({"feature": feat_names, "abs_mean_shap": abs_mean}).sort_values("abs_mean_shap", ascending=False)
    display(shap_importance.head(10))
    # shap.summary_plot(shap_values, Xte, feature_names=feat_names)  # ejecutar localmente si se desea

# %% [markdown]
# ## 9) Scoring 10k y exportación CSV

# %%
p10 = best_calib.predict_proba(X10)[:,1]
y_pred_eval_10k = (p10 >= best_pack["t*"]).astype(int)
y_aprobado_10k  = (p10 < best_pack["tau"]).astype(int)

ids = df10[ID_COL] if ID_COL in df10.columns else pd.Series(np.arange(len(df10)), name="ID_AUTOGEN")
out10 = pd.DataFrame({"ID": ids, "p_mora": p10, "y_pred_eval": y_pred_eval_10k, "y_aprobado": y_aprobado_10k})
out10.to_csv("base_prueba_10k_predicciones.csv", index=False)
print("Archivo exportado: base_prueba_10k_predicciones.csv")
out10.head()

# %% [markdown]
# ## 10) Bitácora de decisiones
# - **Leakage**: escenario B excluye variables sospechosas (`ESTADO_MORA_FIN`, `ESTADO_MORA_REAL`).
# - **Balanceo**: `scale_pos_weight ≈ N_neg/N_pos` como base (∼11.7–12.0).
# - **No se usa** el 10k para *tuning* (sólo *scoring* y proyección).
# - **Calibración**: Platt (`sigmoid`) tras *early stopping* o ajuste del modelo.
# - **Umbrales separados**: `t*` (Recall/F2) vs `τ` (política ≤ 2.5%).
# - **Semillas**: 42 predeterminada; repetir con 2021 y 7 si se requiere robustez.
# - **Proyección**: 50k, 20% E.A., sensibilidad en ticket (2M/5M/10M) y LGD (0.3/0.6).
