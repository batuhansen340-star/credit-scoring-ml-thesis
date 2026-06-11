"""
================================================================================
BA DEGREE PROJECT 2 — CREDIT SCORING ML PIPELINE  (methodology-uyumlu, v2)
================================================================================
Batuhan SEN | 200527078 | Istinye University

Bu script, dokumanin Methodology (5.3-5.6) bolumunu BIREBIR uygular ve
notebook'lardaki (02_preprocessing + 03_modeling) ile AYNI yontemleri tek
dosyada toplar. Amac: FINAL sonuclari TAM VERIYLE (307K) tek komutta uretmek.

KULLANIM:
  Gelistirme (hizli): SAMPLE_SIZE = 50000
  FINAL (tam veri):   SAMPLE_SIZE = None   (Boruta + n_iter=100 ile saatler surebilir)
  Terminalde:         python credit_scoring_pipeline.py

METHODOLOGY UYUMU (taslak v1'den farklar):
  + SMOTE (yalniz training fold, imblearn Pipeline ile)  [5.3 Class Imbalance]
  + Target encoding (yuksek kardinalite)                 [5.3 Feature Encoding]
  + Feature selection: near-zero + korr>0.85 + Boruta    [5.3 Feature Selection]
  + Tum donusumler yalniz train'de fit (leakage yok)     [5.6 Experimental Procedure]
  + Optimal threshold (F1-max, out-of-fold)              [5.5 Evaluation]
  + RandomizedSearchCV n_iter=100                         [5.6]
================================================================================
"""

# ============================================================================
# BOLUM 0: KUTUPHANELER VE AYARLAR
# ============================================================================
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import (train_test_split, RandomizedSearchCV,
                                      StratifiedKFold, cross_val_score, cross_val_predict)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, f1_score, precision_score,
                             recall_score, confusion_matrix)
from category_encoders import TargetEncoder
from boruta import BorutaPy

# Modeller
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Class imbalance (SMOTE) — imblearn pipeline CV fold'unda uygular
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Istatistik
from scipy import stats
from itertools import combinations

warnings.filterwarnings("ignore")

# ---- AYARLAR (buradan kontrol et) ----
RANDOM_STATE  = 42          # tekrarlanabilirlik (her yerde sabit seed)
SAMPLE_SIZE   = int(os.environ["PIPE_SAMPLE"]) if os.environ.get("PIPE_SAMPLE") else None  # None=tam 307K
N_ITER        = int(os.environ.get("PIPE_NITER", "100"))  # tezdeki kosu: 100 (tam veride hiz icin PIPE_NITER=50 verilebilir)
CV_FOLDS      = 5
BORUTA_MAXN   = 50000       # Boruta'yi en fazla bu kadar satirla calistir (hiz; secim icin yeterli)
DATA_PATH     = "data/application_train.csv"
FIG_DIR       = "outputs/figures"
RES_DIR       = "outputs/results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

print("=" * 70)
print("CREDIT SCORING PIPELINE (methodology-uyumlu)")
print(f"Ornek: {SAMPLE_SIZE if SAMPLE_SIZE else 'TAM VERI (307K)'} | N_ITER={N_ITER}")
print("=" * 70)


# ============================================================================
# BOLUM 1: VERI YUKLEME + ORNEKLEM
# ============================================================================
print("\n[1/9] Veri yukleniyor...")
df = pd.read_csv(DATA_PATH)
if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
    df, _ = train_test_split(df, train_size=SAMPLE_SIZE,
                             stratify=df["TARGET"], random_state=RANDOM_STATE)
    df = df.reset_index(drop=True)
print(f"  {df.shape[0]:,} satir | default %{df['TARGET'].mean()*100:.2f}")


# ============================================================================
# BOLUM 2: SPLIT-ONCESI TEMIZLIK (sizintisiz, satir-bazli)
# ============================================================================
print("\n[2/9] Split-oncesi temizlik...")
# DAYS_EMPLOYED anomalisi (365243 = ~1000 yil placeholder) -> NaN + flag
df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
# XNA -> NaN
for kol in ["CODE_GENDER", "ORGANIZATION_TYPE"]:
    df[kol] = df[kol].replace("XNA", np.nan)
# Turetilmis bankacilik oranlari + yas
df["CREDIT_INCOME_RATIO"]  = df["AMT_CREDIT"]  / df["AMT_INCOME_TOTAL"]
df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
df["PAYMENT_RATE"]         = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
df["DAYS_EMPLOYED_PERC"]   = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
df["AGE_YEARS"]            = -df["DAYS_BIRTH"] / 365

y = df["TARGET"]
X = df.drop(columns=["TARGET", "SK_ID_CURR"])
high_missing = X.columns[X.isnull().mean() > 0.60]   # >%60 eksik kolonlari at
X = X.drop(columns=high_missing)
print(f"  %60+ eksik {len(high_missing)} kolon atildi -> {X.shape[1]} degisken")


# ============================================================================
# BOLUM 3: TRAIN/TEST SPLIT (80:20 stratified)
# ============================================================================
print("\n[3/9] Train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print(f"  Train {X_train.shape[0]:,} | Test {X_test.shape[0]:,}")


# ============================================================================
# BOLUM 4: ENCODING (yalniz train'de fit)
# ============================================================================
print("\n[4/9] Encoding (one-hot + target)...")
kategorik = X_train.select_dtypes("object").columns.tolist()
for kol in kategorik:                                  # kategorik eksik -> 'Missing'
    X_train[kol] = X_train[kol].fillna("Missing")
    X_test[kol]  = X_test[kol].fillna("Missing")
kard   = X_train[kategorik].nunique()
dusuk  = kard[kard < 10].index.tolist()                # one-hot
yuksek = kard[kard >= 10].index.tolist()               # target encoding

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
ohe.fit(X_train[dusuk])
tr_ohe = pd.DataFrame(ohe.transform(X_train[dusuk]), columns=ohe.get_feature_names_out(dusuk), index=X_train.index)
te_ohe = pd.DataFrame(ohe.transform(X_test[dusuk]),  columns=ohe.get_feature_names_out(dusuk), index=X_test.index)

tenc = TargetEncoder(cols=yuksek)
tenc.fit(X_train[yuksek], y_train)
tr_te = tenc.transform(X_train[yuksek])
te_te = tenc.transform(X_test[yuksek])

sayisal = X_train.select_dtypes("number").columns.tolist()
X_train = pd.concat([X_train[sayisal], tr_ohe, tr_te], axis=1)
X_test  = pd.concat([X_test[sayisal],  te_ohe, te_te], axis=1)
print(f"  Encoding sonrasi {X_train.shape[1]} feature")


# ============================================================================
# BOLUM 5: IMPUTATION + WINSORIZATION (yalniz train'de fit)
# ============================================================================
print("\n[5/9] Imputation (median) + winsorization (1-99)...")
imp = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test  = pd.DataFrame(imp.transform(X_test),      columns=X_test.columns,  index=X_test.index)
for kol in sayisal:                                    # train sinirlariyla kirp
    alt, ust = X_train[kol].quantile([0.01, 0.99])
    X_train[kol] = X_train[kol].clip(alt, ust)
    X_test[kol]  = X_test[kol].clip(alt, ust)
print(f"  Eksik kalan: {int(X_train.isnull().sum().sum())}")


# ============================================================================
# BOLUM 6: FEATURE SELECTION (near-zero + korr>0.85 + Boruta)
# ============================================================================
print("\n[6/9] Feature selection...")
# 6.1 Near-zero variance
vt = VarianceThreshold(0.0); vt.fit(X_train)
sabit = X_train.columns[~vt.get_support()].tolist()
X_train = X_train.drop(columns=sabit); X_test = X_test.drop(columns=sabit)
# 6.2 Korelasyon>0.85 (dusuk univariate-AUC olani at)
uni = {c: max(roc_auc_score(y_train, X_train[c]), 1 - roc_auc_score(y_train, X_train[c]))
       for c in X_train.columns}
corr = X_train.corr().abs()
ust = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
at_corr = set()
for c in ust.columns:
    for r in ust.index[ust[c] > 0.85]:
        at_corr.add(c if uni[c] < uni[r] else r)
X_train = X_train.drop(columns=list(at_corr)); X_test = X_test.drop(columns=list(at_corr))
# 6.3 Boruta (hiz icin en fazla BORUTA_MAXN satirla)
if len(X_train) > BORUTA_MAXN:
    Xb, _, yb, _ = train_test_split(X_train, y_train, train_size=BORUTA_MAXN,
                                    stratify=y_train, random_state=RANDOM_STATE)
else:
    Xb, yb = X_train, y_train
rf_b = RandomForestClassifier(max_depth=6, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
boruta = BorutaPy(rf_b, n_estimators=80, max_iter=40, random_state=RANDOM_STATE, verbose=0)
boruta.fit(Xb.values, yb.values)
confirmed = X_train.columns[boruta.support_].tolist()
tentative = X_train.columns[boruta.support_weak_].tolist()
secilen = confirmed if len(confirmed) >= 20 else confirmed + tentative
X_train = X_train[secilen]; X_test = X_test[secilen]
print(f"  near-zero {len(sabit)} + korr {len(at_corr)} atildi; Boruta {len(secilen)} sececti")
pd.Series(secilen).to_csv(f"{RES_DIR}/selected_features.csv", index=False, header=["feature"])


# ============================================================================
# BOLUM 7: MODEL EGITIMI (SMOTE pipeline + RandomizedSearchCV)
# ============================================================================
print("\n[7/9] Model egitimi (bu uzun surebilir)...")
smote = SMOTE(random_state=RANDOM_STATE)
configs = {
 "Logistic Regression": (ImbPipeline([("sc", StandardScaler()), ("sm", smote),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
    {"clf__C": np.logspace(-3, 2, 20)}),
 "Random Forest": (ImbPipeline([("sm", smote),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1))]),
    {"clf__n_estimators": [100, 200, 300, 500], "clf__max_depth": [5, 10, 15, 20],
     "clf__min_samples_split": [5, 10, 20, 50], "clf__max_features": ["sqrt", "log2", 0.5]}),  # 03_modeling.ipynb (tez kosusu) ile birebir ayni grid
 "XGBoost": (ImbPipeline([("sm", smote),
    ("clf", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1, verbosity=0))]),
    {"clf__learning_rate": [0.01, 0.05, 0.1, 0.3], "clf__max_depth": [3, 5, 7, 10],
     "clf__reg_alpha": [0, 1, 5, 10], "clf__reg_lambda": [0, 1, 5, 10],
     "clf__subsample": [0.7, 0.8, 1.0], "clf__colsample_bytree": [0.7, 0.8, 1.0]}),
 "LightGBM": (ImbPipeline([("sm", smote),
    ("clf", LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, verbose=-1))]),
    {"clf__num_leaves": [20, 50, 100, 150], "clf__learning_rate": [0.01, 0.05, 0.1, 0.3],
     "clf__min_child_samples": [10, 30, 50, 100], "clf__colsample_bytree": [0.7, 0.8, 1.0]}),
}
cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results, cv_scores, trained, roc_data, thresholds = {}, {}, {}, {}, {}
for name, (pipe, params) in configs.items():
    t0 = time.time()
    search = RandomizedSearchCV(pipe, params, n_iter=N_ITER, cv=cv, scoring="roc_auc",
                                random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_; trained[name] = best
    cv_scores[name] = cross_val_score(best, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    ths = np.linspace(0.05, 0.95, 91)
    th = ths[int(np.argmax([f1_score(y_train, (oof >= t).astype(int)) for t in ths]))]
    thresholds[name] = th
    proba = best.predict_proba(X_test)[:, 1]; pred = (proba >= th).astype(int)
    auc = roc_auc_score(y_test, proba)
    results[name] = {"AUC-ROC": round(auc, 4), "Gini": round(2*auc-1, 4),
        "F1": round(f1_score(y_test, pred), 4), "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4), "Threshold": round(th, 3),
        "CV_AUC_mean": round(cv_scores[name].mean(), 4), "CV_AUC_std": round(cv_scores[name].std(), 4)}
    fpr, tpr, _ = roc_curve(y_test, proba); roc_data[name] = (fpr, tpr, auc)
    print(f"  {name:22s} AUC={auc:.4f} F1={results[name]['F1']:.3f} ({time.time()-t0:.0f}s)")


# ============================================================================
# BOLUM 8: SONUC TABLOSU + GRAFIKLER + ISTATISTIK
# ============================================================================
print("\n[8/9] Sonuclar ve grafikler...")
results_df = pd.DataFrame(results).T.sort_values("AUC-ROC", ascending=False)
results_df.to_csv(f"{RES_DIR}/model_comparison.csv")
best_name = results_df.index[0]
print(results_df.to_string())
print(f"  >>> EN IYI: {best_name} (AUC={results_df.iloc[0]['AUC-ROC']})")

renkler = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
# ROC
plt.figure(figsize=(8, 7))
for (name, (fpr, tpr, auc)), c in zip(sorted(roc_data.items(), key=lambda x: -x[1][2]), renkler):
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", lw=2, color=c)
plt.plot([0, 1], [0, 1], "k--", alpha=0.5); plt.legend(loc="lower right")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Egrileri", fontweight="bold"); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/model_roc_comparison.png", dpi=150, bbox_inches="tight"); plt.close()
# Confusion matrix
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for (name, model), ax in zip(trained.items(), axes.ravel()):
    pred = (model.predict_proba(X_test)[:, 1] >= thresholds[name]).astype(int)
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Odedi", "Default"], yticklabels=["Odedi", "Default"])
    ax.set_title(name, fontweight="bold")
plt.suptitle("Confusion Matrix", fontweight="bold"); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/model_confusion_all.png", dpi=150, bbox_inches="tight"); plt.close()

# Paired t-test + Bonferroni
names = list(cv_scores.keys()); n_comp = len(list(combinations(names, 2))); alpha = 0.05 / n_comp
rows = []
for m1, m2 in combinations(names, 2):
    t, p = stats.ttest_rel(cv_scores[m1], cv_scores[m2])
    rows.append({"Model 1": m1, "Model 2": m2,
        "Mean Diff": round(cv_scores[m1].mean()-cv_scores[m2].mean(), 4),
        "t-stat": round(t, 3), "p-value": round(p, 4),
        "Significant": "EVET" if p < alpha else "Hayir"})
pd.DataFrame(rows).to_csv(f"{RES_DIR}/statistical_tests.csv", index=False)
print(f"  Istatistik testleri kaydedildi (Bonferroni alpha={alpha:.4f})")


# ============================================================================
# BOLUM 9: SHAP (en iyi model)
# ============================================================================
print("\n[9/9] SHAP analizi...")
try:
    import shap
    best = trained[best_name]; clf = best.named_steps["clf"]
    X_shap = X_test.sample(min(1000, len(X_test)), random_state=RANDOM_STATE)
    if "sc" in best.named_steps:                       # LogReg: scale + LinearExplainer
        Xs = pd.DataFrame(best.named_steps["sc"].transform(X_shap), columns=X_shap.columns, index=X_shap.index)
        sv = shap.LinearExplainer(clf, Xs).shap_values(Xs); Xplot = Xs
    else:                                              # Tree: TreeExplainer
        sv = shap.TreeExplainer(clf).shap_values(X_shap)
        if isinstance(sv, list): sv = sv[1]
        Xplot = X_shap
    plt.figure(); shap.summary_plot(sv, Xplot, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
    plt.figure(); shap.summary_plot(sv, Xplot, plot_type="bar", show=False, max_display=15)
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/shap_bar.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"  SHAP grafikleri kaydedildi (en iyi model: {best_name})")
except Exception as e:
    print(f"  SHAP hatasi (atlanabilir): {e}")

print("\n" + "=" * 70)
print("PIPELINE TAMAMLANDI! Sonuclar: outputs/results/ ve outputs/figures/")
print("=" * 70)
