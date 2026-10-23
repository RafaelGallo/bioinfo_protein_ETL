"""Train a logistic regression to predict nuclear localization of proteins.

Treina uma regressão logística para prever localização nuclear de proteínas,
com features da sequência, split por organismo, ajuste de limiar na validação
e avaliação final no teste.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# project folders / pastas do projeto
base_dir = Path(r"C:\Users\rafae\Documents\Projetos_data_science_IA\ETL\uniprot")
data_file = base_dir / "data" / "uniprot_target.csv"
output_dir = base_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# True = keep only organisms with at least one positive (proxy for eukaryotes)
# True = mantém só organismos com ao menos um positivo (aproximação para eucariotos)
only_eukaryote_proxy = True

# random seed for reproducibility / semente aleatória para reprodutibilidade
seed = 42

# load only the columns we need / carrega só as colunas necessárias
df = pd.read_csv(data_file, usecols=["Entry", "Organism (ID)", "Length", "Sequence", "target"])
print(f"rows loaded: {len(df)} | positive rate: {df['target'].mean():.2%}")

# optional organism filter / filtro opcional por organismo
if only_eukaryote_proxy:
    positives_by_organism = df.groupby("Organism (ID)")["target"].transform("sum")
    df = df[positives_by_organism > 0].reset_index(drop=True)
    print(f"rows after organism filter: {len(df)} | positive rate: {df['target'].mean():.2%}")

# amino acid composition features (20 columns) / features de composição de aminoácidos (20 colunas)
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
seq_len = df["Sequence"].str.len()
features = pd.DataFrame(index=df.index)
for aa in amino_acids:
    features[f"aa_{aa}"] = df["Sequence"].str.count(aa) / seq_len

# physicochemical features from composition / features físico-químicas a partir da composição
kyte_doolittle = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
features["log_length"] = np.log(seq_len)
features["gravy"] = sum(features[f"aa_{aa}"] * value for aa, value in kyte_doolittle.items())
features["frac_positive"] = features[["aa_K", "aa_R"]].sum(axis=1)
features["frac_negative"] = features[["aa_D", "aa_E"]].sum(axis=1)
features["net_charge_frac"] = features["frac_positive"] - features["frac_negative"]
features["frac_aromatic"] = features[["aa_F", "aa_W", "aa_Y"]].sum(axis=1)

# nuclear localization signals are rich in lysine and arginine, so add k-mer counts
# sinais de localização nuclear são ricos em lisina e arginina, então conta trechos com K e R
features["kr_motif_count"] = df["Sequence"].str.count(r"[KR]{4,}")
features["kr_motif_pat7"] = df["Sequence"].str.count(r"P[KR]{2,}[A-Z][KR]")

y = df["target"].to_numpy()
groups = df["Organism (ID)"].to_numpy()
print(f"features: {features.shape[1]}")

# split by organism: test first, then validation from the remaining train
# divide por organismo: primeiro o teste, depois a validação a partir do treino restante
outer_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
train_val_idx, test_idx = next(outer_split.split(features, y, groups))

inner_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
train_rel, val_rel = next(inner_split.split(features.iloc[train_val_idx], y[train_val_idx], groups[train_val_idx]))
train_idx = train_val_idx[train_rel]
val_idx = train_val_idx[val_rel]

x_train, y_train = features.iloc[train_idx], y[train_idx]
x_val, y_val = features.iloc[val_idx], y[val_idx]
x_test, y_test = features.iloc[test_idx], y[test_idx]
print(f"train: {len(x_train)} | validation: {len(x_val)} | test: {len(x_test)}")
print(f"positive rate train/val/test: {y_train.mean():.2%} / {y_val.mean():.2%} / {y_test.mean():.2%}")

# logistic regression with scaling and balanced class weights
# regressão logística com padronização e pesos de classe balanceados
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "logreg",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                random_state=seed,
            ),
        ),
    ]
)
model.fit(x_train, y_train)

# choose the threshold that maximizes f1 on validation / escolhe o limiar que maximiza o f1 na validação
val_proba = model.predict_proba(x_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, val_proba)
f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
best_threshold = float(thresholds[np.argmax(f1_scores)])
print(f"best threshold on validation: {best_threshold:.3f}")

# final evaluation on the test set / avaliação final no conjunto de teste
test_proba = model.predict_proba(x_test)[:, 1]
test_pred = (test_proba >= best_threshold).astype(int)

metrics = {
    "roc_auc": float(roc_auc_score(y_test, test_proba)),
    "pr_auc": float(average_precision_score(y_test, test_proba)),
    "f1": float(f1_score(y_test, test_pred)),
    "threshold": best_threshold,
    "baseline_pr_auc": float(y_test.mean()),
}
print(json.dumps(metrics, indent=2))
print(confusion_matrix(y_test, test_pred))
print(classification_report(y_test, test_pred, digits=3))

# most important coefficients / coeficientes mais importantes
coefs = pd.Series(model.named_steps["logreg"].coef_[0], index=features.columns)
print("top positive coefficients:")
print(coefs.sort_values(ascending=False).head(8))
print("top negative coefficients:")
print(coefs.sort_values().head(8))

# save model, metrics and test predictions / salva modelo, métricas e previsões do teste
joblib.dump({"model": model, "threshold": best_threshold, "features": list(features.columns)}, output_dir / "model_logreg.joblib")
with open(output_dir / "metrics_logreg.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
predictions = df.iloc[test_idx][["Entry", "Organism (ID)", "target"]].copy()
predictions["proba"] = test_proba
predictions["pred"] = test_pred
predictions.to_csv(output_dir / "predictions_logreg_test.csv", index=False)
print(f"files saved at: {output_dir}")