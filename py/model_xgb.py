"""Train an XGBoost classifier to predict nuclear localization of proteins.

Treina um classificador XGBoost para prever localização nuclear de proteínas,
com split por organismo, parada antecipada na validação, avaliação no teste e
previsões 0 ou 1 de todas as proteínas salvas em csv e em banco sql.
"""

import json
import sqlite3
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

# project folders / pastas do projeto
base_dir = Path(r"C:\Users\rafae\Documents\Projetos_data_science_IA\ETL\uniprot")
data_file = base_dir / "data" / "uniprot_target.csv"
output_dir = base_dir / "output"
sql_dir = base_dir / "sql"
db_file = sql_dir / "uniprot.db"
output_dir.mkdir(parents=True, exist_ok=True)
sql_dir.mkdir(parents=True, exist_ok=True)

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

# nuclear localization signals are rich in lysine and arginine, so add motif counts
# sinais de localização nuclear são ricos em lisina e arginina, então conta motivos com K e R
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

# class weight: negatives divided by positives in the training set
# peso da classe: negativos divididos por positivos no conjunto de treino
scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# xgboost on gpu with early stopping on the validation pr-auc
# xgboost na gpu com parada antecipada no pr-auc da validação
model = XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    device="cuda",
    max_bin=256,
    eval_metric="aucpr",
    early_stopping_rounds=50,
    random_state=seed,
)
start = time.time()
model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=100)
print(f"training time: {time.time() - start:.1f}s | best iteration: {model.best_iteration}")

# run predictions on cpu to avoid the device mismatch warning
# roda as previsões na cpu para evitar o aviso de dispositivos diferentes
model.set_params(device="cpu")

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
    "best_iteration": int(model.best_iteration),
    "baseline_pr_auc": float(y_test.mean()),
}
print(json.dumps(metrics, indent=2))
print(confusion_matrix(y_test, test_pred))
print(classification_report(y_test, test_pred, digits=3))

# feature importance by gain / importância das features por ganho
importance = pd.Series(model.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
print("top features by gain:")
print(importance.head(12))

# predict every protein with the validation threshold / prevê todas as proteínas com o limiar da validação
all_proba = model.predict_proba(features)[:, 1]
all_pred = (all_proba >= best_threshold).astype(int)

# mark which split each protein belongs to / marca a qual conjunto cada proteína pertence
split_name = np.full(len(df), "train", dtype=object)
split_name[val_idx] = "validation"
split_name[test_idx] = "test"

# build the predictions table / monta a tabela de previsões
predictions = df[["Entry", "Organism (ID)", "Length", "target"]].copy()
predictions["proba"] = all_proba.round(4)
predictions["pred"] = all_pred
predictions["correct"] = (predictions["pred"] == predictions["target"]).astype(int)
predictions["split"] = split_name

# save model, metrics and predictions in files / salva modelo, métricas e previsões em arquivos
joblib.dump({"model": model, "threshold": best_threshold, "features": list(features.columns)}, output_dir / "model_xgb.joblib")
with open(output_dir / "metrics_xgb.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
predictions.to_csv(output_dir / "predictions_xgb_all.csv", index=False)
predictions[predictions["split"] == "test"].to_csv(output_dir / "predictions_xgb_test.csv", index=False)

# save the predictions in the sql database / salva as previsões no banco de dados sql
predictions_sql = predictions.rename(
    columns={"Entry": "entry", "Organism (ID)": "organism_id", "Length": "length"}
)
with sqlite3.connect(db_file) as conn:
    predictions_sql.to_sql("predictions_xgb", conn, if_exists="replace", index=False, chunksize=50_000)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_pred_entry ON predictions_xgb (entry)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_split ON predictions_xgb (split)")

# quick check of the saved predictions / conferência rápida das previsões salvas
print(predictions["pred"].value_counts())
print(predictions.groupby("split")["correct"].mean().round(3))
print(f"files saved at: {output_dir}")
print(f"table predictions_xgb saved at: {db_file}")