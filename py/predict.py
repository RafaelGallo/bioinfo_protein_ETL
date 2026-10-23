"""Predict nuclear localization for the new (unannotated) proteins.

Aplica o modelo XGBoost treinado nas proteínas novas (TrEMBL, sem localização
anotada), usando as mesmas features e o limiar escolhido na validação, e salva
as previsões em csv e no banco sql.
"""

import sqlite3
import sys
from pathlib import Path

import joblib
import pandas as pd

# project folders / pastas do projeto
base_dir = Path(r"C:\Users\rafae\Documents\Projetos_data_science_IA\ETL\uniprot")
sys.path.insert(0, str(base_dir))

from src.transform import build_features, clean_sequences  # noqa: E402

input_file = base_dir / "data" / "uniprot_new.parquet"
model_file = base_dir / "models" / "model_xgb.joblib"
output_dir = base_dir / "output"
db_file = base_dir / "sql" / "uniprot.db"
output_dir.mkdir(parents=True, exist_ok=True)

# load the trained model, threshold and feature names / carrega modelo, limiar e nomes das features
bundle = joblib.load(model_file)
model = bundle["model"]
threshold = bundle["threshold"]
feature_names = bundle["features"]
print(f"model loaded | threshold: {threshold:.3f} | features: {len(feature_names)}")

# load the new proteins / carrega as proteínas novas
df = pd.read_parquet(input_file)
print(f"rows loaded: {len(df)}")

# same sequence filter used in training / mesmo filtro de sequência usado no treino
before = len(df)
df = clean_sequences(df)
print(f"removed non standard sequences: {before - len(df)}")

# build features in the exact column order the model expects / monta as features na ordem que o modelo espera
features = build_features(df)[feature_names]

# predict with the validation threshold / prevê com o limiar da validação
model.set_params(device="cpu")
proba = model.predict_proba(features)[:, 1]

predictions = df[["Entry", "Entry Name", "Protein names", "Organism", "Organism (ID)"]].copy()
predictions["length"] = df["Sequence"].str.len()
predictions["proba"] = proba.round(4)
predictions["pred"] = (proba >= threshold).astype(int)

print(predictions["pred"].value_counts())
print(f"predicted nuclear: {predictions['pred'].mean():.2%}")

# save the predictions in csv / salva as previsões em csv
output_file = output_dir / "predictions_xgb_new.csv"
predictions.to_csv(output_file, index=False)
print(f"file saved at: {output_file}")

# save the predictions in the sql database / salva as previsões no banco de dados sql
predictions_sql = predictions.rename(
    columns={
        "Entry": "entry",
        "Entry Name": "entry_name",
        "Protein names": "protein_names",
        "Organism": "organism",
        "Organism (ID)": "organism_id",
    }
)
with sqlite3.connect(db_file) as conn:
    predictions_sql.to_sql("predictions_xgb_new", conn, if_exists="replace", index=False, chunksize=50_000)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_pred_new_entry ON predictions_xgb_new (entry)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_new_pred ON predictions_xgb_new (pred)")
print(f"table predictions_xgb_new saved at: {db_file}")
