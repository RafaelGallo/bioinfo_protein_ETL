"""Create the binary target column for protein subcellular localization.

Cria a coluna alvo binária (0 ou 1) a partir do texto de localização subcelular
do UniProt, no formato um contra todos, limpa sequências inválidas e duplicadas,
remove colunas que vazam o alvo e salva o resultado em parquet, csv e banco sql.
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd

# project folders / pastas do projeto
base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"
sql_dir = base_dir / "sql"
input_file = data_dir / "uniprot_raw.parquet"
output_parquet = data_dir / "uniprot_target.parquet"
output_csv = data_dir / "uniprot_target.csv"
db_file = sql_dir / "uniprot.db"
data_dir.mkdir(parents=True, exist_ok=True)
sql_dir.mkdir(parents=True, exist_ok=True)

# the raw parquet may live in input when data was cloned from github
# o parquet bruto pode estar em input quando os dados vieram do github
if not input_file.exists():
    input_file = base_dir / "input" / "uniprot_raw.parquet"

# location of interest and its regex, change to build another target
# localização de interesse e seu regex, troque para criar outro alvo
target_name = "nucleus"
target_pattern = r"\bnucle(?:us|ar)\b"

# True = exclude proteins located only in the nuclear envelope or membrane
# True = exclui proteínas localizadas apenas no envelope ou membrana nuclear
strict_nucleus = False
strict_pattern = r"\bnucleus\b(?!\s+(?:membrane|envelope|inner|outer|lamina))"

# True = keep only organisms that have at least one positive protein (proxy for eukaryotes)
# True = mantém só organismos com ao menos uma proteína positiva (aproximação para eucariotos)
only_eukaryote_proxy = False

# columns used to build the target or that leak it / colunas usadas para criar o alvo ou que vazam o alvo
location_col = "Subcellular location [CC]"
leak_cols = ["Gene Ontology IDs"]

# load the parquet file / carrega o arquivo parquet
df = pd.read_parquet(input_file)
print(f"rows loaded: {len(df)}")
print(f"columns: {list(df.columns)}")

# stop with a clear message if the location column is missing / para com mensagem clara se a coluna de localização não existir
if location_col not in df.columns:
    raise SystemExit(
        f"column '{location_col}' not found in {input_file.name}; "
        "use the training file (uniprot_raw.parquet), not the new proteins file"
    )

# keep only proteins with location text and sequence / mantém só proteínas com texto de localização e sequência
df = df.dropna(subset=[location_col, "Sequence"]).copy()

# keep only standard amino acids / mantém só aminoácidos padrão
before = len(df)
df = df[df["Sequence"].str.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$")].copy()
print(f"removed non standard sequences: {before - len(df)}")

# remove evidence tags, prefix and notes so only locations remain
# remove tags de evidência, prefixo e notas para sobrar apenas as localizações
clean_text = (
    df[location_col]
    .str.replace(r"\{[^}]*\}", "", regex=True)
    .str.replace(r"Note=.*?(?=\[Isoform|$)", "", regex=True, flags=re.DOTALL)
    .str.replace("SUBCELLULAR LOCATION:", "", regex=False)
)

# target: 1 if any location matches, 0 otherwise / alvo: 1 se alguma localização bate, 0 caso contrário
pattern = strict_pattern if strict_nucleus else target_pattern
df["target"] = clean_text.str.contains(pattern, flags=re.IGNORECASE, regex=True).astype(int)

# remove sequences with conflicting labels, then exact duplicates
# remove sequências com rótulos conflitantes e depois duplicatas exatas
before = len(df)
conflict = df.groupby("Sequence")["target"].transform("nunique") > 1
df = df[~conflict].drop_duplicates(subset="Sequence").reset_index(drop=True)
print(f"removed conflicting or duplicated sequences: {before - len(df)}")

# optional filter: organisms with at least one positive / filtro opcional: organismos com ao menos um positivo
if only_eukaryote_proxy:
    positives_by_organism = df.groupby("Organism (ID)")["target"].transform("sum")
    before = len(df)
    df = df[positives_by_organism > 0].reset_index(drop=True)
    print(f"removed organisms without positives: {before - len(df)}")

# drop columns that leak the target / remove colunas que vazam o alvo
df = df.drop(columns=[col for col in leak_cols if col in df.columns])

# class balance check / conferência do balanceamento das classes
print(f"target: {target_name} | strict: {strict_nucleus} | eukaryote proxy: {only_eukaryote_proxy}")
print(df["target"].value_counts())
print(f"positive rate: {df['target'].mean():.2%}")
print(f"final shape: {df.shape}")

# save the final parquet / salva o parquet final
df.to_parquet(output_parquet, index=False, compression="snappy")
print(f"file saved at: {output_parquet}")

# save the same dataframe as csv / salva o mesmo dataframe em csv
df.to_csv(output_csv, index=False, encoding="utf-8")
print(f"file saved at: {output_csv}")

# size comparison between parquet and csv / comparação de tamanho entre parquet e csv
parquet_mb = output_parquet.stat().st_size / 1024 / 1024
csv_mb = output_csv.stat().st_size / 1024 / 1024
print(f"parquet: {parquet_mb:.1f} MB | csv: {csv_mb:.1f} MB")

# column names without spaces or brackets, friendlier for sql
# nomes de colunas sem espaços ou colchetes, mais amigáveis para sql
sql_columns = {
    "Entry": "entry",
    "Entry Name": "entry_name",
    "Protein names": "protein_names",
    "Gene Names": "gene_names",
    "Organism": "organism",
    "Organism (ID)": "organism_id",
    "Length": "length",
    "Sequence": "sequence",
    "Subcellular location [CC]": "subcellular_location",
    "EC number": "ec_number",
    "Pfam": "pfam",
    "target": "target",
}
df_sql = df.rename(columns=sql_columns)

# write the table, replacing it if it already exists / grava a tabela, substituindo se já existir
with sqlite3.connect(db_file) as conn:
    df_sql.to_sql("uniprot_target", conn, if_exists="replace", index=False, chunksize=50_000)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_uniprot_entry ON uniprot_target (entry)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_uniprot_organism ON uniprot_target (organism_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_uniprot_target ON uniprot_target (target)")
    total = conn.execute("SELECT COUNT(*) FROM uniprot_target").fetchone()[0]

db_mb = db_file.stat().st_size / 1024 / 1024
print(f"table uniprot_target: {total} rows | database: {db_mb:.1f} MB")
print(f"file saved at: {db_file}")