"""Extract unannotated proteins from the UniProt REST API.

Extrai proteínas sem localização subcelular anotada (TrEMBL) para servir como
conjunto de proteínas novas a serem previstas pelo modelo.
"""

import io
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

# project folders / pastas do projeto
base_dir = Path(r"C:\Users\rafae\Documents\Projetos_data_science_IA\ETL\uniprot")
input_dir = base_dir / "input"
output_file = input_dir / "uniprot_new.csv"
input_dir.mkdir(parents=True, exist_ok=True)

# maximum number of rows to extract / número máximo de linhas a extrair
max_rows = 100_000

# unreviewed proteins without subcellular location, same length range as training
# proteínas não revisadas sem localização subcelular, mesma faixa de tamanho do treino
query = "(reviewed:false) AND (NOT cc_scl_term:*) AND (length:[50 TO 1500])"

# fields must match the training extraction / campos devem ser os mesmos da extração de treino
fields = ",".join(
    [
        "accession",
        "id",
        "protein_name",
        "gene_names",
        "organism_name",
        "organism_id",
        "length",
        "sequence",
    ]
)

# session with automatic retry / sessão com retry automático
session = requests.Session()
retry = Retry(
    total=6,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
session.mount("https://", HTTPAdapter(max_retries=retry))

# first page request / requisição da primeira página
page_size = 500
next_url = "https://rest.uniprot.org/uniprotkb/search"
params = {"query": query, "format": "tsv", "fields": fields, "size": page_size}

# remove an old output file / remove arquivo antigo
if output_file.exists():
    output_file.unlink()

total_written = 0
first_page = True
progress = tqdm(total=max_rows, desc="extracting new proteins", unit="rows", unit_scale=True)

# cursor pagination loop / laço de paginação por cursor
while next_url and total_written < max_rows:
    response = session.get(next_url, params=params if first_page else None, timeout=300)
    response.raise_for_status()

    # parse the page and append to csv / lê a página e anexa ao csv
    page = pd.read_csv(io.StringIO(response.text), sep="\t")
    if total_written + len(page) > max_rows:
        page = page.iloc[: max_rows - total_written]

    page.to_csv(output_file, mode="a", index=False, header=first_page, encoding="utf-8")
    total_written += len(page)
    first_page = False
    progress.update(len(page))

    # next page url comes in the link header / url da próxima página vem no header link
    next_url = response.links.get("next", {}).get("url")

progress.close()
print(f"done | rows written: {total_written}")
print(f"file saved at: {output_file}")