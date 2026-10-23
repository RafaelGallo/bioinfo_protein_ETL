"""Extract a large protein dataset from the UniProt REST API.

Extrai um dataset grande de proteínas da API REST do UniProt usando paginação
por cursor, barra de progresso tqdm e gravação incremental do CSV na pasta
input do projeto.
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
output_file = input_dir / "uniprot_raw.csv"

# create the input folder if it does not exist / cria a pasta input caso não exista
input_dir.mkdir(parents=True, exist_ok=True)

# maximum number of rows to extract / número máximo de linhas a extrair
max_rows = 8000_000

# True = only reviewed Swiss-Prot, False = adds TrEMBL (millions)
# True = apenas Swiss-Prot revisadas, False = inclui TrEMBL (milhões)
only_reviewed = True

# query: all organisms, with subcellular location, sequence length between 50 and 1500
# consulta: todos os organismos, com localização subcelular, tamanho entre 50 e 1500
query = "(cc_scl_term:*) AND (length:[50 TO 1500])"
if only_reviewed:
    query = "(reviewed:true) AND " + query

# fields to extract / campos a extrair
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
        "cc_subcellular_location",
        "ec",
        "go_id",
        "xref_pfam",
    ]
)

# session with automatic retry for unstable connections / sessão com retry automático para conexões instáveis
session = requests.Session()
retry = Retry(
    total=6,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
session.mount("https://", HTTPAdapter(max_retries=retry))

# page size and first page request / tamanho da página e primeira requisição
page_size = 500
next_url = "https://rest.uniprot.org/uniprotkb/search"
params = {"query": query, "format": "tsv", "fields": fields, "size": page_size}

# remove an old output file so we do not append to it / remove arquivo antigo para não anexar a ele
if output_file.exists():
    output_file.unlink()

total_written = 0
first_page = True
progress = None

# cursor pagination loop / laço de paginação por cursor
while next_url and total_written < max_rows:
    response = session.get(next_url, params=params if first_page else None, timeout=300)
    response.raise_for_status()

    # create the progress bar using the total available in uniprot / cria a barra usando o total disponível no uniprot
    if progress is None:
        total_available = int(response.headers.get("x-total-results", max_rows))
        progress = tqdm(
            total=min(total_available, max_rows),
            desc="extracting uniprot",
            unit="rows",
            unit_scale=True,
        )

    # parse the page and append to csv / lê a página e anexa ao csv
    page = pd.read_csv(io.StringIO(response.text), sep="\t")
    if total_written + len(page) > max_rows:
        page = page.iloc[: max_rows - total_written]

    page.to_csv(
        output_file,
        mode="a",
        index=False,
        header=first_page,
        encoding="utf-8",
    )
    total_written += len(page)
    first_page = False

    # update the progress bar / atualiza a barra de progresso
    progress.update(len(page))

    # next page url comes in the link header / url da próxima página vem no header link
    next_url = response.links.get("next", {}).get("url")

# close the progress bar / fecha a barra de progresso
if progress is not None:
    progress.close()

print(f"done | rows written: {total_written}")
print(f"file saved at: {output_file}")