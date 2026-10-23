"""Convert the extracted UniProt CSV files to Parquet.

Converte os CSVs extraídos do UniProt para Parquet, lendo em blocos para não
estourar a memória, e salva o resultado na pasta data do projeto.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
from tqdm.auto import tqdm

# project folders / pastas do projeto
base_dir = Path(__file__).resolve().parent.parent
input_dir = base_dir / "input"
data_dir = base_dir / "data"

# create the data folder if it does not exist / cria a pasta data caso não exista
data_dir.mkdir(parents=True, exist_ok=True)

# csv files to convert / arquivos csv a converter
csv_files = [
    input_dir / "uniprot_new.csv",
    input_dir / "uniprot_raw.csv",
]

# read block size in bytes (64 MB) / tamanho do bloco de leitura em bytes (64 MB)
block_size = 64 * 1024 * 1024

for csv_file in csv_files:
    # skip files that were not extracted yet / pula arquivos que ainda não foram extraídos
    if not csv_file.exists():
        print(f"skipped, file not found: {csv_file.name}")
        continue

    parquet_file = data_dir / f"{csv_file.stem}.parquet"
    if parquet_file.exists():
        parquet_file.unlink()

    # read all columns as string to avoid type conflicts between blocks
    # lê todas as colunas como texto para evitar conflito de tipos entre blocos
    reader = pv.open_csv(
        csv_file,
        read_options=pv.ReadOptions(block_size=block_size),
        convert_options=pv.ConvertOptions(
            strings_can_be_null=True,
            column_types=None,
        ),
    )

    writer = None
    total_rows = 0
    progress = tqdm(desc=f"converting {csv_file.name}", unit="rows", unit_scale=True)

    # write each block as a parquet row group / grava cada bloco como um row group do parquet
    for batch in reader:
        table = pa.Table.from_batches([batch])
        if writer is None:
            writer = pq.ParquetWriter(parquet_file, table.schema, compression="snappy")
        writer.write_table(table)
        total_rows += table.num_rows
        progress.update(table.num_rows)

    progress.close()
    if writer is not None:
        writer.close()

    # size comparison between csv and parquet / comparação de tamanho entre csv e parquet
    csv_mb = csv_file.stat().st_size / 1024 / 1024
    parquet_mb = parquet_file.stat().st_size / 1024 / 1024
    print(f"rows: {total_rows} | csv: {csv_mb:.1f} MB | parquet: {parquet_mb:.1f} MB")
    print(f"file saved at: {parquet_file}")