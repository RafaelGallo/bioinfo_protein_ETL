"""Run the UniProt ETL pipeline end to end.

Executa o pipeline ETL do UniProt de ponta a ponta: extração da API, conversão
para parquet, transformação (alvo, limpeza) e carga em parquet, csv e banco sql.
Os modelos de machine learning ficam fora deste pipeline.

Usage / Uso:
    python main.py                          runs all etl steps / roda todas as etapas do etl
    python main.py --steps create_target    only the listed steps / só as etapas listadas
    python main.py --skip extract extract_new    all steps except the listed / todas menos as listadas
    python main.py --list                   show the steps and exit / mostra as etapas e sai
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# project folders / pastas do projeto
base_dir = Path(r"C:\Users\rafae\Documents\Projetos_data_science_IA\ETL\uniprot")

# scripts folder: uses py if it exists, otherwise src / pasta dos scripts: usa py se existir, senão src
scripts_dir = base_dir / "py" if (base_dir / "py").exists() else base_dir / "src"

# etl steps in execution order: extract, transform and load
# etapas do etl em ordem de execução: extração, transformação e carga
pipeline = {
    "extract": "extract.py",
    "extract_new": "extract_new.py",
    "csv_to_parquet": "csv_to_parquet.py",
    "create_target": "create_target.py",
    "publish": "publish_kaggle.py",
}

# the publish step prepares files for public release, so it only runs when requested
# a etapa publish prepara arquivos para publicação, então só roda quando pedida
opt_in_steps = {"publish"}

# command line arguments / argumentos da linha de comando
parser = argparse.ArgumentParser(description="UniProt ETL pipeline")
parser.add_argument("--steps", nargs="+", choices=list(pipeline), help="steps to run / etapas a executar")
parser.add_argument("--skip", nargs="+", choices=list(pipeline), default=[], help="steps to skip / etapas a pular")
parser.add_argument("--list", action="store_true", help="list the steps and exit / lista as etapas e sai")
args = parser.parse_args()

# list the steps and exit / lista as etapas e sai
if args.list:
    print(f"scripts folder: {scripts_dir}")
    for step, script_name in pipeline.items():
        found = "found" if (scripts_dir / script_name).exists() else "missing"
        mode = "opt-in" if step in opt_in_steps else "default"
        print(f"  {step:15s} {script_name:20s} {mode:8s} {found}")
    sys.exit(0)

# decide which steps will run / define quais etapas vão rodar
if args.steps:
    selected = [step for step in pipeline if step in args.steps]
else:
    selected = [step for step in pipeline if step not in opt_in_steps]
selected = [step for step in selected if step not in args.skip]

print(f"python: {sys.executable}")
print(f"scripts folder: {scripts_dir}")
print(f"steps to run: {', '.join(selected)}")

# run each step as a separate process / roda cada etapa como um processo separado
results = {}
pipeline_start = time.time()

for step in selected:
    script = scripts_dir / pipeline[step]

    # skip steps whose script does not exist / pula etapas cujo script não existe
    if not script.exists():
        print(f"\nstep '{step}' skipped: {script.name} not found")
        results[step] = "skipped"
        continue

    print(f"\nstep '{step}' started: {script.name}")
    step_start = time.time()

    # stop the pipeline if a step fails / interrompe o pipeline se uma etapa falhar
    completed = subprocess.run([sys.executable, str(script)], cwd=base_dir)
    elapsed = time.time() - step_start

    if completed.returncode != 0:
        results[step] = "failed"
        print(f"step '{step}' failed with code {completed.returncode} after {elapsed:.1f}s")
        break

    results[step] = f"ok ({elapsed:.1f}s)"
    print(f"step '{step}' finished in {elapsed:.1f}s")

# final summary / resumo final
print("\npipeline summary:")
for step, status in results.items():
    print(f"  {step}: {status}")
print(f"total time: {time.time() - pipeline_start:.1f}s")

# non-zero exit code if any step failed / código de saída diferente de zero se alguma etapa falhou
if "failed" in results.values():
    sys.exit(1)