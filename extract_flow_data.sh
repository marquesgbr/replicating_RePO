#!/bin/bash
# Extrai os CSVs de fluxo do CICIDS2017.
#
# Uso local (raiz do repositório):
#   bash extract_flow_data.sh
# Uso no cluster, gravando em $SCRATCH:
#   DATA_ROOT="$SCRATCH/replicating-repo-data/flow_based" bash extract_flow_data.sh

set -euo pipefail

ZIP_DIR="${ZIP_DIR:-data/flow_based}"
DATA_ROOT="${DATA_ROOT:-data/flow_based}"

mkdir -p "${DATA_ROOT}"

for day in Monday Tuesday Wednesday Thursday Friday; do
  unzip -o "${ZIP_DIR}/${day}-WH-generate-labeled.zip" -d "${DATA_ROOT}"
done
