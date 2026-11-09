#!/bin/bash
# Extrai os arquivos .npy de pacotes do CICIDS2017.
#
# Uso local (raiz do repositório):
#   bash extract_packet_data.sh
# Uso no cluster, gravando em $SCRATCH:
#   DATA_ROOT="$SCRATCH/replicating-repo-data/packet_based" bash extract_packet_data.sh

set -euo pipefail

ZIP_DIR="${ZIP_DIR:-data/packet_based}"
DATA_ROOT="${DATA_ROOT:-data/packet_based}"

mkdir -p "${DATA_ROOT}"

# Os arquivos de segunda, terça e quarta estão divididos em partes.
cat "${ZIP_DIR}"/monday.zip_parta* > "${DATA_ROOT}/monday.zip"
cat "${ZIP_DIR}"/tuesday.zip_parta* > "${DATA_ROOT}/tuesday.zip"
cat "${ZIP_DIR}"/wednesday.zip_parta* > "${DATA_ROOT}/wednesday.zip"

for day in monday tuesday wednesday; do
  unzip -o "${DATA_ROOT}/${day}.zip" -d "${DATA_ROOT}"
done

for day in thursday friday; do
  unzip -o "${ZIP_DIR}/${day}.zip" -d "${DATA_ROOT}"
done
