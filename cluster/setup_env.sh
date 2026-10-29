#!/usr/bin/env bash
# Prepara o ambiente do cluster Apuana (executar uma vez no nó de login).
#
# O que faz:
#   1. carrega um módulo Python (>= 3.10);
#   2. cria um venv em $VENV e instala cluster/requirements-cluster.txt;
#   3. extrai os datasets para $DATA_ROOT e cria symlinks em data/;
#   4. valida os modelos, os dados e o import do TensorFlow;
#   5. grava um pip freeze em $RESULTS_ROOT/env-lock.txt.
#
# Uso:
#   bash cluster/setup_env.sh
#   bash cluster/setup_env.sh --skip-data
#   bash cluster/setup_env.sh --force

set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${CLUSTER_DIR}/common.sh"

SKIP_DATA=0
SKIP_VENV=0
FORCE=0
FORCE_DATA=0
FORCE_VENV=0
while (($#)); do
  case "$1" in
    --skip-data) SKIP_DATA=1 ;;
    --skip-venv) SKIP_VENV=1 ;;
    --force) FORCE=1 ;;
    --force-data) FORCE_DATA=1 ;;
    --force-venv) FORCE_VENV=1 ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "[error] argumento desconhecido: $1" >&2
      exit 2
      ;;
  esac
  shift
done

load_python_module
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
require_python_310 "${PYTHON_BIN}"

echo "[setup] Python: $("${PYTHON_BIN}" --version 2>&1) ($(command -v "${PYTHON_BIN}"))"

# Avisa se o $HOME for usado como base e o espaço for limitado.
if [[ "${SCRATCH}" == "${HOME}" ]]; then
  echo "[warn] SCRATCH aponta para HOME (quota de 100 GB)." >&2
  echo "[warn] O dataset extraído usa ~13 GB e o venv com CUDA ~5 GB." >&2
fi
if command -v df >/dev/null 2>&1; then
  avail_kb="$(df -Pk "${SCRATCH}" | awk 'NR == 2 {print $4}')"
  if [[ "${avail_kb:-0}" =~ ^[0-9]+$ ]] && (( avail_kb < 25 * 1024 * 1024 )); then
    echo "[warn] menos de 25 GB livres em ${SCRATCH} (${avail_kb} KB)." >&2
  fi
fi

if (( ! SKIP_VENV )); then
  if [[ ! -f "${VENV}/bin/activate" ]] || (( FORCE || FORCE_VENV )); then
    if (( FORCE || FORCE_VENV )) && [[ -d "${VENV}" ]]; then
      echo "[setup] removendo venv anterior: ${VENV}"
      rm -rf "${VENV}"
    fi
    echo "[setup] criando venv em ${VENV}"
    "${PYTHON_BIN}" -m venv "${VENV}"
  else
    echo "[setup] reutilizando venv: ${VENV}"
  fi

  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  PY="${VENV}/bin/python"

  if ! "${PY}" -m pip --version >/dev/null 2>&1; then
    echo "[warn] pip ausente no venv; tentando ensurepip" >&2
    "${PY}" -m ensurepip --upgrade >/dev/null 2>&1 || true
  fi
  if ! "${PY}" -m pip --version >/dev/null 2>&1; then
    echo "[error] pip ausente no venv ${VENV}" >&2
    echo "[hint] carregue um módulo Python com pip e recrie o venv com --force" >&2
    exit 2
  fi

  echo "[setup] instalando dependências..."
  "${PY}" -m pip install --upgrade pip
  "${PY}" -m pip install -r "${CLUSTER_DIR}/requirements-cluster.txt"

  if ! python_runs; then
    echo "[error] TensorFlow/NumPy/Pandas não importam no venv" >&2
    exit 2
  fi
  "${PY}" - <<'PY'
import tensorflow as tf
import numpy as np
import pandas as pd

print(f"[setup] TensorFlow={tf.__version__} Keras={tf.keras.__version__}")
print(f"[setup] NumPy={np.__version__} Pandas={pd.__version__}")
PY
  "${PY}" -m pip freeze > "${RESULTS_ROOT}/env-lock.txt"
  echo "[setup] lock gravado em ${RESULTS_ROOT}/env-lock.txt"
fi

if (( ! SKIP_DATA )); then
  FLOW_DATA_DIR="${DATA_ROOT}/flow_based"
  PACKET_DATA_DIR="${DATA_ROOT}/packet_based"
  mkdir -p "${FLOW_DATA_DIR}" "${PACKET_DATA_DIR}"

  if (( FORCE || FORCE_DATA )) || [[ ! -s "${FLOW_DATA_DIR}/Monday-WH-generate-labeled.csv" ]]; then
    echo "[setup] extraindo fluxos para ${FLOW_DATA_DIR}"
    DATA_ROOT="${FLOW_DATA_DIR}" ZIP_DIR="${REPO_ROOT}/data/flow_based" \
      bash "${REPO_ROOT}/extract_flow_data.sh"
  else
    echo "[setup] fluxos já extraídos: ${FLOW_DATA_DIR}"
  fi

  if (( FORCE || FORCE_DATA )) || [[ ! -s "${PACKET_DATA_DIR}/tuesday/part_00000.npy" ]]; then
    echo "[setup] extraindo pacotes para ${PACKET_DATA_DIR} (pode demorar)"
    DATA_ROOT="${PACKET_DATA_DIR}" ZIP_DIR="${REPO_ROOT}/data/packet_based" \
      bash "${REPO_ROOT}/extract_packet_data.sh"
  else
    echo "[setup] pacotes já extraídos: ${PACKET_DATA_DIR}"
  fi

  # Os notebooks e o runner usam caminhos relativos a data/. Symlinks mantêm
  # a árvore do repositório limpa e apontam para o armazenamento rápido.
  for day in Monday Tuesday Wednesday Thursday Friday; do
    source_file="${FLOW_DATA_DIR}/${day}-WH-generate-labeled.csv"
    destination="${REPO_ROOT}/data/flow_based/${day}-WH-generate-labeled.csv"
    if [[ -L "${destination}" || ! -e "${destination}" ]]; then
      ln -sfn "${source_file}" "${destination}"
    else
      echo "[warn] mantendo arquivo existente: ${destination}" >&2
    fi
  done

  for day in monday tuesday wednesday thursday friday; do
    source_dir="${PACKET_DATA_DIR}/${day}"
    destination="${REPO_ROOT}/data/packet_based/${day}"
    if [[ -L "${destination}" || ! -e "${destination}" ]]; then
      ln -sfn "${source_dir}" "${destination}"
    else
      echo "[warn] mantendo diretório existente: ${destination}" >&2
    fi
  done
fi

# Preflight de dados e modelos pré-treinados.
require_flow_data
require_packet_data
require_pretrained_models flow pretrained
require_pretrained_models packet pretrained

echo "[setup] amostras de dados:"
du -sh "${DATA_ROOT}/flow_based" "${DATA_ROOT}/packet_based" 2>/dev/null || true

echo
echo "[ok] ambiente pronto."
echo "[ok] venv:    ${VENV}"
echo "[ok] dados:   ${DATA_ROOT}"
echo "[ok] results: ${RESULTS_ROOT}"
echo
echo "Próximos passos:"
echo "  1) teste interativo (partição debug):"
echo "     srun --partition=debug --gres=${GRES:-gpu:1} -c 8 --mem=32G --time=00:30:00 --pty bash"
echo "     source ${VENV}/bin/activate"
echo "     python cluster/repo_runner.py smoke --model flow"
echo "     python cluster/repo_runner.py smoke --model packet"
echo "  2) piloto e execução completa:"
echo "     bash cluster/submit.sh normal flow"
echo "     bash cluster/submit.sh adversarial flow \"DoS Slowhttptest\""
echo "     bash cluster/submit.sh adversarial packet all"
