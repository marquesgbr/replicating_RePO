#!/usr/bin/env bash
# Funções compartilhadas pelos scripts do cluster Apuana.
# Este arquivo é carregado com `source`; não o execute diretamente.

set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CLUSTER_DIR}/.." && pwd)"

SCRATCH="${SCRATCH:-$HOME}"
if [[ -z "${SCRATCH}" ]]; then
  echo "[error] SCRATCH vazio e HOME não definido" >&2
  exit 2
fi
if [[ ! -d "${SCRATCH}" || ! -w "${SCRATCH}" ]]; then
  echo "[error] base não gravável: SCRATCH=${SCRATCH}" >&2
  echo "[hint] export SCRATCH=/caminho/gravável (ex.: \$HOME)" >&2
  exit 2
fi

VENV="${VENV:-${SCRATCH}/replicating-repo-venv}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/replicating-repo-data}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRATCH}/replicating-repo-results}"
SEED="${SEED:-2019}"
MODEL_SOURCE="${MODEL_SOURCE:-pretrained}"

export SCRATCH VENV DATA_ROOT RESULTS_ROOT SEED MODEL_SOURCE
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

load_python_module() {
  if ! command -v module >/dev/null 2>&1; then
    echo "[warn] comando module ausente; usando ${PYTHON_BIN:-python3} do PATH" >&2
    return 0
  fi

  if [[ -n "${PYTHON_MODULE:-}" ]]; then
    if module load "${PYTHON_MODULE}" >/dev/null 2>&1; then
      echo "[env] module ${PYTHON_MODULE} carregado"
      return 0
    fi
    echo "[error] não foi possível carregar PYTHON_MODULE=${PYTHON_MODULE}" >&2
    echo "[hint] veja os módulos: module avail 2>&1 | grep -i python" >&2
    return 2
  fi

  local candidates=(Python/3.13 Python/3.12 Python/3.11 Python/3.10 Python3.13 Python3.12 Python3.11 Python3.10 python/3.13 python/3.12 python/3.11 python/3.10 python3.13 python3.12 python3.11 python3.10 miniforge3/2025.07 miniforge3)
  local discovered
  discovered="$(module avail 2>&1 | tr ' ()' '\n' \
    | grep -iE '^python3?/3\.[0-9]+' | sort -V | tail -n1 || true)"
  if [[ -n "${discovered}" ]]; then
    candidates+=("${discovered}")
  fi

  local mod
  for mod in "${candidates[@]}"; do
    if module load "${mod}" >/dev/null 2>&1; then
      echo "[env] module ${mod} carregado"
      return 0
    fi
  done

  echo "[warn] nenhum módulo Python carregado; usando ${PYTHON_BIN:-python3}" >&2
  echo "[hint] módulos disponíveis: module avail 2>&1 | grep -i python" >&2
  return 0
}

require_python_310() {
  local python_bin="${1:-python3}"
  if ! "${python_bin}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "[error] Python >= 3.10 é necessário; encontrado: $("${python_bin}" --version 2>&1)" >&2
    exit 2
  fi
}

activate_venv() {
  if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "[error] venv não encontrado: ${VENV}" >&2
    echo "[hint] rode bash cluster/setup_env.sh no nó de login" >&2
    exit 2
  fi
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  PY="${VENV}/bin/python"
  export PY
}

python_runs() {
  "${PY:-python3}" -c "import tensorflow, numpy, pandas" >/dev/null 2>&1
}

model_dir_for() {
  local model="$1"
  local source="${2:-${MODEL_SOURCE}}"
  local suffix
  case "${model}" in
    flow) suffix="flw_model" ;;
    packet) suffix="pkt_model" ;;
    *) echo "[error] modelo inválido: ${model}" >&2; return 2 ;;
  esac
  if [[ -n "${MODEL_DIR:-}" ]]; then
    printf '%s\n' "${MODEL_DIR}"
    return 0
  fi
  if [[ "${source}" == "retrained" ]]; then
    printf '%s\n' "${REPO_ROOT}/models/retrained/${suffix}"
  else
    printf '%s\n' "${REPO_ROOT}/models/${suffix}"
  fi
}

normalization_dir_for() {
  local model="$1"
  local model_dir="$2"
  if [[ -f "${model_dir}/x_train_meta/train_min.npy" ]]; then
    printf '%s\n' "${model_dir}/x_train_meta"
    return 0
  fi
  case "${model}" in
    flow) printf '%s\n' "${REPO_ROOT}/data/flow_based/x_train_meta" ;;
    packet) printf '%s\n' "${REPO_ROOT}/data/packet_based/x_train_meta" ;;
    *) return 2 ;;
  esac
}

require_pretrained_models() {
  local model_dir
  model_dir="$(model_dir_for "$1" "${2:-pretrained}")"
  if [[ ! -f "${model_dir}/saved_model.pb" ]]; then
    echo "[error] SavedModel ausente: ${model_dir}/saved_model.pb" >&2
    echo "[hint] use MODEL_SOURCE=pretrained (padrão) ou treine com submit.sh train" >&2
    exit 2
  fi
}

require_flow_data() {
  local missing=0
  local day
  for day in Tuesday Wednesday Thursday Friday Monday; do
    if [[ ! -f "${REPO_ROOT}/data/flow_based/${day}-WH-generate-labeled.csv" ]]; then
      echo "[error] CSV ausente: data/flow_based/${day}-WH-generate-labeled.csv" >&2
      missing=1
    fi
  done
  if (( missing )); then
    echo "[hint] rode bash cluster/setup_env.sh para extrair os dados" >&2
    exit 2
  fi
}

require_packet_data() {
  local missing=0
  local day
  for day in monday tuesday wednesday thursday friday; do
    if [[ ! -d "${REPO_ROOT}/data/packet_based/${day}" ]]; then
      echo "[error] diretório ausente: data/packet_based/${day}" >&2
      missing=1
    fi
  done
  if (( missing )); then
    echo "[hint] rode bash cluster/setup_env.sh para extrair os dados" >&2
    exit 2
  fi
}

print_runtime_env() {
  echo "[env] repo=${REPO_ROOT}"
  echo "[env] host=$(hostname) user=${USER:-$(id -un)}"
  echo "[env] scratch=${SCRATCH} venv=${VENV}"
  echo "[env] data=${DATA_ROOT} results=${RESULTS_ROOT}"
  echo "[env] model_source=${MODEL_SOURCE} seed=${SEED}"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[job] id=${SLURM_JOB_ID} name=${SLURM_JOB_NAME:-?} partition=${SLURM_JOB_PARTITION:-?} array=${SLURM_ARRAY_TASK_ID:-none}"
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | sed 's/^/[gpu] /' || true
  else
    echo "[gpu] nvidia-smi não encontrado"
  fi
}

mkdir -p "${RESULTS_ROOT}/logs" "${RESULTS_ROOT}/manifests" "${RESULTS_ROOT}/attack-lists"
