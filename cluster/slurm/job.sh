#!/usr/bin/env bash
# Entrypoint SLURM de um job de treino, avaliação normal ou adversarial.
# Normalmente é submetido por cluster/submit.sh; não precisa ser chamado direto.
#SBATCH --job-name=repo
#SBATCH --partition=short-simple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-23:00:00
#SBATCH --requeue
#SBATCH --output=repo-%j.out
#SBATCH --error=repo-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

load_python_module
activate_venv
cd "${REPO_ROOT}"

: "${TASK:?defina TASK=train|normal|adversarial}"

print_runtime_env

MODEL="${MODEL:?defina MODEL=flow|packet}"
if [[ "${TASK}" == "train" ]]; then
  export MODEL_SOURCE="${MODEL_SOURCE:-retrained}"
else
  export MODEL_SOURCE="${MODEL_SOURCE:-pretrained}"
fi
export SEED="${SEED:-2019}"

MODEL_DIR_RESOLVED="${MODEL_DIR:-$(model_dir_for "${MODEL}" "${MODEL_SOURCE}")}"
NORM_DIR_RESOLVED="${NORMALIZATION_DIR:-$(normalization_dir_for "${MODEL}" "${MODEL_DIR_RESOLVED}")}"

COMMON_ARGS=(
  --model "${MODEL}"
  --model-source "${MODEL_SOURCE}"
  --model-dir "${MODEL_DIR_RESOLVED}"
  --normalization-dir "${NORM_DIR_RESOLVED}"
  --data-root "${REPO_ROOT}/data"
  --results-root "${RESULTS_ROOT}"
  --seed "${SEED}"
)

if [[ "${FORCE:-0}" == "1" ]]; then
  COMMON_ARGS+=(--force)
fi

case "${TASK}" in
  train)
    case "${MODEL_DIR_RESOLVED}" in
      "${REPO_ROOT}/models/flw_model"|"${REPO_ROOT}/models/pkt_model")
        echo "[error] recusando sobrescrever o modelo dos autores: ${MODEL_DIR_RESOLVED}" >&2
        echo "[hint] use MODEL_DIR=.../models/retrained/... ou MODEL_SOURCE=retrained" >&2
        exit 2
        ;;
    esac
    TRAIN_ARGS=()
    if [[ "${SMOKE:-0}" == "1" ]]; then
      TRAIN_ARGS+=(--smoke)
    fi
    exec "${PY}" "${REPO_ROOT}/cluster/train.py" \
      --model "${MODEL}" \
      --data-root "${REPO_ROOT}/data" \
      --results-root "${RESULTS_ROOT}" \
      --output-dir "${MODEL_DIR_RESOLVED}" \
      --seed "${SEED}" \
      "${TRAIN_ARGS[@]}"
    ;;
  normal)
    MODES="${SCORE_MODE:-repo,repo_plus}"
    exec "${PY}" "${REPO_ROOT}/cluster/repo_runner.py" normal \
      "${COMMON_ARGS[@]}" \
      --modes "${MODES}"
    ;;
  adversarial)
    ATTACK="${ATTACK:-}"
    if [[ -z "${ATTACK}" && -n "${ATTACK_LIST_FILE:-}" ]]; then
      if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        echo "[error] ATTACK_LIST_FILE exige um job array (SLURM_ARRAY_TASK_ID)" >&2
        exit 2
      fi
      ATTACK="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${ATTACK_LIST_FILE}")"
      ATTACK="${ATTACK%$'\r'}"
    fi
    if [[ -z "${ATTACK}" ]]; then
      echo "[error] defina ATTACK ou ATTACK_LIST_FILE" >&2
      exit 2
    fi
    ADV_ARGS=(
      --attack "${ATTACK}"
      --checkpoint-every "${CHECKPOINT_EVERY:-50}"
      --log-every "${LOG_EVERY:-50}"
    )
    if [[ -n "${MAX_SAMPLES:-}" ]]; then
      ADV_ARGS+=(--max-samples "${MAX_SAMPLES}")
    fi
    if [[ -n "${THRESHOLD:-}" ]]; then
      ADV_ARGS+=(--threshold "${THRESHOLD}")
    fi
    if [[ -n "${REAL_THRESHOLD:-}" ]]; then
      ADV_ARGS+=(--real-threshold "${REAL_THRESHOLD}")
    fi
    exec "${PY}" "${REPO_ROOT}/cluster/repo_runner.py" adversarial \
      "${COMMON_ARGS[@]}" \
      "${ADV_ARGS[@]}"
    ;;
  smoke)
    exec "${PY}" "${REPO_ROOT}/cluster/repo_runner.py" smoke "${COMMON_ARGS[@]}"
    ;;
  *)
    echo "[error] TASK inválido: ${TASK}" >&2
    exit 2
    ;;
esac
