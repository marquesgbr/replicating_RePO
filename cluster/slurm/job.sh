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

if [[ -n "${CLUSTER_DIR:-}" ]]; then
  SCRIPT_DIR="${CLUSTER_DIR}"
  COMMON_SH="${CLUSTER_DIR}/common.sh"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  COMMON_SH="${SCRIPT_DIR}/../common.sh"
fi
# shellcheck source=../common.sh
source "${COMMON_SH}"

load_python_module
activate_venv
cd "${REPO_ROOT}"

: "${TASK:?defina TASK=train|normal|adversarial|new_data_*}"

# Listas passadas por --export usam '+' no lugar de ',' (mesma convenção do
# SCORE_MODE dos jobs de fluxo/pacotes).
if [[ "${TASK}" == mask_* ]]; then
  MASK_VARIANTS="${MASK_VARIANTS:-}"
  MASK_VARIANTS="${MASK_VARIANTS//+/,}"
  MASK_CHEAP_ATTACK="${MASK_CHEAP_ATTACK:-}"
  MASK_CHEAP_ATTACK="${MASK_CHEAP_ATTACK//+/,}"
  export MASK_VARIANTS MASK_CHEAP_ATTACK
fi

print_runtime_env
if [[ -n "${MASK_STEP_NOTE:-}" ]]; then
  echo "[step] ${MASK_STEP_NOTE}"
fi

case "${TASK}" in
  new_data_*) MODEL="${MODEL:-new_data}" ;;
  *) MODEL="${MODEL:?defina MODEL=flow|packet}" ;;
esac
if [[ "${TASK}" == "train" || "${TASK}" == "new_data_train" ]]; then
  export MODEL_SOURCE="${MODEL_SOURCE:-retrained}"
else
  export MODEL_SOURCE="${MODEL_SOURCE:-pretrained}"
fi
export SEED="${SEED:-2019}"

MODEL_DIR_RESOLVED=""
NORM_DIR_RESOLVED=""
if [[ "${TASK}" != new_data_* ]]; then
  MODEL_DIR_RESOLVED="${MODEL_DIR:-$(model_dir_for "${MODEL}" "${MODEL_SOURCE}")}"
  NORM_DIR_RESOLVED="${NORMALIZATION_DIR:-$(normalization_dir_for "${MODEL}" "${MODEL_DIR_RESOLVED}")}"
fi

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

# The new dataset CLI lives in code/ and is imported as `new_data_eval`.
export PYTHONPATH="${REPO_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
NEW_DATA_DIR="${REPO_ROOT}/data/new_data"
NEW_DATA_META_DIR="${NEW_DATA_DIR}/x_train_meta"
NEW_DATA_RESULTS="${NEW_DATA_RESULTS:-$(new_data_results_root)}"

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
    MODES="${MODES//+/,}"
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
    if [[ "${PACKET_GPU_LOOP:-0}" == "1" && "${MODEL}" == "packet" ]]; then
      ADV_ARGS+=(--packet-gpu-loop)
    fi
    exec "${PY}" "${REPO_ROOT}/cluster/repo_runner.py" adversarial \
      "${COMMON_ARGS[@]}" \
      "${ADV_ARGS[@]}"
    ;;
  smoke)
    exec "${PY}" "${REPO_ROOT}/cluster/repo_runner.py" smoke "${COMMON_ARGS[@]}"
    ;;
  new_data_normal)
    NEW_DATA_MODEL_DIR="${MODEL_DIR:-$(new_data_model_dir_for "${MODEL_SOURCE}")}"
    NEW_DATA_MODES_RESOLVED="${NEW_DATA_MODES:-repo,repo_plus}"
    NEW_DATA_MODES_RESOLVED="${NEW_DATA_MODES_RESOLVED//+/,}"
    ND_ARGS=(
      --data-dir "${NEW_DATA_DIR}"
      --model-dir "${NEW_DATA_MODEL_DIR}"
      --meta-dir "${NEW_DATA_META_DIR}"
      --results-dir "${NEW_DATA_RESULTS}"
      --modes "${NEW_DATA_MODES_RESOLVED}"
      --seed "${SEED}"
      --batch-size "${BATCH_SIZE:-1024}"
    )
    if [[ -n "${N_MASKS:-}" ]]; then
      ND_ARGS+=(--n-masks "${N_MASKS}")
    fi
    if [[ -n "${SCORE_REPEATS:-}" ]]; then
      ND_ARGS+=(--score-repeats "${SCORE_REPEATS}")
    fi
    if [[ "${PAIRED_SCORING:-0}" == "1" ]]; then
      ND_ARGS+=(--paired-scoring)
    fi
    mkdir -p "${NEW_DATA_RESULTS}"
    exec "${PY}" -m new_data_eval normal "${ND_ARGS[@]}"
    ;;
  new_data_adversarial)
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

    if [[ ! -f "${NEW_DATA_RESULTS}/normal/repo_plus.json" ]]; then
      echo "[error] normal ausente: ${NEW_DATA_RESULTS}/normal/repo_plus.json" >&2
      echo "[hint] rode 'bash cluster/submit.sh new-data normal' primeiro" >&2
      exit 2
    fi

    NEW_DATA_MODEL_DIR="${MODEL_DIR:-$(new_data_model_dir_for "${MODEL_SOURCE}")}"
    ND_ADV=(
      --data-dir "${NEW_DATA_DIR}"
      --model-dir "${NEW_DATA_MODEL_DIR}"
      --meta-dir "${NEW_DATA_META_DIR}"
      --results-dir "${NEW_DATA_RESULTS}"
      --attack "${ATTACK}"
      --variant "${VARIANT:-primary}"
      --sample-mode "${SAMPLE_MODE:-random}"
      --attack-threshold-policy "${THRESHOLD_POLICY:-calibrated}"
      --seed "${SEED}"
      --batch-size "${BATCH_SIZE:-1024}"
      --checkpoint-every "${CHECKPOINT_EVERY:-50}"
    )
    if [[ -n "${MAX_SAMPLES:-}" ]]; then
      ND_ADV+=(--max-samples "${MAX_SAMPLES}")
    fi
    if [[ -n "${N_MASKS:-}" ]]; then
      ND_ADV+=(--n-masks "${N_MASKS}")
    fi
    if [[ -n "${SCORE_REPEATS:-}" ]]; then
      ND_ADV+=(--score-repeats "${SCORE_REPEATS}")
    fi
    if [[ "${PAIRED_SCORING:-0}" == "1" ]]; then
      ND_ADV+=(--paired-scoring)
    fi
    if [[ "${FORCE:-0}" == "1" ]]; then
      ND_ADV+=(--force)
    fi
    mkdir -p "${NEW_DATA_RESULTS}"
    exec "${PY}" -m new_data_eval adversarial "${ND_ADV[@]}"
    ;;
  new_data_smoke)
    NEW_DATA_MODEL_DIR="${MODEL_DIR:-$(new_data_model_dir_for "${MODEL_SOURCE}")}"
    ND_SMOKE=(
      --data-dir "${NEW_DATA_DIR}"
      --model-dir "${NEW_DATA_MODEL_DIR}"
      --meta-dir "${NEW_DATA_META_DIR}"
      --results-dir "${NEW_DATA_RESULTS}"
      --attack "${ATTACK:-ddos}"
      --seed "${SEED}"
      --batch-size "${BATCH_SIZE:-1024}"
    )
    if [[ "${FORCE:-0}" == "1" ]]; then
      ND_SMOKE+=(--force)
    fi
    exec "${PY}" -m new_data_eval smoke "${ND_SMOKE[@]}"
    ;;
  new_data_train)
    NEW_DATA_MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/models/retrained/new_data_model}"
    if new_data_model_is_protected "${NEW_DATA_MODEL_DIR}"; then
      echo "[error] recusando sobrescrever o modelo dos autores: ${NEW_DATA_MODEL_DIR}" >&2
      echo "[hint] use MODEL_DIR=.../models/retrained/new_data_model" >&2
      exit 2
    fi
    exec "${PY}" "${REPO_ROOT}/code/train_new_data_model.py" \
      --train-csv "${REPO_ROOT}/data/new_data/new_train.csv" \
      --output-dir "${NEW_DATA_MODEL_DIR}" \
      --seed "${SEED}" \
      --epochs "${EPOCHS:-5}" \
      --batch-size "${TRAIN_BATCH_SIZE:-256}"
    ;;
  new_data_export)
    exec "${PY}" "${REPO_ROOT}/cluster/aggregate_new_data.py" \
      --results-dir "${NEW_DATA_RESULTS}" \
      --out "${NEW_DATA_EXPORT_DIR:-${RESULTS_ROOT}/new_data/export}"
    ;;
  mask_baseline)
    MASK_RESULTS="${MASK_RESULTS:-${RESULTS_ROOT}/mask_efficiency}"
    MASK_MODEL_DIR="${MASK_MODEL_DIR:-${REPO_ROOT}/models/flw_model}"
    mkdir -p "${MASK_RESULTS}"
    MASK_ARGS=(
      --data-dir "${REPO_ROOT}/data"
      --model-dir "${MASK_MODEL_DIR}"
      --max-benign "${MASK_MAX_BENIGN:-100000}"
      --max-per-attack "${MASK_MAX_PER_ATTACK:-5000}"
      --n-masks "${MASK_N_MASKS:-100}"
      --batch-size "${MASK_BATCH_SIZE:-1024}"
      --seed "${SEED}"
      --out "${MASK_RESULTS}/baseline_flow.json"
    )
    if [[ -n "${MASK_CACHE:-}" ]]; then
      MASK_ARGS+=(--cache "${MASK_CACHE}")
    fi
    exec "${PY}" -m mask_experiments.runner "${MASK_ARGS[@]}"
    ;;
  mask_variance)
    MASK_RESULTS="${MASK_RESULTS:-${RESULTS_ROOT}/mask_efficiency}"
    MASK_MODEL_DIR="${MASK_MODEL_DIR:-${REPO_ROOT}/models/flw_model}"
    mkdir -p "${MASK_RESULTS}"
    # The curve is split by policy so each SLURM job fits the walltime; the
    # merge step joins the per-policy documents back into variance_flow.json.
    MASK_VARIANCE_SUFFIX=""
    if [[ -n "${MASK_VARIANCE_TAG:-}" ]]; then
      MASK_VARIANCE_SUFFIX="_${MASK_VARIANCE_TAG}"
    fi
    MASK_ARGS=(
      --data-dir "${REPO_ROOT}/data"
      --model-dir "${MASK_MODEL_DIR}"
      --max-benign "${MASK_MAX_BENIGN:-100000}"
      --max-per-attack "${MASK_MAX_PER_ATTACK:-5000}"
      --n-masks "${MASK_VARIANCE_N:-5,10,20,25,50,100}"
      --policies "${MASK_POLICIES:-iid,balanced,covering}"
      --repeats "${MASK_REPEATS:-3}"
      --batch-size "${MASK_BATCH_SIZE:-1024}"
      --seed "${SEED}"
      --out "${MASK_RESULTS}/variance_flow${MASK_VARIANCE_SUFFIX}.json"
      --csv "${MASK_RESULTS}/variance_flow${MASK_VARIANCE_SUFFIX}.csv"
    )
    if [[ -n "${MASK_CACHE:-}" ]]; then
      MASK_ARGS+=(--cache "${MASK_CACHE}")
    fi
    exec "${PY}" -m mask_experiments.variance "${MASK_ARGS[@]}"
    ;;
  mask_adversarial)
    MASK_RESULTS="${MASK_RESULTS:-${RESULTS_ROOT}/mask_efficiency}"
    MASK_MODEL_DIR="${MASK_MODEL_DIR:-${REPO_ROOT}/models/flw_model}"
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
    mkdir -p "${MASK_RESULTS}/adversarial"
    MASK_ARGS=(
      --data-dir "${REPO_ROOT}/data"
      --model-dir "${MASK_MODEL_DIR}"
      --max-benign "${MASK_MAX_BENIGN:-100000}"
      --max-per-attack "${MASK_MAX_PER_ATTACK:-5000}"
      --per-attack "${MASK_PER_ATTACK:-5000}"
      --attacks "${ATTACK}"
      --variants "${MASK_VARIANTS:-iid:100,iid:25,balanced:25,covering:25}"
      --batch-size "${MASK_BATCH_SIZE:-1024}"
      --seed "${SEED}"
      --out "${MASK_RESULTS}/adversarial/${ATTACK}.json"
    )
    if [[ -n "${MASK_CACHE:-}" ]]; then
      MASK_ARGS+=(--cache "${MASK_CACHE}")
    fi
    exec "${PY}" -m mask_experiments.adversarial "${MASK_ARGS[@]}"
    ;;
  mask_cascade)
    MASK_RESULTS="${MASK_RESULTS:-${RESULTS_ROOT}/mask_efficiency}"
    MASK_MODEL_DIR="${MASK_MODEL_DIR:-${REPO_ROOT}/models/flw_model}"
    mkdir -p "${MASK_RESULTS}"
    MASK_ARGS=(
      --data-dir "${REPO_ROOT}/data"
      --model-dir "${MASK_MODEL_DIR}"
      --max-benign "${MASK_MAX_BENIGN:-100000}"
      --max-per-attack "${MASK_MAX_PER_ATTACK:-5000}"
      --n-masks "${MASK_N_MASKS:-100}"
      --repeats "${MASK_REPEATS:-3}"
      --cheap "${MASK_CHEAP:-5,10,25}"
      --fn-rates "${MASK_FN_RATES:-0.0,0.001,0.01}"
      --batch-size "${MASK_BATCH_SIZE:-1024}"
      --seed "${SEED}"
      --out "${MASK_RESULTS}/cascade_flow.json"
    )
    if [[ -n "${MASK_CACHE:-}" ]]; then
      MASK_ARGS+=(--cache "${MASK_CACHE}")
    fi
    exec "${PY}" -m mask_experiments.cascade "${MASK_ARGS[@]}"
    ;;
  mask_cascade_attack)
    MASK_RESULTS="${MASK_RESULTS:-${RESULTS_ROOT}/mask_efficiency}"
    MASK_MODEL_DIR="${MASK_MODEL_DIR:-${REPO_ROOT}/models/flw_model}"
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
    mkdir -p "${MASK_RESULTS}/cascade_attack"
    MASK_ARGS=(
      --data-dir "${REPO_ROOT}/data"
      --model-dir "${MASK_MODEL_DIR}"
      --max-benign "${MASK_MAX_BENIGN:-100000}"
      --max-per-attack "${MASK_MAX_PER_ATTACK:-5000}"
      --per-attack "${MASK_PER_ATTACK:-5000}"
      --attacks "${ATTACK}"
      --cheap "${MASK_CHEAP_ATTACK:-5,25}"
      --fn-rate "${MASK_CASCADE_FN:-0.01}"
      --fpr "${MASK_CASCADE_FPR:-0.10}"
      --batch-size "${MASK_BATCH_SIZE:-1024}"
      --seed "${SEED}"
      --out "${MASK_RESULTS}/cascade_attack/${ATTACK}.json"
    )
    if [[ -n "${MASK_CACHE:-}" ]]; then
      MASK_ARGS+=(--cache "${MASK_CACHE}")
    fi
    exec "${PY}" -m mask_experiments.cascade_attack "${MASK_ARGS[@]}"
    ;;
  mask_distill)
    MASK_RESULTS="${MASK_RESULTS:-${RESULTS_ROOT}/mask_efficiency}"
    MASK_MODEL_DIR="${MASK_MODEL_DIR:-${REPO_ROOT}/models/flw_model}"
    mkdir -p "${MASK_RESULTS}"
    MASK_ARGS=(
      --data-dir "${REPO_ROOT}/data"
      --model-dir "${MASK_MODEL_DIR}"
      --max-benign "${MASK_MAX_BENIGN:-100000}"
      --max-per-attack "${MASK_MAX_PER_ATTACK:-5000}"
      --n-masks "${MASK_N_MASKS:-100}"
      --epochs "${MASK_EPOCHS:-40}"
      --batch-size "${MASK_BATCH_SIZE:-1024}"
      --seed "${SEED}"
      --out "${MASK_RESULTS}/distillation_flow.json"
      --student-out "${MASK_RESULTS}/distillation_student.keras"
    )
    if [[ -n "${MASK_CACHE:-}" ]]; then
      MASK_ARGS+=(--cache "${MASK_CACHE}")
    fi
    exec "${PY}" -m mask_experiments.distill "${MASK_ARGS[@]}"
    ;;
  mask_distill_attack)
    MASK_RESULTS="${MASK_RESULTS:-${RESULTS_ROOT}/mask_efficiency}"
    MASK_MODEL_DIR="${MASK_MODEL_DIR:-${REPO_ROOT}/models/flw_model}"
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
    if [[ ! -f "${MASK_RESULTS}/distillation_student.keras" ]]; then
      echo "[error] aluno ausente: ${MASK_RESULTS}/distillation_student.keras" >&2
      echo "[hint] rode 'bash cluster/submit.sh mask-exp distill' primeiro" >&2
      exit 2
    fi
    mkdir -p "${MASK_RESULTS}/distill_attack"
    MASK_ARGS=(
      --data-dir "${REPO_ROOT}/data"
      --model-dir "${MASK_MODEL_DIR}"
      --student "${MASK_RESULTS}/distillation_student.keras"
      --max-benign "${MASK_MAX_BENIGN:-100000}"
      --max-per-attack "${MASK_MAX_PER_ATTACK:-5000}"
      --per-attack "${MASK_PER_ATTACK:-5000}"
      --attacks "${ATTACK}"
      --batch-size "${MASK_BATCH_SIZE:-1024}"
      --seed "${SEED}"
      --out "${MASK_RESULTS}/distill_attack/${ATTACK}.json"
    )
    if [[ -n "${MASK_CACHE:-}" ]]; then
      MASK_ARGS+=(--cache "${MASK_CACHE}")
    fi
    exec "${PY}" -m mask_experiments.distill_attack "${MASK_ARGS[@]}"
    ;;
  mask_packet)
    MASK_RESULTS="${MASK_RESULTS:-${RESULTS_ROOT}/mask_efficiency}"
    mkdir -p "${MASK_RESULTS}"
    MASK_ARGS=(
      --data-dir "${REPO_ROOT}/data"
      --model-dir "${REPO_ROOT}/models/pkt_model"
      --windows "${MASK_PACKET_WINDOWS:-2000}"
      --variants "${MASK_PACKET_VARIANTS:-iid:100,balanced:25,covering:25}"
      --seed "${SEED}"
      --out "${MASK_RESULTS}/packet_normal.json"
    )
    exec "${PY}" -m mask_experiments.packet "${MASK_ARGS[@]}"
    ;;
  mask_merge)
    MASK_RESULTS="${MASK_RESULTS:-${RESULTS_ROOT}/mask_efficiency}"
    exec "${PY}" -m mask_experiments.merge --results-dir "${MASK_RESULTS}"
    ;;
  mask_aggregate)
    MASK_RESULTS="${MASK_RESULTS:-${RESULTS_ROOT}/mask_efficiency}"
    "${PY}" -m mask_experiments.merge --results-dir "${MASK_RESULTS}"
    exec "${PY}" -m mask_experiments.aggregate \
      --results-dir "${MASK_RESULTS}" \
      --out "${MASK_RESULTS}/MASK_EFFICIENCY_RESULTS.md"
    ;;
  *)
    echo "[error] TASK inválido: ${TASK}" >&2
    exit 2
    ;;
esac
