#!/usr/bin/env bash
# Submete jobs da replicação ao SLURM do Apuana.
#
# Exemplos:
#   bash cluster/submit.sh train flow
#   bash cluster/submit.sh train packet
#   bash cluster/submit.sh normal flow
#   bash cluster/submit.sh adversarial flow "DoS Slowhttptest"
#   bash cluster/submit.sh adversarial flow all
#   bash cluster/submit.sh adversarial packet all
#   bash cluster/submit.sh smoke flow
#   bash cluster/submit.sh export
#
# Variáveis úteis (ou use cluster/config.env):
#   PARTITION, CPUS_PER_TASK, MEM, GRES, CONCURRENCY,
#   WALLTIME_TRAIN, WALLTIME_NORMAL, WALLTIME_ADV,
#   MODEL_SOURCE=pretrained|retrained, MODEL_DIR, SEED, FORCE=1, NODELIST.

set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${CLUSTER_DIR}/common.sh"

if [[ -f "${CLUSTER_DIR}/config.env" ]]; then
  # shellcheck disable=SC1091
  source "${CLUSTER_DIR}/config.env"
fi

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[error] comando não encontrado: $1" >&2
    exit 2
  fi
}

require_venv() {
  if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "[error] venv não encontrado: ${VENV}" >&2
    echo "[hint] rode bash cluster/setup_env.sh no nó de login" >&2
    exit 2
  fi
}

require_command sbatch
require_venv

PARTITION="${PARTITION:-short-simple}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-32G}"
GRES="${GRES:-gpu:1}"
CONCURRENCY="${CONCURRENCY:-2}"
WALLTIME_TRAIN="${WALLTIME_TRAIN:-1-00:00:00}"
WALLTIME_NORMAL="${WALLTIME_NORMAL:-02:00:00}"
WALLTIME_ADV="${WALLTIME_ADV:-1-23:00:00}"

export_args() {
  local task="$1"
  local model="$2"
  local extra="$3"
  local exports="ALL,TASK=${task},MODEL=${model},RESULTS_ROOT=${RESULTS_ROOT},SEED=${SEED},MODEL_SOURCE=${MODEL_SOURCE:-pretrained},FORCE=${FORCE:-0}"
  if [[ -n "${MODEL_DIR:-}" ]]; then
    exports+=",MODEL_DIR=${MODEL_DIR}"
  fi
  if [[ -n "${NORMALIZATION_DIR:-}" ]]; then
    exports+=",NORMALIZATION_DIR=${NORMALIZATION_DIR}"
  fi
  if [[ -n "${extra}" ]]; then
    exports+=",${extra}"
  fi
  printf '%s\n' "${exports}"
}

submit_gpu_job() {
  local job_name="$1"
  local walltime="$2"
  local exports="$3"
  local extra_args=()
  shift 3 || true
  while (($#)); do
    extra_args+=("$@")
    shift
  done
  sbatch --parsable \
    --job-name="${job_name}" \
    --partition "${PARTITION}" \
    --gres "${GRES}" \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem "${MEM}" \
    --time "${walltime}" \
    --requeue \
    "${extra_args[@]}" \
    --export "${exports}" \
    --output "${RESULTS_ROOT}/logs/%x-%j.out" \
    --error "${RESULTS_ROOT}/logs/%x-%j.err" \
    "${CLUSTER_DIR}/slurm/job.sh"
}

task_train() {
  local model="$1"
  MODEL_SOURCE="${MODEL_SOURCE:-retrained}"
  local model_dir="${MODEL_DIR:-$(model_dir_for "${model}" "retrained")}"
  if [[ "${model_dir}" == "${REPO_ROOT}/models/flw_model" || "${model_dir}" == "${REPO_ROOT}/models/pkt_model" ]]; then
    echo "[error] treino recusa sobrescrever o modelo dos autores: ${model_dir}" >&2
    exit 2
  fi
  local job_id
  job_id="$(submit_gpu_job "repo-train-${model}" "${WALLTIME_TRAIN}" \
    "$(export_args train "${model}" "MODEL_DIR=${model_dir},SMOKE=${SMOKE:-0}")")"
  echo "[submit] train ${model}: ${job_id} (model_dir=${model_dir})"
}

task_normal() {
  local model="$1"
  local modes="${2:-repo repo_plus}"
  modes="${modes//,/ }"
  local mode
  for mode in ${modes}; do
    case "${mode}" in
      repo|repo_plus) ;;
      *) echo "[error] modo inválido: ${mode}" >&2; exit 2 ;;
    esac
    local job_id
    job_id="$(submit_gpu_job "repo-normal-${model}-${mode}" "${WALLTIME_NORMAL}" \
      "$(export_args normal "${model}" "SCORE_MODE=${mode}")")"
    echo "[submit] normal ${model}/${mode}: ${job_id}"
  done
}

task_adversarial() {
  local model="$1"
  local attack="$2"
  local checkpoint_every="${3:-50}"
  local log_every="${4:-50}"

  local list_file
  list_file="$(mktemp "${RESULTS_ROOT}/attack-lists/${model}-XXXXXXXX.txt")"
  if [[ "${attack}" == "all" ]]; then
    "${VENV}/bin/python" - "${model}" "${list_file}" "${CLUSTER_DIR}" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[3])
import metrics

model, destination = sys.argv[1], Path(sys.argv[2])
destination.write_text("\n".join(metrics.expected_attacks(model)) + "\n", encoding="utf-8")
print(f"[list] {len(metrics.expected_attacks(model))} ataques -> {destination}")
PY
  else
    printf '%s\n' "${attack}" > "${list_file}"
    echo "[list] 1 ataque -> ${list_file}"
  fi

  local count
  count="$(grep -cve '^[[:space:]]*$' "${list_file}")"
  local array_spec="0-$((count - 1))%${CONCURRENCY}"
  local job_id
  local job_name="repo-adv-${model}"
  if [[ "${attack}" != "all" ]]; then
    job_name+="-$(echo "${attack}" | tr ' /' '--' | tr -cd '[:alnum:]._-')"
  fi
  job_id="$(submit_gpu_job "${job_name}" "${WALLTIME_ADV}" \
    "$(export_args adversarial "${model}" "ATTACK_LIST_FILE=${list_file},CHECKPOINT_EVERY=${checkpoint_every},LOG_EVERY=${log_every}")" \
    --array="${array_spec}")"
  echo "[submit] adversarial ${model}: array ${array_spec} -> ${job_id}"
  echo "[list] ${list_file}"
}

task_smoke() {
  local model="$1"
  echo "O teste rápido deve ser interativo (partição debug, máx. 30 min):"
  echo
  echo "  srun --partition=debug --gres=${GRES} -c ${CPUS_PER_TASK} --mem=${MEM} --time=00:30:00 --pty bash"
  echo "  source ${VENV}/bin/activate"
  echo "  python cluster/repo_runner.py smoke --model ${model} \\"
  echo "      --model-dir $(model_dir_for "${model}" "${MODEL_SOURCE:-pretrained}") \\"
  echo "      --results-root ${RESULTS_ROOT}/smoke"
  echo
  echo "Alternativa não interativa (fila normal):"
  echo "  SMOKE=1 bash cluster/submit.sh smoke ${model}"
  if [[ "${SMOKE:-0}" == "1" ]]; then
    local job_id
    job_id="$(submit_gpu_job "repo-smoke-${model}" "00:30:00" \
      "$(export_args smoke "${model}" "SMOKE=1")")"
    echo "[submit] smoke ${model}: ${job_id}"
  fi
}

task_export() {
  local out_dir="${OUT_DIR:-${REPO_ROOT}/results/cluster}"
  "${VENV}/bin/python" "${CLUSTER_DIR}/aggregate_results.py" \
    --results-root "${RESULTS_ROOT}" \
    --out "${out_dir}"
  echo "[ok] resumo em ${out_dir}"
  echo "[info] para arquivar os resultados brutos:"
  echo "  tar cJf \"\${ARCHIVE:-\$HOME}/replicating-repo-results-$(date -u +%Y%m%d).tar.xz\" -C \"${RESULTS_ROOT}\" ."
}

usage() {
  sed -n '2,25p' "$0"
}

command="${1:-}"
shift || true
case "${command}" in
  train)
    [[ $# -ge 1 ]] || { usage; exit 2; }
    task_train "$1"
    ;;
  normal)
    [[ $# -ge 1 ]] || { usage; exit 2; }
    task_normal "$1" "${2:-repo repo_plus}"
    ;;
  adversarial)
    [[ $# -ge 2 ]] || { usage; exit 2; }
    task_adversarial "$1" "$2" "${3:-50}" "${4:-50}"
    ;;
  smoke)
    [[ $# -ge 1 ]] || { usage; exit 2; }
    task_smoke "$1"
    ;;
  export)
    task_export
    ;;
  status)
    squeue -u "${USER:-$(id -un)}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
