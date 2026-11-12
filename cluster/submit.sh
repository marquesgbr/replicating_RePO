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
#   bash cluster/submit.sh new-data normal
#   bash cluster/submit.sh new-data adversarial all primary
#   bash cluster/submit.sh new-data smoke ddos
#   bash cluster/submit.sh new-data train
#   bash cluster/submit.sh new-data export
#
# Variáveis úteis (ou use cluster/config.env):
#   PARTITION, CPUS_PER_TASK, MEM, GRES, CONCURRENCY,
#   WALLTIME_TRAIN, WALLTIME_NORMAL, WALLTIME_ADV,
#   WALLTIME_NEW_DATA_NORMAL, WALLTIME_NEW_DATA_ADV,
#   WALLTIME_NEW_DATA_TRAIN,
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
WALLTIME_NEW_DATA_NORMAL="${WALLTIME_NEW_DATA_NORMAL:-02:00:00}"
WALLTIME_NEW_DATA_ADV="${WALLTIME_NEW_DATA_ADV:-1-23:00:00}"
WALLTIME_NEW_DATA_TRAIN="${WALLTIME_NEW_DATA_TRAIN:-1-00:00:00}"
WALLTIME_MASK_BASELINE="${WALLTIME_MASK_BASELINE:-04:00:00}"
WALLTIME_MASK_VARIANCE="${WALLTIME_MASK_VARIANCE:-1-23:00:00}"
WALLTIME_MASK_ADV="${WALLTIME_MASK_ADV:-1-23:00:00}"
WALLTIME_MASK_CASCADE="${WALLTIME_MASK_CASCADE:-06:00:00}"
WALLTIME_MASK_DISTILL="${WALLTIME_MASK_DISTILL:-06:00:00}"
WALLTIME_MASK_PACKET="${WALLTIME_MASK_PACKET:-06:00:00}"
WALLTIME_MASK_AGGREGATE="${WALLTIME_MASK_AGGREGATE:-00:30:00}"
WALLTIME_MASK_MERGE="${WALLTIME_MASK_MERGE:-00:30:00}"

export_args() {
  local task="$1"
  local model="$2"
  local extra="$3"
  local exports="ALL,TASK=${task},MODEL=${model},RESULTS_ROOT=${RESULTS_ROOT},SEED=${SEED},MODEL_SOURCE=${MODEL_SOURCE:-pretrained},FORCE=${FORCE:-0},CLUSTER_DIR=${CLUSTER_DIR}"
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
  if [[ -n "${DEPENDENCY:-}" ]]; then
    extra_args+=(--dependency "${DEPENDENCY}")
  fi
  if [[ -n "${EXCLUDE:-}" ]]; then
    # --exclude não altera a contagem de nós (ao contrário de --nodelist).
    extra_args+=(--exclude "${EXCLUDE}")
  fi
  local out_pattern="${OUTPUT_PATTERN:-%x-%j.out}"
  local err_pattern="${out_pattern%.out}.err"
  sbatch --parsable \
    --job-name="${job_name}" \
    --chdir "${REPO_ROOT}" \
    --partition "${PARTITION}" \
    --gres "${GRES}" \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem "${MEM}" \
    --time "${walltime}" \
    --requeue \
    "${extra_args[@]}" \
    --export "${exports}" \
    --output "${RESULTS_ROOT}/logs/${out_pattern}" \
    --error "${RESULTS_ROOT}/logs/${err_pattern}" \
    "${CLUSTER_DIR}/slurm/job.sh"
}

submit_cpu_job() {
  local job_name="$1"
  local walltime="$2"
  local exports="$3"
  local extra_args=()
  shift 3
  while (($#)); do
    extra_args+=("$@")
    shift
  done
  if [[ -n "${DEPENDENCY:-}" ]]; then
    extra_args+=(--dependency "${DEPENDENCY}")
  fi
  if [[ -n "${EXCLUDE:-}" ]]; then
    extra_args+=(--exclude "${EXCLUDE}")
  fi
  local out_pattern="${OUTPUT_PATTERN:-%x-%j.out}"
  local err_pattern="${out_pattern%.out}.err"
  sbatch --parsable \
    --job-name="${job_name}" \
    --chdir "${REPO_ROOT}" \
    --partition "${PARTITION}" \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem "${MEM}" \
    --time "${walltime}" \
    --requeue \
    "${extra_args[@]}" \
    --export "${exports}" \
    --output "${RESULTS_ROOT}/logs/${out_pattern}" \
    --error "${RESULTS_ROOT}/logs/${err_pattern}" \
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
  # Uma única execução pontua as amostras uma vez e deriva RePO e RePO+ dos
  # mesmos erros por máscara. Dois jobs separados repetiriam todo o scoring.
  modes="${modes// /,}"
  modes="${modes//,,/,}"
  local mode
  IFS=',' read -r -a mode_list <<< "${modes}"
  for mode in "${mode_list[@]}"; do
    case "${mode}" in
      repo|repo_plus) ;;
      *) echo "[error] modo inválido: ${mode}" >&2; exit 2 ;;
    esac
  done
  # O --export do SLURM separa variáveis por vírgula; use '+' no ambiente.
  local joined="${modes//[[:space:]]/}"
  joined="${joined//,/\+}"
  local job_id
  job_id="$(submit_gpu_job "repo-normal-${model}" "${WALLTIME_NORMAL}" \
    "$(export_args normal "${model}" "SCORE_MODE=${joined}")")"
  echo "[submit] normal ${model} (${joined}): ${job_id}"
}

task_adversarial() {
  local model="$1"
  local attack="$2"
  local checkpoint_every="${3:-50}"
  local log_every="${4:-50}"

  local list_file
  list_file="$(mktemp "${RESULTS_ROOT}/attack-lists/${model}-XXXXXXXX.txt")"
  if [[ "${attack,,}" == "all" ]]; then
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
  if [[ "${attack,,}" != "all" ]]; then
    job_name+="-$(echo "${attack}" | tr ' /' '--' | tr -cd '[:alnum:]._-')"
  fi
  local extra="ATTACK_LIST_FILE=${list_file},CHECKPOINT_EVERY=${checkpoint_every},LOG_EVERY=${log_every}"
  if [[ -n "${MAX_SAMPLES:-}" ]]; then
    extra+=",MAX_SAMPLES=${MAX_SAMPLES}"
  fi
  if [[ "${PACKET_GPU_LOOP:-0}" == "1" && "${model}" == "packet" ]]; then
    extra+=",PACKET_GPU_LOOP=1"
  fi
  job_id="$(OUTPUT_PATTERN='%x-%A_%a.out' submit_gpu_job "${job_name}" "${WALLTIME_ADV}" \
    "$(export_args adversarial "${model}" "${extra}")" \
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

new_data_variant_settings() {
  local variant="$1"
  case "${variant}" in
    primary)
      printf 'random calibrated\n'
      ;;
    sensitivity_first_n)
      printf 'first calibrated\n'
      ;;
    sensitivity_paper_0.8)
      printf 'random paper_0.8\n'
      ;;
    smoke)
      printf 'random calibrated\n'
      ;;
    *)
      echo "[error] variante inválida: ${variant}" >&2
      return 2
      ;;
  esac
}

task_new_data_normal() {
  local raw_modes="${1:-${NEW_DATA_MODES:-repo repo_plus}}"
  raw_modes="${raw_modes//,/ }"
  raw_modes="${raw_modes//+/ }"
  local -a mode_list=()
  read -r -a mode_list <<< "${raw_modes}"
  local mode
  for mode in "${mode_list[@]}"; do
    case "${mode}" in
      repo|repo_plus) ;;
      *) echo "[error] modo inválido: ${mode}" >&2; exit 2 ;;
    esac
  done
  local joined=""
  for mode in "${mode_list[@]}"; do
    joined+="${joined:+,}${mode}"
  done
  joined="${joined//,/\+}"
  local extra="DATASET=new_data,NEW_DATA_RESULTS=${RESULTS_ROOT}/new_data,NEW_DATA_MODES=${joined},BATCH_SIZE=${BATCH_SIZE:-1024}"
  if [[ -n "${N_MASKS:-}" ]]; then
    extra+=",N_MASKS=${N_MASKS}"
  fi
  if [[ -n "${SCORE_REPEATS:-}" ]]; then
    extra+=",SCORE_REPEATS=${SCORE_REPEATS}"
  fi
  if [[ "${PAIRED_SCORING:-0}" == "1" ]]; then
    extra+=",PAIRED_SCORING=1"
  fi
  local job_id
  job_id="$(submit_gpu_job "repo-new-data-normal" "${WALLTIME_NEW_DATA_NORMAL}" \
    "$(export_args new_data_normal new_data "${extra}")")"
  echo "[submit] new-data normal (${joined}): ${job_id}"
}

task_new_data_adversarial() {
  local attack="$1"
  local variant="${2:-primary}"
  local settings
  settings="$(new_data_variant_settings "${variant}")" || exit 2
  local sample_mode policy
  read -r sample_mode policy <<< "${settings}"

  local list_dir="${RESULTS_ROOT}/new_data/attack-lists"
  mkdir -p "${list_dir}"
  local list_file
  list_file="$(mktemp "${list_dir}/attacks-XXXXXXXX.txt")"
  if [[ "${attack,,}" == "all" ]]; then
    PYTHONPATH="${REPO_ROOT}/code" "${VENV}/bin/python" - "${list_file}" <<'PY'
import sys
from pathlib import Path

from new_data_eval.data import ATTACK_LABELS

destination = Path(sys.argv[1])
destination.write_text("\n".join(ATTACK_LABELS) + "\n", encoding="utf-8")
print(f"[list] {len(ATTACK_LABELS)} ataques -> {destination}")
PY
  else
    printf '%s\n' "${attack}" > "${list_file}"
    echo "[list] 1 ataque -> ${list_file}"
  fi

  local count
  count="$(grep -cve '^[[:space:]]*$' "${list_file}")"
  local array_spec="0-$((count - 1))%${CONCURRENCY}"
  local extra="DATASET=new_data,NEW_DATA_RESULTS=${RESULTS_ROOT}/new_data,ATTACK_LIST_FILE=${list_file},VARIANT=${variant},SAMPLE_MODE=${sample_mode},THRESHOLD_POLICY=${policy},BATCH_SIZE=${BATCH_SIZE:-1024},CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-50}"
  if [[ -n "${MAX_SAMPLES:-}" ]]; then
    extra+=",MAX_SAMPLES=${MAX_SAMPLES}"
  fi
  if [[ -n "${N_MASKS:-}" ]]; then
    extra+=",N_MASKS=${N_MASKS}"
  fi
  if [[ -n "${SCORE_REPEATS:-}" ]]; then
    extra+=",SCORE_REPEATS=${SCORE_REPEATS}"
  fi
  if [[ "${PAIRED_SCORING:-0}" == "1" ]]; then
    extra+=",PAIRED_SCORING=1"
  fi

  local job_name="repo-new-data-adv-${variant}"
  if [[ "${attack,,}" != "all" ]]; then
    job_name+="-$(echo "${attack}" | tr ' /' '--' | tr -cd '[:alnum:]._-')"
  fi
  local job_id
  job_id="$(OUTPUT_PATTERN='%x-%A_%a.out' submit_gpu_job "${job_name}" "${WALLTIME_NEW_DATA_ADV}" \
    "$(export_args new_data_adversarial new_data "${extra}")" \
    --array="${array_spec}")"
  echo "[submit] new-data adversarial ${variant}: array ${array_spec} -> ${job_id}"
  echo "[list] ${list_file}"
}

task_new_data_smoke() {
  local attack="${1:-ddos}"
  local model_dir
  model_dir="$(new_data_model_dir_for "${MODEL_SOURCE:-pretrained}")"
  echo "O smoke do novo dataset deve ser interativo (partição debug, máx. 30 min):"
  echo
  echo "  srun --partition=debug --gres=${GRES} -c ${CPUS_PER_TASK} --mem=${MEM} --time=00:30:00 --pty bash"
  echo "  source ${VENV}/bin/activate"
  echo "  PYTHONPATH=${REPO_ROOT}/code python -m new_data_eval smoke \\"
  echo "      --data-dir ${REPO_ROOT}/data/new_data \\"
  echo "      --model-dir ${model_dir} \\"
  echo "      --meta-dir ${REPO_ROOT}/data/new_data/x_train_meta \\"
  echo "      --results-dir ${RESULTS_ROOT}/new_data \\"
  echo "      --attack ${attack}"
  echo
  echo "Alternativa não interativa (fila normal):"
  echo "  SMOKE=1 bash cluster/submit.sh new-data smoke ${attack}"
  if [[ "${SMOKE:-0}" == "1" ]]; then
    local job_id
    job_id="$(submit_gpu_job "repo-new-data-smoke-${attack}" "00:30:00" \
      "$(export_args new_data_smoke new_data "DATASET=new_data,NEW_DATA_RESULTS=${RESULTS_ROOT}/new_data,ATTACK=${attack},BATCH_SIZE=${BATCH_SIZE:-1024}")")"
    echo "[submit] new-data smoke ${attack}: ${job_id}"
  fi
}

task_new_data_train() {
  local model_dir="${MODEL_DIR:-${REPO_ROOT}/models/retrained/new_data_model}"
  if new_data_model_is_protected "${model_dir}"; then
    echo "[error] treino recusa sobrescrever o modelo dos autores: ${model_dir}" >&2
    echo "[hint] use MODEL_DIR=.../models/retrained/new_data_model" >&2
    exit 2
  fi
  local saved_model_dir="${MODEL_DIR:-}"
  MODEL_DIR="${model_dir}"
  local job_id
  job_id="$(submit_gpu_job "repo-new-data-train" "${WALLTIME_NEW_DATA_TRAIN}" \
    "$(export_args new_data_train new_data "DATASET=new_data,EPOCHS=${EPOCHS:-5},TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}")")"
  MODEL_DIR="${saved_model_dir}"
  echo "[submit] new-data train (${model_dir}): ${job_id}"
}

task_new_data_export() {
  local out_dir="${OUT_DIR:-${REPO_ROOT}/results/cluster/new_data}"
  "${VENV}/bin/python" "${CLUSTER_DIR}/aggregate_new_data.py" \
    --results-dir "${RESULTS_ROOT}/new_data" \
    --out "${out_dir}"
  echo "[ok] resumo do novo dataset em ${out_dir}"
}

task_new_data_export_job() {
  local out_dir="${NEW_DATA_EXPORT_DIR:-${RESULTS_ROOT}/new_data/export}"
  local job_id
  job_id="$(submit_cpu_job "repo-new-data-export" "00:30:00" \
    "$(export_args new_data_export new_data "DATASET=new_data,NEW_DATA_RESULTS=${RESULTS_ROOT}/new_data,NEW_DATA_EXPORT_DIR=${out_dir}")")"
  echo "[submit] new-data export job: ${job_id} -> ${out_dir}"
}

mask_attack_set() {
  local set="${1:-pilot}"
  PYTHONPATH="${REPO_ROOT}/code" "${VENV}/bin/python" - "${set}" <<'PY'
import sys

if sys.argv[1] == "all":
    from mask_experiments.data import FLOW_ATTACKS
    attacks = FLOW_ATTACKS
else:
    from mask_experiments.adversarial import DEFAULT_ATTACKS
    attacks = DEFAULT_ATTACKS
print("\n".join(attacks))
PY
}

task_mask_single() {
  local kind="$1"
  local walltime
  case "${kind}" in
    baseline) walltime="${WALLTIME_MASK_BASELINE}" ;;
    variance) walltime="${WALLTIME_MASK_VARIANCE}" ;;
    cascade) walltime="${WALLTIME_MASK_CASCADE}" ;;
    distill) walltime="${WALLTIME_MASK_DISTILL}" ;;
    packet) walltime="${WALLTIME_MASK_PACKET}" ;;
    *) echo "[error] kind inválido: ${kind}" >&2; exit 2 ;;
  esac
  local extra="MASK_RESULTS=${RESULTS_ROOT}/mask_efficiency"
  local job_id
  job_id="$(submit_gpu_job "repo-mask-${kind}" "${walltime}" \
    "$(export_args "mask_${kind}" flow "${extra}")")"
  echo "[submit] mask-exp ${kind}: ${job_id}"
}

task_mask_variance() {
  local policy="${1:-}"
  local walltime="${WALLTIME_MASK_VARIANCE}"
  local extra="MASK_RESULTS=${RESULTS_ROOT}/mask_efficiency"
  local job_name="repo-mask-variance"
  if [[ -n "${policy}" ]]; then
    extra+=",MASK_POLICIES=${policy},MASK_VARIANCE_TAG=${policy}"
    job_name+="-${policy}"
  fi
  local job_id
  job_id="$(submit_gpu_job "${job_name}" "${walltime}" \
    "$(export_args mask_variance flow "${extra}")")"
  echo "[submit] mask-exp variance: ${job_id}"
}

task_mask_merge() {
  local job_id
  job_id="$(submit_cpu_job "repo-mask-merge" "${WALLTIME_MASK_MERGE}" \
    "$(export_args mask_merge flow "MASK_RESULTS=${RESULTS_ROOT}/mask_efficiency")")"
  echo "[submit] mask-exp merge: ${job_id}"
}

task_mask_attack() {
  local kind="$1"
  local attack="$2"
  local walltime="${WALLTIME_MASK_ADV}"
  local task
  case "${kind}" in
    adversarial) task="mask_adversarial" ;;
    cascade-attack) task="mask_cascade_attack" ;;
    distill-attack) task="mask_distill_attack" ;;
    *) echo "[error] kind inválido: ${kind}" >&2; exit 2 ;;
  esac
  if [[ "${kind}" == "distill-attack" \
     && -z "${DEPENDENCY:-}" \
     && ! -f "${RESULTS_ROOT}/mask_efficiency/distillation_student.keras" ]]; then
    echo "[error] aluno ausente: ${RESULTS_ROOT}/mask_efficiency/distillation_student.keras" >&2
    echo "[hint] rode 'bash cluster/submit.sh mask-exp distill' primeiro" >&2
    exit 2
  fi
  local list_dir="${RESULTS_ROOT}/mask_efficiency/attack-lists"
  mkdir -p "${list_dir}"
  local list_file
  list_file="$(mktemp "${list_dir}/${kind}-XXXXXXXX.txt")"
  if [[ "${attack,,}" == "all" ]]; then
    mask_attack_set "${MASK_ATTACK_SET:-all}" > "${list_file}"
    echo "[list] $(grep -cve '^[[:space:]]*$' "${list_file}") ataques -> ${list_file}"
  else
    printf '%s\n' "${attack}" > "${list_file}"
    echo "[list] 1 ataque -> ${list_file}"
  fi
  local count
  count="$(grep -cve '^[[:space:]]*$' "${list_file}")"
  local array_spec="0-$((count - 1))%${CONCURRENCY}"
  local extra="MASK_RESULTS=${RESULTS_ROOT}/mask_efficiency"
  extra+=",ATTACK_LIST_FILE=${list_file}"
  if [[ -n "${MASK_PER_ATTACK:-}" ]]; then
    extra+=",MASK_PER_ATTACK=${MASK_PER_ATTACK}"
  fi
  if [[ -n "${MASK_VARIANTS:-}" ]]; then
    extra+=",MASK_VARIANTS=${MASK_VARIANTS//,/+}"
  fi
  if [[ -n "${MASK_CHEAP_ATTACK:-}" ]]; then
    extra+=",MASK_CHEAP_ATTACK=${MASK_CHEAP_ATTACK//,/+}"
  fi
  if [[ -n "${MASK_CASCADE_FN:-}" ]]; then
    extra+=",MASK_CASCADE_FN=${MASK_CASCADE_FN}"
  fi
  if [[ -n "${MASK_CASCADE_FPR:-}" ]]; then
    extra+=",MASK_CASCADE_FPR=${MASK_CASCADE_FPR}"
  fi
  local job_name="repo-mask-${kind}"
  if [[ "${attack,,}" != "all" ]]; then
    job_name+="-$(echo "${attack}" | tr ' /' '--' | tr -cd '[:alnum:]._-')"
  fi
  local job_id
  job_id="$(OUTPUT_PATTERN='%x-%A_%a.out' submit_gpu_job "${job_name}" "${walltime}" \
    "$(export_args "${task}" flow "${extra}")" \
    --array="${array_spec}")"
  echo "[submit] mask-exp ${kind}: array ${array_spec} -> ${job_id}"
  echo "[list] ${list_file}"
}

task_mask_aggregate() {
  local job_id
  job_id="$(submit_cpu_job "repo-mask-aggregate" "${WALLTIME_MASK_AGGREGATE}" \
    "$(export_args mask_aggregate flow "MASK_RESULTS=${RESULTS_ROOT}/mask_efficiency")")"
  echo "[submit] mask-exp aggregate: ${job_id}"
}

usage() {
  cat <<'EOF'
Uso: bash cluster/submit.sh <comando> [argumentos]

Comandos do artigo (fluxo/pacotes):
  train <flow|packet>
  normal <flow|packet> [repo repo_plus]
  adversarial <flow|packet> <ataque|all> [checkpoint_every] [log_every]
  smoke <flow|packet>
  export

Novo dataset:
  new-data normal [repo repo_plus]
  new-data adversarial <ataque|all> [primary|sensitivity_first_n|sensitivity_paper_0.8|smoke]
  new-data smoke [ataque]
  new-data train
  new-data export
  new-data export-job        # exportação encadeada por dependência SLURM

Máscaras (eficiência do RePO):
  mask-exp baseline
  mask-exp variance [politica]
  mask-exp adversarial <ataque|all>
  mask-exp cascade
  mask-exp cascade-attack <ataque|all>
  mask-exp distill
  mask-exp distill-attack <ataque|all>
  mask-exp packet
  mask-exp merge
  mask-exp aggregate

Variáveis: PARTITION, CPUS_PER_TASK, MEM, GRES, CONCURRENCY,
  WALLTIME_* (inclui WALLTIME_NEW_DATA_* e WALLTIME_MASK_*), MODEL_SOURCE,
  MODEL_DIR, SEED, FORCE=1, NODELIST, MAX_SAMPLES, N_MASKS, BATCH_SIZE,
  CHECKPOINT_EVERY, SMOKE=1, DEPENDENCY=afterok:<jobid>.
  Máscaras: MASK_MAX_BENIGN, MASK_MAX_PER_ATTACK, MASK_PER_ATTACK,
  MASK_VARIANTS, MASK_CHEAP_ATTACK, MASK_CASCADE_FN, MASK_CASCADE_FPR,
  MASK_ATTACK_SET=pilot|all.
EOF
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
  new-data)
    subcommand="${1:-}"
    shift || true
    case "${subcommand}" in
      normal) task_new_data_normal "${1:-}" ;;
      adversarial)
        [[ $# -ge 1 ]] || { usage; exit 2; }
        task_new_data_adversarial "$1" "${2:-primary}"
        ;;
      smoke) task_new_data_smoke "${1:-ddos}" ;;
      train) task_new_data_train ;;
      export) task_new_data_export ;;
      export-job) task_new_data_export_job ;;
      *) usage; exit 2 ;;
    esac
    ;;
  mask-exp)
    subcommand="${1:-}"
    shift || true
    case "${subcommand}" in
      baseline) task_mask_single baseline ;;
      variance) task_mask_variance "${1:-}" ;;
      cascade) task_mask_single cascade ;;
      distill) task_mask_single distill ;;
      packet) task_mask_single packet ;;
      merge) task_mask_merge ;;
      adversarial)
        [[ $# -ge 1 ]] || { usage; exit 2; }
        task_mask_attack adversarial "$1"
        ;;
      cascade-attack)
        [[ $# -ge 1 ]] || { usage; exit 2; }
        task_mask_attack cascade-attack "$1"
        ;;
      distill-attack)
        [[ $# -ge 1 ]] || { usage; exit 2; }
        task_mask_attack distill-attack "$1"
        ;;
      aggregate) task_mask_aggregate ;;
      *) usage; exit 2 ;;
    esac
    ;;
  status)
    squeue -u "${USER:-$(id -un)}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
