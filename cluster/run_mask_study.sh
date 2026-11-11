#!/usr/bin/env bash
# Submete a matriz completa do estudo de máscaras ao SLURM do Apuana.
#
# Uso (no no de login do Apuana, dentro do repositorio):
#   bash cluster/run_mask_study.sh              # matriz completa
#   bash cluster/run_mask_study.sh --smoke      # smoke de todas as etapas
#   bash cluster/run_mask_study.sh --dry-run    # so imprime o plano
#   bash cluster/run_mask_study.sh --no-aggregate
#
# A cadeia usa dependencias `afterok`: cada etapa so comeca se a anterior
# terminou com sucesso.  O ultimo passo faz o merge dos arrays e gera o
# Markdown consolidado.  O manifesto dos jobs fica em
#   $RESULTS_ROOT/mask_efficiency/chain_manifest.tsv
#
# Cada job recebe MASK_STEP_NOTE, que o cluster/slurm/job.sh imprime no log
# como "[step] <descricao>"; assim o .out de cada job diz o que ele esta
# fazendo.

set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${CLUSTER_DIR}/common.sh"

if [[ -f "${CLUSTER_DIR}/config.env" ]]; then
  # shellcheck disable=SC1091
  source "${CLUSTER_DIR}/config.env"
fi

MODE="full"
DRY_RUN=0
SKIP_AGGREGATE=0

usage() {
  cat <<'EOF'
Uso: bash cluster/run_mask_study.sh [opcoes]

  (sem opcoes)      matriz de producao completa
  --smoke           escala reduzida para validar a cadeia inteira
  --dry-run         imprime os comandos e o manifesto sem submeter
  --no-aggregate    submete tudo menos o passo de agregacao
  -h, --help        esta ajuda
EOF
}

while (($#)); do
  case "$1" in
    --smoke) MODE="smoke" ;;
    --dry-run) DRY_RUN=1 ;;
    --no-aggregate) SKIP_AGGREGATE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] opcao desconhecida: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

if [[ "${DRY_RUN}" -eq 0 ]]; then
  command -v sbatch >/dev/null 2>&1 || {
    echo "[error] sbatch nao encontrado; rode este script no no de login do Apuana" >&2
    exit 2
  }
fi

# ---------------------------------------------------------------------------
# Parametros de cada modo.  Tudo e exportado para o `sbatch --export ALL` do
# submit.sh; o job.sh decodifica as listas com '+'.
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "smoke" ]]; then
  export MASK_MAX_BENIGN=2000
  export MASK_MAX_PER_ATTACK=200
  export MASK_PER_ATTACK=20
  export MASK_ATTACK_SET=pilot
  export MASK_VARIANCE_N="5,25"
  export MASK_POLICIES="iid,balanced,covering"
  export MASK_REPEATS=1
  export MASK_VARIANTS="iid:100,iid:25,balanced:25,covering:25"
  export MASK_CHEAP="5,25"
  export MASK_CHEAP_ATTACK="5,25"
  export MASK_FN_RATES="0.0,0.01"
  export MASK_CASCADE_FN=0.01
  export MASK_CASCADE_FPR=0.10
  export MASK_EPOCHS=2
  export MASK_PACKET_WINDOWS=300
else
  export MASK_MAX_BENIGN=100000
  export MASK_MAX_PER_ATTACK=5000
  export MASK_PER_ATTACK=5000
  export MASK_ATTACK_SET=all
  export MASK_VARIANCE_N="5,10,20,25,50,100"
  export MASK_POLICIES="iid,balanced,covering"
  export MASK_REPEATS=5
  export MASK_VARIANTS="iid:100,iid:50,iid:25,iid:10,iid:5,balanced:50,balanced:25,balanced:10,covering:25"
  export MASK_CHEAP="5,10,25"
  export MASK_CHEAP_ATTACK="5,10,25"
  export MASK_FN_RATES="0.0,0.001,0.01"
  export MASK_CASCADE_FN=0.01
  export MASK_CASCADE_FPR=0.10
  export MASK_EPOCHS=40
  export MASK_PACKET_WINDOWS=2000
fi
export MASK_CACHE=""

MANIFEST_DIR="${RESULTS_ROOT}/mask_efficiency"
MANIFEST="${MANIFEST_DIR}/chain_manifest.tsv"
mkdir -p "${MANIFEST_DIR}"
printf 'step\tjob_id\tdependency\tdescription\n' > "${MANIFEST}"

declare -A JOB=()

# submit_step <step> <descricao> <dependencia> <comando...>
submit_step() {
  local step="$1"
  local note="$2"
  local dep="$3"
  shift 3

  echo
  echo "[chain] ${step}"
  echo "[chain]   ${note}"
  if [[ -n "${dep}" ]]; then
    echo "[chain]   dependencia: ${dep}"
  else
    echo "[chain]   dependencia: nenhuma"
  fi

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] ${*}"
    JOB["${step}"]="dry-${step}"
    printf '%s\t%s\t%s\t%s\n' \
      "${step}" "dry-${step}" "${dep:-—}" "${note}" >> "${MANIFEST}"
    return 0
  fi

  local output job_id
  if ! output="$(MASK_STEP_NOTE="${note}" DEPENDENCY="${dep}" "$@" 2>&1)"; then
    printf '%s\n' "${output}"
    echo "[error] ${step} falhou ao submeter" >&2
    exit 1
  fi
  printf '%s\n' "${output}"

  job_id="$(printf '%s\n' "${output}" \
    | awk '/^\[submit\]/{line=$0} END{if(line==""){exit 1} n=split(line,a," "); print a[n]}')" || {
      echo "[error] nao encontrei o job id de ${step}" >&2
      exit 1
    }
  job_id="${job_id//[^0-9]/}"
  if [[ -z "${job_id}" ]]; then
    echo "[error] job id vazio em ${step}" >&2
    exit 1
  fi
  JOB["${step}"]="${job_id}"
  printf '%s\t%s\t%s\t%s\n' \
    "${step}" "${job_id}" "${dep:-—}" "${note}" >> "${MANIFEST}"
  echo "[chain]   job=${job_id}"
}

echo "=================================================================="
echo "[chain] Estudo de mascaras do RePO — modo ${MODE}"
echo "[chain] repositorio: ${REPO_ROOT}"
echo "[chain] resultados: ${RESULTS_ROOT}/mask_efficiency"
echo "[chain] manifesto:  ${MANIFEST}"
echo "[chain] amostra:    ${MASK_MAX_BENIGN} benignos + ${MASK_MAX_PER_ATTACK}/ataque"
echo "[chain] ataques:    ${MASK_ATTACK_SET}"
echo "[chain] variantes:  ${MASK_VARIANTS}"
echo "[chain] aprovacao:  ${MASK_PER_ATTACK}/ataque"
echo "=================================================================="

# ---------------------------------------------------------------------------
# 1. Baseline (Gate G1).
# ---------------------------------------------------------------------------
submit_step baseline \
  "Congela thresholds e TPR do RePO/RePO+ (100 mascaras) na mesma amostra; Gate G1." \
  "" \
  bash "${CLUSTER_DIR}/submit.sh" mask-exp baseline

# ---------------------------------------------------------------------------
# 2. Direcao A — curva de variancia normal (depende do baseline).
# ---------------------------------------------------------------------------
submit_step variance \
  "Curva TPR normal x n_masks x politica (5 repeticoes) e custo; define o joelho e testa o Gate G2." \
  "afterok:${JOB[baseline]}" \
  bash "${CLUSTER_DIR}/submit.sh" mask-exp variance

# ---------------------------------------------------------------------------
# 3. Direcao A — ataque adaptativo em todas as variantes.
# ---------------------------------------------------------------------------
submit_step adversarial \
  "Ataque adaptativo por ataque (array de 11); mede TPR de vitima, queda condicional, fooled, no_change e custo do ataque." \
  "afterok:${JOB[variance]}" \
  bash "${CLUSTER_DIR}/submit.sh" mask-exp adversarial all

# ---------------------------------------------------------------------------
# 4. Direcao B — cascata e RePO sequencial no cenario normal.
# ---------------------------------------------------------------------------
submit_step cascade \
  "Cascata prefixo + RePO sequencial; mede mascaras/amostra na mistura 99/1, benignos que pulam o ensemble e FN do estagio barato." \
  "afterok:${JOB[variance]}" \
  bash "${CLUSTER_DIR}/submit.sh" mask-exp cascade

# ---------------------------------------------------------------------------
# 5. Direcao B — ataque ao estagio barato.
# ---------------------------------------------------------------------------
submit_step cascade_attack \
  "Ataca o estagio barato da cascata; mede bypass, TPR de vitima da cascata e o ensemble completo nos mesmos exemplos." \
  "afterok:${JOB[cascade]}" \
  bash "${CLUSTER_DIR}/submit.sh" mask-exp cascade-attack all

# ---------------------------------------------------------------------------
# 6. Direcao C — treino do aluno de passe unico.
# ---------------------------------------------------------------------------
submit_step distill \
  "Treina o aluno que imita o escore RePO+ e grava distillation_student.keras; mede TPR normal, Spearman, RMSE e custo." \
  "afterok:${JOB[variance]}" \
  bash "${CLUSTER_DIR}/submit.sh" mask-exp distill

# ---------------------------------------------------------------------------
# 7. Direcao C — ataque white-box ao aluno (exige o .keras do passo 6).
# ---------------------------------------------------------------------------
submit_step distill_attack \
  "Ataque white-box no aluno; compara o TPR de vitima do aluno com o do professor nos mesmos exemplos." \
  "afterok:${JOB[distill]}" \
  bash "${CLUSTER_DIR}/submit.sh" mask-exp distill-attack all

# ---------------------------------------------------------------------------
# 8. Confirmacao em pacotes (normal).
# ---------------------------------------------------------------------------
submit_step packet \
  "Confirma a ordenacao das politicas no modelo de pacotes (2000 janelas, normal apenas)." \
  "afterok:${JOB[baseline]}" \
  bash "${CLUSTER_DIR}/submit.sh" mask-exp packet

# ---------------------------------------------------------------------------
# 9. Merge dos arrays + Markdown consolidado.
# ---------------------------------------------------------------------------
if [[ "${SKIP_AGGREGATE}" -eq 0 ]]; then
  AGG_DEP="afterok:${JOB[baseline]}:${JOB[variance]}:${JOB[adversarial]}:${JOB[cascade]}:${JOB[cascade_attack]}:${JOB[distill]}:${JOB[distill_attack]}:${JOB[packet]}"
  submit_step aggregate \
    "Faz o merge dos arrays e gera MASK_EFFICIENCY_RESULTS.md com as metricas por direcao." \
    "${AGG_DEP}" \
    bash "${CLUSTER_DIR}/submit.sh" mask-exp aggregate
fi

echo
echo "[chain] cadeia submetida (${MODE})"
printf '[chain] %-16s %-10s %s\n' "etapa" "job" "descricao"
for step in baseline variance adversarial cascade cascade_attack distill distill_attack packet aggregate; do
  if [[ -n "${JOB[${step}]:-}" ]]; then
    printf '[chain] %-16s %-10s\n' "${step}" "${JOB[${step}]}"
  fi
done
echo "[chain] manifesto: ${MANIFEST}"
echo "[chain] acompanhe: squeue -u \"\$USER\""
echo "[chain] logs:      ${RESULTS_ROOT}/logs/"
