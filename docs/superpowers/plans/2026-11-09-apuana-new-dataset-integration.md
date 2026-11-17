# Apuana New Dataset Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrar o pacote `code/new_data_eval/` à branch `feat/apuana-cluster` e adicionar setup, submissão, treino, smoke e agregação do novo dataset no cluster Apuana, sem alterar o fluxo de fluxo/pacotes.

**Architecture:** O merge traz a CLI `new_data_eval` para a branch. `cluster/setup_env.sh` extrai o ZIP LFS para `$DATA_ROOT/new_data` e cria symlinks em `data/new_data/`. `cluster/slurm/job.sh` ganha tasks `new_data_*` que chamam `python -m new_data_eval` com `PYTHONPATH=code`. `cluster/submit.sh` expõe o namespace `new-data`. `cluster/aggregate_new_data.py` consolida os JSONs.

**Tech Stack:** Bash, SLURM, Python 3.10, TensorFlow 2.21, NumPy, Pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-11-09-apuana-new-dataset-integration-design.md`

## Global Constraints

- Trabalhar em `feat/apuana-cluster` no repositório principal; não reescrever a branch do novo dataset.
- Não alterar `cluster/repo_runner.py`, `cluster/metrics.py`, `cluster/aggregate_results.py` nem o comportamento de fluxo/pacotes.
- Restaurar, após o merge, os arquivos de fluxo/pacotes que a branch do novo dataset alterou por herança: `results/flow_based_adversarial_results.json`, `results/packet_based_adversarial_results.json`, `code/Flow_based_RePO_Adversarial.ipynb`, `code/Packet_based_RePO_Adversarial.ipynb`, `code/Colab_Setup.ipynb`.
- Não commitar CSVs extraídos nem `.npz`.
- O pacote é importado como `new_data_eval` com `PYTHONPATH=code`; nunca criar `code/__init__.py`.
- Resultados de jobs ficam em `$RESULTS_ROOT/new_data/`; nada é escrito em `results/new_data/` do repositório.
- Dataset: `data/new_data/New-dataset-split-clean.zip` (LFS), contendo `new_train.csv` e `new_test.csv`.
- Modelo: `models/new_data_model/` (pretrained) ou `models/retrained/new_data_model/` (retrained).
- Metadados: `data/new_data/x_train_meta/train_{min,max}.npy` (versionados).
- Protocolo e defaults do pacote não mudam: 100 máscaras, 75% mascarado, RePO/RePO+, FPR 1%/10%, seed 2019, 5.000 amostras por ataque.
- Variantes: `primary` = `random` + `calibrated`; `sensitivity_first_n` = `first` + `calibrated`; `sensitivity_paper_0.8` = `random` + `paper_0.8`.
- O adversarial canônico exige `normal/repo_plus.json`; não recalcular threshold em nove jobs.
- Defaults SLURM: `PARTITION=short-simple`, `CPUS_PER_TASK=8`, `MEM=32G`, `GRES=gpu:1`, `CONCURRENCY=2`.
- Walltimes: normal `02:00:00`, adversarial `1-23:00:00`, treino `1-00:00:00`, smoke `00:30:00`.
- Escrever JSON atomicamente; nunca append.
- Documentação em português, consistente com `cluster/README.md` e `docs/APUANA_RUNBOOK.md`.

## Review Focus

- O merge não pode mudar os resultados de fluxo/pacotes nem seus notebooks.
- Um clone sem `git-lfs` deve conseguir extrair o ZIP pelo fallback, com erro claro se a rede bloquear.
- `setup_env.sh --skip-new-data` não pode exigir o novo dataset nem quebrar fluxo/pacotes.
- `new-data normal` e `new-data adversarial` devem usar o mesmo `--results-dir`.
- O job adversarial deve falhar antes de carregar o modelo se o normal não existir.
- O array deve ter exatamente nove índices para `all`, um por ataque, e não pode sobrescrever `primary` ao rodar sensibilidades.
- Checkpoint/retomada e `FORCE=1` devem funcionar com `BATCH_SIZE` diferente.
- O export deve ignorar `smoke/` e listar resultados ausentes.
- Nenhum CSV/`.npz` pode aparecer em `git status`.

---

### Task 1: Merge e auditoria

**Files:**
- Modify: `docs/PROJECT_TASKS.md` (resolver conflito)
- Restore: `results/flow_based_adversarial_results.json`, `results/packet_based_adversarial_results.json`, `code/Flow_based_RePO_Adversarial.ipynb`, `code/Packet_based_RePO_Adversarial.ipynb`, `code/Colab_Setup.ipynb`
- Restore: `docs/colab_browser.md`, `docs/gpu_tensorflow_compatibility.md` (desfazer rename cosmético)
- Bring: todo o conteúdo novo de `feat/new-dataset-eval`

**Interfaces:**
- Consumes: branches `feat/apuana-cluster` (checkout atual) e `feat/new-dataset-eval`.
- Produces: commit de merge com o pacote `new_data_eval`, dados/modelo novos e docs, mais as correções de integração.

- [ ] **Step 1: Confirmar estado limpo**

```bash
cd /home/gl-pereira/Projects/replicating_RePO
git status --short
git log -1 --oneline --decorate
```

Expected: sem alterações não commitadas; HEAD em `feat/apuana-cluster`.

- [ ] **Step 2: Fazer o merge sem commit**

```bash
git merge --no-ff --no-commit feat/new-dataset-eval
```

Expected: conflito apenas em `docs/PROJECT_TASKS.md`.

- [ ] **Step 3: Resolver `docs/PROJECT_TASKS.md`**

Ler o arquivo e manter as duas seções: a de fluxo/pacotes da Apuana e a do
novo dataset. Não remover itens concluídos de nenhum dos lados. Marcar como
pendente apenas o que já estava pendente.

- [ ] **Step 4: Restaurar arquivos fora do escopo**

```bash
git checkout HEAD -- \
  results/flow_based_adversarial_results.json \
  results/packet_based_adversarial_results.json \
  code/Flow_based_RePO_Adversarial.ipynb \
  code/Packet_based_RePO_Adversarial.ipynb \
  code/Colab_Setup.ipynb

git checkout HEAD -- docs/colab_browser.md docs/gpu_tensorflow_compatibility.md
git rm -f docs/COLAB_BROWSER.md docs/GPU_TENSORFLOW_COMPATIBILITY.md
```

- [ ] **Step 5: Auditar o merge**

```bash
git diff --cached --name-status HEAD
```

Expected: apenas arquivos do novo dataset, `README.md`, `docs/PROJECT_TASKS.md`,
`.gitignore`, `requirements.txt`, specs/plans e `extract_new_data.sh`. Nenhum
`results/flow_*`, `results/packet_*`, notebook antigo ou arquivo de `cluster/`.

```bash
git diff --cached HEAD -- results/flow_based_adversarial_results.json \
  results/packet_based_adversarial_results.json
```

Expected: diff vazio.

- [ ] **Step 6: Rodar os testes do pacote no clone integrado**

```bash
PYTHONPATH=code .venv/bin/python -m pytest code/new_data_eval/tests -q
```

Expected: `212 passed` (ou o total atual da branch).

- [ ] **Step 7: Commitar o merge**

```bash
git commit -m "merge: integrate new dataset evaluation into Apuana tooling"
```

---

### Task 2: Extração parametrizável e fallback LFS

**Files:**
- Create: `cluster/lfs_fetch.py`
- Create: `cluster/tests/test_lfs_fetch.py`
- Modify: `extract_new_data.sh`

**Interfaces:**
- `parse_pointer(text: str) -> LfsPointer` com `oid: str` e `size: int`.
- `normalize_repository(url: str) -> str` converte `git@github.com:o/r.git` e
  `ssh://git@github.com/o/r.git` para `https://github.com/o/r.git`.
- `batch_payload(oid: str, size: int) -> dict[str, object]`.
- `download_object(repository: str, pointer: LfsPointer, output: Path, token: str | None = None) -> None`.
- CLI: `python cluster/lfs_fetch.py --pointer PATH --output PATH [--repository URL]`.

- [ ] **Step 1: Escrever os testes**

Casos mínimos:

```python
def test_parse_pointer_reads_oid_and_size():
    pointer = parse_pointer(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "a" * 64 + "\n"
        "size 813379301\n"
    )
    assert pointer.oid == "a" * 64
    assert pointer.size == 813379301


def test_normalize_repository_accepts_ssh_and_https():
    assert normalize_repository("git@github.com:o/r.git") == "https://github.com/o/r.git"
    assert normalize_repository("https://github.com/o/r.git") == "https://github.com/o/r.git"


def test_batch_payload_declares_download():
    payload = batch_payload("a" * 64, 10)
    assert payload["operation"] == "download"
    assert payload["objects"][0]["oid"] == "a" * 64


def test_download_object_verifies_digest(monkeypatch, tmp_path):
    # mock urllib.request.urlopen for the batch call and the object download
    ...
```

- [ ] **Step 2: Rodar os testes e ver falhar**

```bash
PYTHONPATH=code .venv/bin/python -m pytest cluster/tests/test_lfs_fetch.py -q
```

Expected: FAIL porque o módulo não existe.

- [ ] **Step 3: Implementar `cluster/lfs_fetch.py`**

Requisitos:

- só usa stdlib (`argparse`, `hashlib`, `json`, `urllib.request`);
- valida `oid` de 64 hex e `size > 0`;
- envia `Accept: application/vnd.git-lfs+json` e
  `Content-Type: application/vnd.git-lfs+json`;
- usa `Authorization: Bearer $GITHUB_TOKEN` se definido;
- baixa para arquivo temporário no diretório de saída, verifica tamanho e
  SHA-256, e faz `os.replace`;
- erro com mensagem clara se a API não devolver `actions.download.href`.

- [ ] **Step 4: Parametrizar `extract_new_data.sh`**

Adicionar no topo:

```bash
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/new_data}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}}"
ZIP_PATH="${ZIP_PATH:-${DATA_DIR}/New-dataset-split-clean.zip}"
TRAIN_CSV="${OUTPUT_DIR}/new_train.csv"
TEST_CSV="${OUTPUT_DIR}/new_test.csv"
```

Fluxo:

1. `mkdir -p "$OUTPUT_DIR"`.
2. Se `git lfs version` funciona e `ZIP_PATH` está dentro do repo:
   `git lfs pull --include=<caminho relativo>`.
3. Se o ZIP continuar sendo ponteiro ou `git lfs` não existir:
   `python3 cluster/lfs_fetch.py --pointer "$ZIP_PATH" --output "$ZIP_PATH"`.
4. `unzip -o "$ZIP_PATH" -d "$OUTPUT_DIR"`.
5. Validar CSVs em `OUTPUT_DIR` e resumir linhas/hash.

- [ ] **Step 5: Verificar sintaxe e testes**

```bash
bash -n extract_new_data.sh
PYTHONPATH=code .venv/bin/python -m pytest cluster/tests/test_lfs_fetch.py -q
```

---

### Task 3: Helpers em `cluster/common.sh`

**Files:**
- Modify: `cluster/common.sh`

**Interfaces:**
- `new_data_model_dir_for(source: str) -> str`
- `new_data_results_root() -> str`
- `require_new_data(model_source: str) -> None`

- [ ] **Step 1: Implementar**

```bash
new_data_model_dir_for() {
  local source="${1:-${MODEL_SOURCE}}"
  if [[ -n "${MODEL_DIR:-}" ]]; then
    printf '%s\n' "${MODEL_DIR}"
    return 0
  fi
  if [[ "${source}" == "retrained" ]]; then
    printf '%s\n' "${REPO_ROOT}/models/retrained/new_data_model"
  else
    printf '%s\n' "${REPO_ROOT}/models/new_data_model"
  fi
}

new_data_results_root() {
  printf '%s\n' "${RESULTS_ROOT}/new_data"
}

require_new_data() {
  local source="${1:-${MODEL_SOURCE}}"
  local missing=0
  local file
  for file in \
    "${REPO_ROOT}/data/new_data/new_train.csv" \
    "${REPO_ROOT}/data/new_data/new_test.csv" \
    "${REPO_ROOT}/data/new_data/x_train_meta/train_min.npy" \
    "${REPO_ROOT}/data/new_data/x_train_meta/train_max.npy"; do
    if [[ ! -s "${file}" ]]; then
      echo "[error] arquivo ausente ou vazio: ${file}" >&2
      missing=1
    fi
  done
  local model_dir
  model_dir="$(new_data_model_dir_for "${source}")"
  if [[ ! -f "${model_dir}/saved_model.pb" ]]; then
    echo "[error] SavedModel ausente: ${model_dir}/saved_model.pb" >&2
    missing=1
  fi
  if (( missing )); then
    echo "[hint] rode bash cluster/setup_env.sh (ou use --skip-new-data se só precisa de fluxo/pacotes)" >&2
    return 2
  fi
}
```

Adicionar ao final:

```bash
mkdir -p "$(new_data_results_root)/logs" "$(new_data_results_root)/attack-lists"
```

- [ ] **Step 2: Verificar sintaxe**

```bash
bash -n cluster/common.sh
```

---

### Task 4: `cluster/setup_env.sh`

**Files:**
- Modify: `cluster/setup_env.sh`

**Interfaces:**
- Novas flags: `--skip-new-data`, `--force-new-data`.
- Usa `new_data_model_dir_for` para preflight.
- Cria symlinks `data/new_data/new_{train,test}.csv`.

- [ ] **Step 1: Adicionar flags**

```bash
SKIP_NEW_DATA=0
FORCE_NEW_DATA=0
# no case:
--skip-new-data) SKIP_NEW_DATA=1 ;;
--force-new-data) FORCE_NEW_DATA=1 ;;
```

Atualizar o bloco de ajuda (`sed -n '2,18p'`).

- [ ] **Step 2: Extrair e linkar o novo dataset**

Depois do bloco de pacotes e antes do preflight:

```bash
if (( ! SKIP_NEW_DATA )); then
  NEW_DATA_DIR="${DATA_ROOT}/new_data"
  mkdir -p "${NEW_DATA_DIR}"

  if (( FORCE || FORCE_DATA || FORCE_NEW_DATA )) \
     || [[ ! -s "${NEW_DATA_DIR}/new_train.csv" ]]; then
    echo "[setup] extraindo novo dataset para ${NEW_DATA_DIR}"
    OUTPUT_DIR="${NEW_DATA_DIR}" \
    ZIP_PATH="${REPO_ROOT}/data/new_data/New-dataset-split-clean.zip" \
      bash "${REPO_ROOT}/extract_new_data.sh"
  else
    echo "[setup] novo dataset já extraído: ${NEW_DATA_DIR}"
  fi

  for name in new_train.csv new_test.csv; do
    source_file="${NEW_DATA_DIR}/${name}"
    destination="${REPO_ROOT}/data/new_data/${name}"
    if [[ -L "${destination}" || ! -e "${destination}" ]]; then
      ln -sfn "${source_file}" "${destination}"
    else
      echo "[warn] mantendo arquivo existente: ${destination}" >&2
    fi
  done
fi
```

- [ ] **Step 3: Preflight condicional**

```bash
require_flow_data
require_packet_data
require_pretrained_models flow pretrained
require_pretrained_models packet pretrained

if (( ! SKIP_NEW_DATA )); then
  require_new_data "${MODEL_SOURCE}"
else
  echo "[warn] novo dataset ignorado (--skip-new-data)" >&2
fi
```

- [ ] **Step 4: Ajustar mensagens finais**

Adicionar ao bloco de próximos passos:

```bash
echo "     bash cluster/submit.sh new-data normal"
echo "     SMOKE=1 bash cluster/submit.sh new-data smoke ddos"
```

- [ ] **Step 5: Verificar**

```bash
bash -n cluster/setup_env.sh
bash cluster/setup_env.sh --help
```

Expected: ajuda lista as flags novas.

---

### Task 5: Tasks no `cluster/slurm/job.sh`

**Files:**
- Modify: `cluster/slurm/job.sh`

**Interfaces:**
- Novos `TASK`: `new_data_normal`, `new_data_adversarial`, `new_data_smoke`, `new_data_train`.
- Exporta `PYTHONPATH=$REPO_ROOT/code`.
- `NEW_DATA_RESULTS` default `$(new_data_results_root)`.

- [ ] **Step 1: Adicionar resolução de paths**

Antes do `case "${TASK}"`:

```bash
export PYTHONPATH="${REPO_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
NEW_DATA_DIR="${REPO_ROOT}/data/new_data"
NEW_DATA_META_DIR="${NEW_DATA_DIR}/x_train_meta"
NEW_DATA_RESULTS="${NEW_DATA_RESULTS:-$(new_data_results_root)}"
```

- [ ] **Step 2: Implementar `new_data_normal`**

```bash
new_data_normal)
  NEW_DATA_MODEL_DIR="${MODEL_DIR:-$(new_data_model_dir_for "${MODEL_SOURCE}")}"
  ND_ARGS=(
    --data-dir "${NEW_DATA_DIR}"
    --model-dir "${NEW_DATA_MODEL_DIR}"
    --meta-dir "${NEW_DATA_META_DIR}"
    --results-dir "${NEW_DATA_RESULTS}"
    --modes "${NEW_DATA_MODES:-repo,repo_plus}"
    --seed "${SEED}"
    --batch-size "${BATCH_SIZE:-1024}"
  )
  [[ -n "${N_MASKS:-}" ]] && ND_ARGS+=(--n-masks "${N_MASKS}")
  [[ -n "${SCORE_REPEATS:-}" ]] && ND_ARGS+=(--score-repeats "${SCORE_REPEATS}")
  [[ "${PAIRED_SCORING:-0}" == "1" ]] && ND_ARGS+=(--paired-scoring)
  mkdir -p "${NEW_DATA_RESULTS}"
  exec "${PY}" -m new_data_eval normal "${ND_ARGS[@]}"
  ;;
```

- [ ] **Step 3: Implementar `new_data_adversarial`**

```bash
new_data_adversarial)
  ATTACK="${ATTACK:-}"
  if [[ -z "${ATTACK}" && -n "${ATTACK_LIST_FILE:-}" ]]; then
    [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || {
      echo "[error] ATTACK_LIST_FILE exige job array" >&2; exit 2;
    }
    ATTACK="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${ATTACK_LIST_FILE}")"
    ATTACK="${ATTACK%$'\r'}"
  fi
  [[ -n "${ATTACK}" ]] || { echo "[error] defina ATTACK" >&2; exit 2; }

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
  [[ -n "${MAX_SAMPLES:-}" ]] && ND_ADV+=(--max-samples "${MAX_SAMPLES}")
  [[ -n "${N_MASKS:-}" ]] && ND_ADV+=(--n-masks "${N_MASKS}")
  [[ -n "${SCORE_REPEATS:-}" ]] && ND_ADV+=(--score-repeats "${SCORE_REPEATS}")
  [[ "${PAIRED_SCORING:-0}" == "1" ]] && ND_ADV+=(--paired-scoring)
  [[ "${FORCE:-0}" == "1" ]] && ND_ADV+=(--force)
  exec "${PY}" -m new_data_eval adversarial "${ND_ADV[@]}"
  ;;
```

- [ ] **Step 4: Implementar `new_data_smoke` e `new_data_train`**

```bash
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
  [[ "${FORCE:-0}" == "1" ]] && ND_SMOKE+=(--force)
  exec "${PY}" -m new_data_eval smoke "${ND_SMOKE[@]}"
  ;;
new_data_train)
  NEW_DATA_MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/models/retrained/new_data_model}"
  case "${NEW_DATA_MODEL_DIR}" in
    "${REPO_ROOT}/models/new_data_model")
      echo "[error] recusando sobrescrever o modelo dos autores" >&2
      exit 2
      ;;
  esac
  exec "${PY}" "${REPO_ROOT}/code/train_new_data_model.py" \
    --train-csv "${REPO_ROOT}/data/new_data/new_train.csv" \
    --output-dir "${NEW_DATA_MODEL_DIR}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS:-5}" \
    --batch-size "${TRAIN_BATCH_SIZE:-256}"
  ;;
```

- [ ] **Step 5: Verificar sintaxe**

```bash
bash -n cluster/slurm/job.sh
```

---

### Task 6: Submissão em `cluster/submit.sh`

**Files:**
- Modify: `cluster/submit.sh`

**Interfaces:**
- Comandos: `new-data normal`, `new-data adversarial <attack|all> [variant]`, `new-data smoke [attack]`, `new-data train`, `new-data export`.
- Funções: `task_new_data_normal`, `task_new_data_adversarial`, `task_new_data_smoke`, `task_new_data_train`, `task_new_data_export`.

- [ ] **Step 1: Adicionar walltimes**

```bash
WALLTIME_NEW_DATA_NORMAL="${WALLTIME_NEW_DATA_NORMAL:-02:00:00}"
WALLTIME_NEW_DATA_ADV="${WALLTIME_NEW_DATA_ADV:-1-23:00:00}"
WALLTIME_NEW_DATA_TRAIN="${WALLTIME_NEW_DATA_TRAIN:-1-00:00:00}"
```

- [ ] **Step 2: Implementar normal**

```bash
task_new_data_normal() {
  local modes="${1:-repo repo_plus}"
  modes="${modes// /,}"
  local joined="${modes//,/\\+}"
  local job_id
  job_id="$(submit_gpu_job "repo-new-data-normal" "${WALLTIME_NEW_DATA_NORMAL}" \
    "$(export_args new_data_normal new_data "DATASET=new_data,NEW_DATA_RESULTS=${RESULTS_ROOT}/new_data,NEW_DATA_MODES=${joined},BATCH_SIZE=${BATCH_SIZE:-1024}")")"
  echo "[submit] new-data normal (${joined}): ${job_id}"
}
```

- [ ] **Step 3: Implementar adversarial**

Mapear variante:

```bash
case "${variant}" in
  primary)              sample_mode=random; policy=calibrated ;;
  sensitivity_first_n)  sample_mode=first;  policy=calibrated ;;
  sensitivity_paper_0.8) sample_mode=random; policy=paper_0.8 ;;
  *) echo "[error] variante inválida: ${variant}" >&2; exit 2 ;;
esac
```

Montar a lista:

```bash
list_dir="${RESULTS_ROOT}/new_data/attack-lists"
mkdir -p "${list_dir}"
list_file="$(mktemp "${list_dir}/attacks-XXXXXXXX.txt")"
if [[ "${attack}" == "all" ]]; then
  PYTHONPATH="${REPO_ROOT}/code" "${VENV}/bin/python" - "${list_file}" <<'PY'
import sys
from pathlib import Path
from new_data_eval.data import ATTACK_LABELS
Path(sys.argv[1]).write_text("\n".join(ATTACK_LABELS) + "\n", encoding="utf-8")
print(f"[list] {len(ATTACK_LABELS)} ataques -> {sys.argv[1]}")
PY
else
  printf '%s\n' "${attack}" > "${list_file}"
fi
```

Submeter:

```bash
count="$(grep -cve '^[[:space:]]*$' "${list_file}")"
array_spec="0-$((count - 1))%${CONCURRENCY}"
extra="DATASET=new_data,NEW_DATA_RESULTS=${RESULTS_ROOT}/new_data,ATTACK_LIST_FILE=${list_file},VARIANT=${variant},SAMPLE_MODE=${sample_mode},THRESHOLD_POLICY=${policy},BATCH_SIZE=${BATCH_SIZE:-1024},CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-50}"
[[ -n "${MAX_SAMPLES:-}" ]] && extra+=",MAX_SAMPLES=${MAX_SAMPLES}"
job_id="$(OUTPUT_PATTERN='%x-%A_%a.out' submit_gpu_job "repo-new-data-adv-${variant}" "${WALLTIME_NEW_DATA_ADV}" \
  "$(export_args new_data_adversarial new_data "${extra}")" --array="${array_spec}")"
```

- [ ] **Step 4: Implementar smoke, train e export**

- `new-data smoke [attack]`: espelhar `task_smoke`, mas com
  `TASK=new_data_smoke`, `ATTACK`, walltime `00:30:00`; com `SMOKE=1` submete.
- `new-data train`: `TASK=new_data_train`, output default
  `models/retrained/new_data_model`, recusar `models/new_data_model`.
- `new-data export`: rodar direto no login:

```bash
"${VENV}/bin/python" "${CLUSTER_DIR}/aggregate_new_data.py" \
  --results-dir "${RESULTS_ROOT}/new_data" \
  --out "${OUT_DIR:-${REPO_ROOT}/results/cluster/new_data}"
```

- [ ] **Step 5: Dispatch e usage**

```bash
new-data)
  sub="${1:-}"; shift || true
  case "${sub}" in
    normal) task_new_data_normal "${1:-repo repo_plus}" ;;
    adversarial) task_new_data_adversarial "${1:?attack}" "${2:-primary}" ;;
    smoke) task_new_data_smoke "${1:-ddos}" ;;
    train) task_new_data_train ;;
    export) task_new_data_export ;;
    *) usage; exit 2 ;;
  esac
  ;;
```

Atualizar o cabeçalho de exemplos e `usage()`.

- [ ] **Step 6: Verificar**

```bash
bash -n cluster/submit.sh
bash cluster/submit.sh 2>&1 | head -30
```

---

### Task 7: Agregador do novo dataset

**Files:**
- Create: `cluster/aggregate_new_data.py`
- Create: `cluster/tests/test_aggregate_new_data.py`

**Interfaces:**
- `load_normal(results_dir: Path) -> list[dict[str, Any]]`
- `load_adversarial(results_dir: Path) -> list[dict[str, Any]]`
- `summary_rows(results_dir: Path) -> tuple[list[dict[str, Any]], list[str]]`
- `write_summary(results_dir: Path, out_dir: Path) -> None`
- CLI: `python cluster/aggregate_new_data.py --results-dir DIR --out DIR`.

- [ ] **Step 1: Escrever os testes**

Fixture em `tmp_path`:

```text
normal/repo.json
normal/repo_plus.json
adversarial/primary/ddos.json
adversarial/sensitivity_first_n/ddos.json
smoke/adversarial/ddos.json   # deve ser ignorado
```

Asserts:

- `summary_rows` retorna 2 linhas normais e 2 adversariais;
- `smoke/` não aparece;
- ataque ausente entra em `missing`;
- CSV tem cabeçalho e uma linha por resultado;
- Markdown contém `## Normal` e `## Adversarial`.

- [ ] **Step 2: Implementar**

Regras:

- ignorar `smoke/`;
- ler `normal/*.json` e `adversarial/<variant>/*.json`; ignorar variantes
  desconhecidas;
- campos normal: modo, `n`, thresholds 0.01/0.10, FPR, `average_tpr.macro`,
  `weighted`, `runtime_seconds`, `config_hash`, `git_commit`;
- campos adversarial: variante, ataque, `n`, policy, thresholds, FPR, TPR
  normal/vítima, `attack_success_rate`, `conditional_drop`, `fooled`,
  `evaded`, `backup`, `cons_as_mal`, `cons_as_ben`, `fooled_rate`,
  `runtime_seconds`, `config_hash`, `git_commit`;
- escrever `summary.csv`, `summary.md`, `summary.json` com `os.replace`;
- listar arquivos ausentes de `primary`, `sensitivity_first_n` e
  `sensitivity_paper_0.8` em `missing`;
- não importar TensorFlow.

- [ ] **Step 3: Rodar**

```bash
PYTHONPATH=code .venv/bin/python -m pytest cluster/tests/test_aggregate_new_data.py -q
```

---

### Task 8: Configuração e documentação

**Files:**
- Modify: `cluster/config.env.example`
- Modify: `cluster/README.md`
- Modify: `docs/APUANA_RUNBOOK.md`
- Modify: `docs/NEW_DATASET.md`

- [ ] **Step 1: `cluster/config.env.example`**

Adicionar:

```bash
# Novo dataset
export WALLTIME_NEW_DATA_NORMAL="${WALLTIME_NEW_DATA_NORMAL:-02:00:00}"
export WALLTIME_NEW_DATA_ADV="${WALLTIME_NEW_DATA_ADV:-1-23:00:00}"
export WALLTIME_NEW_DATA_TRAIN="${WALLTIME_NEW_DATA_TRAIN:-1-00:00:00}"
# export NEW_DATA_MODES="repo,repo_plus"
# export MAX_SAMPLES=5000
# export N_MASKS=100
# export BATCH_SIZE=1024
# export CHECKPOINT_EVERY=50
```

- [ ] **Step 2: `cluster/README.md`**

Adicionar uma seção “Novo dataset” com:

- o que o setup extrai e onde;
- comandos `new-data` (smoke, normal, adversarial por variante, train, export);
- piloto recomendado (`MAX_SAMPLES=200 N_MASKS=5`);
- estimativas e limites;
- ressalva de que `--attack all` não é usado no cluster;
- proteção do modelo dos autores.

- [ ] **Step 3: `docs/APUANA_RUNBOOK.md`**

Inserir passos lineares:

1. setup do novo dataset (e `--skip-new-data`);
2. smoke;
3. normal;
4. adversarial `primary`;
5. sensibilidades opcionais;
6. treino opcional;
7. export e coleta.

- [ ] **Step 4: `docs/NEW_DATASET.md`**

Atualizar a nota de escopo: o fluxo agora tem integração em `cluster/`, e os
comandos do Apuana estão no runbook. Não afirmar que o dataset “não usa
cluster”.

- [ ] **Step 5: Conferir links e comandos**

```bash
grep -R "new-data" -n cluster/README.md docs/APUANA_RUNBOOK.md docs/NEW_DATASET.md
grep -R "skip-new-data" -n cluster/setup_env.sh cluster/README.md docs/APUANA_RUNBOOK.md
```

---

### Task 9: Verificação final

**Files:**
- Nenhum novo; só validação.

- [ ] **Step 1: Testes Python**

```bash
PYTHONPATH=code .venv/bin/python -m pytest code/new_data_eval/tests cluster/tests -q
```

Expected: tudo passa.

- [ ] **Step 2: Sintaxe dos scripts**

```bash
bash -n extract_new_data.sh
bash -n cluster/common.sh
bash -n cluster/setup_env.sh
bash -n cluster/submit.sh
bash -n cluster/slurm/job.sh
```

- [ ] **Step 3: Auditoria de escopo**

```bash
git diff --name-status feat/apuana-cluster..HEAD
git diff feat/apuana-cluster..HEAD -- \
  results/flow_based_adversarial_results.json \
  results/packet_based_adversarial_results.json \
  cluster/repo_runner.py
```

Expected: nenhum resultado de fluxo/pacotes e nenhum runner antigo alterado.

- [ ] **Step 4: Higiene**

```bash
git status --short --ignored | grep -E '\.csv$|\.npz$' || true
git ls-files | grep -E '\.csv$|\.npz$' || true
```

Expected: nenhum CSV/`.npz` rastreado; extraídos aparecem como ignorados.

- [ ] **Step 5: Verificação no cluster (operador)**

```bash
bash cluster/setup_env.sh
SMOKE=1 bash cluster/submit.sh new-data smoke ddos
bash cluster/submit.sh new-data normal
bash cluster/submit.sh new-data adversarial all primary
bash cluster/submit.sh new-data export
```

Expected:

- smoke grava `$RESULTS_ROOT/new_data/smoke/`;
- normal grava `normal/repo.json` e `normal/repo_plus.json`;
- adversarial grava nove JSONs em `adversarial/primary/`;
- export grava `results/cluster/new_data/{summary.csv,summary.md,summary.json}`.

- [ ] **Step 6: Commit**

```bash
git add cluster/ extract_new_data.sh docs/ cluster/config.env.example
git commit -m "feat: run new dataset evaluation on the Apuana cluster"
```
