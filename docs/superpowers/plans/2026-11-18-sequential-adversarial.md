# Ataque adversarial à regra sequencial (direção B) — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** medir a robustez adversarial da decisão sequencial do RePO (parada antecipada por IC) nos mesmos 11 ataques e no mesmo volume do `cascade_attack`, para colocar cascata e sequencial lado a lado.

**Architecture:** o ataque é adaptativo contra a **estatística que a regra usa** (a média dos erros por máscara) na **parada mais cedo possível** (primeiro bloco). O atacante minimiza `média + z·erro-padrão` do bloco — exatamente o limite superior do IC que a regra compara com o threshold — via gradiente no `FlowAttack` existente, e a decisão da vítima é sempre recalculada com `sequential_repo_decisions` sobre 100 máscaras novas (mesmo protocolo da avaliação normal). O runner novo (`sequential_attack.py`) espelha `cascade_attack.py`: um job SLURM por ataque, JSON por ataque, merge no final.

**Tech Stack:** Python 3, NumPy, TensorFlow/Keras (só no `FlowAttack`), pytest, bash + SLURM (`cluster/submit.sh`, `cluster/slurm/job.sh`).

**Spec:** `docs/MASK_EFFICIENCY_CASCADE.md` (método e resultados da direção B) + `code/new_data_eval/decision.py` (`sequential_repo_decisions`) + `code/mask_experiments/cascade_attack.py` (protocolo adversarial da cascata, a ser espelhado).

## Global Constraints

- Branch de trabalho: `feat/apuana-cluster` (sem worktree: os jobs rodam a partir de `$HOME/Projects/replicating_RePO` no cluster e os resultados vivem em `$HOME/replicating-repo-results/mask_efficiency/`).
- Protocolo adversarial idêntico ao da cascata: `--fpr 0.10` (threshold "real" calibrado em 10 % de FPR nos benignos), threshold de ataque = o mesmo valor, 11 ataques CICIDS2017, 1.000 amostras por ataque, 5.000 benignos, `seed=2019`, política `iid`.
- Configurações: `block ∈ {5, 10, 25}` com `min_masks = block`, `z = 1.96`, `max_masks = 100` (o `block` tem de dividir 100).
- Nenhum arquivo de resultado existente pode ser sobrescrito: a saída nova vive em `mask_efficiency/sequential_attack/`.
- `import mask_experiments.sequential_attack` não pode importar TensorFlow no topo do módulo (importar `FlowAttack` dentro de `run_sequential_attack`).
- Commits pequenos, um por tarefa, mensagens em inglês no padrão do repositório (`feat: ...`, `test: ...`).

## Decisão de agenda (bloqueante, precisa do seu OK)

O QOS do cluster permite **1 job simultâneo por usuário**. Hoje a fila tem `32059_3` (A, slowloris, RUNNING), `32059_4` (A, FTP, PENDING) e `32060` (C, PENDING). Um job novo só começa depois deles. Para rodar o sequencial **até amanhã de manhã** é preciso liberar a GPU agora:

```bash
scancel 32059_3 32059_4 32060
```

Consequência: A para em DDoS/Hulk/GoldenEye (1.000) + FTP/SSH (5.000) e C não sai. Alternativa sem cancelar: começar por volta de 01:00 (fim do slowloris) e cobrir apenas ~5 dos 11 ataques antes das 06:30.

**Duas ondas** (o volume do cascade — 11 ataques × 1.000 × 3 configs — leva ~21 h, não cabe na noite):

- **Onda 1** (`blocks=5`, a configuração headline do estudo normal): 11 ataques × 1.000. É a paridade de amostras com o `cascade_attack` e a tabela que fecha a manhã. ~6–9 h.
- **Onda 2** (`blocks=10,25`, sensibilidade): +~13 h, quinta durante o dia.

## Review Focus

- `block` que não divide `max_masks` (ex.: 7): a regra pararia no último múltiplo e o atacante otimizaria um bloco que a regra nunca usa. Esperado: `ValueError` na validação.
- Ataque sem nenhuma detecção normal (Bot 0,029; SSH 0,007): `conditional_drop` deve ser `None`, nunca divisão por zero nem `NaN`.
- Todas as amostras já benignas no limiar de ataque (`no_change` em 1.000/1.000): as métricas precisam existir com zeros e a média de `masks_evaluated` não pode quebrar com lista vazia.
- `block == max_masks == 100`: um único bloco, sem continuação; `masks_used` = 100 e nenhum laço infinito.
- JSON inválido: `None` em vez de `NaN`; `json.dumps(payload, allow_nan=False)` tem de funcionar.

---

### Task 1: Limite superior do IC compartilhado entre a regra e o atacante

**Files:**
- Modify: `code/new_data_eval/decision.py:159-178` (laço de `sequential_repo_decisions`)
- Test: `code/new_data_eval/tests/test_decision.py`

**Interfaces:**
- Produces: `mean_confidence_bounds(sample: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]` — recebe um bloco `(n_samples, n_masks)` e devolve `(low, high)` com `mean ∓ z·std(ddof=1)/sqrt(n_masks)`; com uma máscara só, erro-padrão 0. Exportar em `__all__`.

- [ ] **Step 1: escrever os testes que falham**

```python
def test_mean_confidence_bounds_match_the_stopping_rule():
    errors = np.array([[0.0, 10.0, 0.0, 10.0, 0.0]])
    low, high = mean_confidence_bounds(errors, z=1.96)
    mean = 4.0
    se = errors[0].std(ddof=1) / np.sqrt(5)
    assert low[0] == pytest.approx(mean - 1.96 * se)
    assert high[0] == pytest.approx(mean + 1.96 * se)

    decided = sequential_repo_decisions(
        errors, threshold=float(high[0]) + 1e-9, block=5, min_masks=5
    )
    assert decided.decision.tolist() == [False]
    assert decided.masks_used.tolist() == [5]

    undecided = sequential_repo_decisions(
        errors, threshold=float(high[0]) - 1e-9, block=5, min_masks=5, max_masks=100
    )
    assert undecided.masks_used.tolist() == [100]


def test_sequential_rule_uses_every_mask_when_block_is_the_whole_budget():
    errors = np.full((2, 100), 10.0)
    result = sequential_repo_decisions(errors, threshold=5.0, block=100, min_masks=100)
    assert result.decision.tolist() == [True, True]
    assert result.masks_used.tolist() == [100, 100]
```

- [ ] **Step 2: rodar e ver falhar**

Run: `python3 -m pytest code/new_data_eval/tests/test_decision.py -q`
Expected: FAIL — `ImportError: cannot import name 'mean_confidence_bounds'`.

- [ ] **Step 3: implementar `mean_confidence_bounds` e usá-la no laço**

Em `code/new_data_eval/decision.py`, adicionar a função (valida `ndim == 2`, `n_masks >= 1`, `z` finito e não negativo) e trocar as linhas 163-173 por `low, high = mean_confidence_bounds(sample, z)`. O comportamento da regra não muda.

- [ ] **Step 4: rodar e ver passar**

Run: `python3 -m pytest code/new_data_eval/tests/test_decision.py code/mask_experiments/tests -q`
Expected: PASS (os testes antigos da cascata e do merge continuam verdes).

- [ ] **Step 5: commit**

```bash
git add code/new_data_eval/decision.py code/new_data_eval/tests/test_decision.py
git commit -m "feat: expose the sequential confidence bound"
```

---

### Task 2: Objetivo `ci` no FlowAttack

**Files:**
- Modify: `code/new_data_eval/attack.py:201-247` (`__init__`) e `:409-433` (`_make_optim`)
- Test: `code/new_data_eval/tests/test_attack_loss.py` (novo)

**Interfaces:**
- Produces: `validate_objective(loss: str, n_masks: int) -> None` e `ci_objective(per_mask_errors, z)` (op TF).
- Produces: `FlowAttack(..., loss: str = "sum", ci_z: float = 1.96)`.
- Consumes: `mean_confidence_bounds` (Task 1) apenas como referência matemática — a implementação em TF é independente.

O objetivo `ci` é `Σe + z·sqrt(n·var)` com `var` de `ddof=1`, que é `n·(média + z·se)`. Multiplicado por `n`, o critério `objetivo < n·τ` é **idêntico** ao `high < threshold` da regra.

- [ ] **Step 1: escrever os testes que falham**

```python
import numpy as np
import pytest

from new_data_eval.attack import validate_objective


def test_validate_objective_rejects_unknown_loss_and_short_ci():
    with pytest.raises(ValueError, match="loss"):
        validate_objective("median", 5)
    with pytest.raises(ValueError, match="at least 2 masks"):
        validate_objective("ci", 1)
    validate_objective("sum", 1)
    validate_objective("ci", 5)


def test_ci_objective_matches_the_rule_upper_bound():
    tf = pytest.importorskip("tensorflow")
    from new_data_eval.attack import ci_objective

    values = np.array([0.0, 10.0, 0.0, 10.0, 0.0])
    n = values.size
    expected = n * (values.mean() + 1.96 * values.std(ddof=1) / np.sqrt(n))
    got = float(ci_objective(tf.constant(values, dtype=tf.float32), 1.96).numpy())
    assert got == pytest.approx(expected, rel=1e-5)
```

- [ ] **Step 2: rodar e ver falhar**

Run: `python3 -m pytest code/new_data_eval/tests/test_attack_loss.py -q`
Expected: FAIL — `ImportError: cannot import name 'validate_objective'` (o segundo teste é `skipped` localmente, roda no cluster).

- [ ] **Step 3: implementar**

`validate_objective` no topo de `attack.py`; `ci_objective` importa TF por `_tensorflow()` e calcula:

```python
n = tf.cast(tf.size(per_mask_errors), tf.float32)
mean = tf.reduce_mean(per_mask_errors)
var = tf.reduce_sum(tf.square(per_mask_errors - mean)) / (n - 1.0)
return tf.reduce_sum(per_mask_errors) + z * tf.sqrt(n * var)
```

Em `__init__`: `self.loss`, `self.ci_z`, chamada a `validate_objective(loss, n_masks)` antes de compilar. Em `_make_optim`, substituir `loss = t.reduce_sum(score)` por `loss = ci_objective(score, self.ci_z) if self.loss == "ci" else t.reduce_sum(score)`, mantendo `score = t.reduce_mean(t.square(rec_x - adv_ex_normalized), axis=1)`.

- [ ] **Step 4: rodar e ver passar**

Run: `python3 -m pytest code/new_data_eval/tests/test_attack_loss.py code/new_data_eval/tests/test_decision.py -q`
Expected: PASS (1 passed + 1 skipped localmente; no cluster com TF, 2 passed).

- [ ] **Step 5: commit**

```bash
git add code/new_data_eval/attack.py code/new_data_eval/tests/test_attack_loss.py
git commit -m "feat: add a confidence-bound objective to the flow attack"
```

---

### Task 3: Runner `sequential_attack`

**Files:**
- Create: `code/mask_experiments/sequential_attack.py`
- Test: `code/mask_experiments/tests/test_sequential_attack.py`

**Interfaces:**
- Produces: `DEFAULT_BLOCKS = (5, 10, 25)`, `DEFAULT_ATTACKS = FLOW_ATTACKS`
- Produces: `validate_blocks(blocks: Iterable[int], max_masks: int) -> tuple[int, ...]`
- Produces: `sequential_attack_metrics(*, normal: SequentialResult, victim: SequentialResult, rows: np.ndarray, block: int, transfer_tpr: float, repo_plus_tpr: float) -> dict[str, Any]`
- Produces: `run_sequential_attack(*, data_dir, model_dir, cache=None, max_benign=5_000, max_per_attack=300, per_attack=20, attacks=DEFAULT_ATTACKS, blocks=DEFAULT_BLOCKS, fpr=0.10, z=1.96, max_masks=100, loss="ci", seed=2019, batch_size=100) -> dict[str, Any]`
- Produces: `main(argv)` com `--blocks`, `--z`, `--loss`, `--max-masks`, `--out` (default `results/mask_efficiency/sequential_attack.json`).
- Consumes: `mean_confidence_bounds` (Task 1), `FlowAttack(..., loss=..., ci_z=z)` (Task 2), `load_sample`, `load_flow_model`, `load_flow_meta`, `POLICIES["iid"]`, `reconstruction_errors`, `repo_scores`, `repo_plus_scores`, `threshold_at_fpr`, `sequential_repo_decisions`, `_macro_tpr`, `write_json_atomic`.

- [ ] **Step 1: escrever os testes que falham**

```python
import json

import numpy as np
import pytest

from new_data_eval.decision import SequentialResult
from mask_experiments.sequential_attack import (
    sequential_attack_metrics,
    validate_blocks,
)


def _result(decision, masks_used):
    return SequentialResult(
        decision=np.array(decision, dtype=bool),
        masks_used=np.array(masks_used, dtype=np.int64),
    )


def test_validate_blocks_rejects_a_block_that_does_not_divide_the_budget():
    with pytest.raises(ValueError, match="divide"):
        validate_blocks((5, 7), 100)
    assert validate_blocks((5, 10, 25), 100) == (5, 10, 25)
    assert validate_blocks((100,), 100) == (100,)


def test_metrics_report_the_early_stop_and_the_paired_decisions():
    normal = _result([True, True, False], [5, 10, 5])
    victim = _result([False, True, False], [5, 15, 25])
    metrics = sequential_attack_metrics(
        normal=normal,
        victim=victim,
        rows=np.array([0, 1, 2]),
        block=5,
        transfer_tpr=0.5,
        repo_plus_tpr=0.4,
    )
    assert metrics["n"] == 3
    assert metrics["normal_tpr"] == pytest.approx(2 / 3)
    assert metrics["victim_tpr"] == pytest.approx(1 / 3)
    assert metrics["fooled"] == 1
    assert metrics["already_benign"] == 1
    assert metrics["early_stop_rate"] == pytest.approx(1 / 3)
    assert metrics["victim_masks_used"] == pytest.approx(15.0)
    assert metrics["conditional_drop"] == pytest.approx(0.5)
    assert metrics["transfer_full100_tpr"] == 0.5
    assert metrics["repo_plus_full100_tpr"] == 0.4
    json.dumps(metrics, allow_nan=False)


def test_metrics_are_json_safe_when_nothing_is_detected():
    empty = _result([False, False], [100, 100])
    metrics = sequential_attack_metrics(
        normal=empty,
        victim=empty,
        rows=np.array([0, 1]),
        block=5,
        transfer_tpr=0.0,
        repo_plus_tpr=0.0,
    )
    assert metrics["conditional_drop"] is None
    assert metrics["early_stop_rate"] == 0.0
    json.dumps(metrics, allow_nan=False)
```

- [ ] **Step 2: rodar e ver falhar**

Run: `python3 -m pytest code/mask_experiments/tests/test_sequential_attack.py -q`
Expected: FAIL — `ModuleNotFoundError: mask_experiments.sequential_attack`.

- [ ] **Step 3: implementar o módulo**

`sequential_attack_metrics` (puro, sem TF):

- `n = len(rows)`; `normal_attack = normal.decision[rows]`, `victim_attack = victim.decision[rows]`.
- `early_stop = (~victim_attack) & (victim.masks_used[rows] == block)`; `early_stop_rate = early_stop.mean()`.
- `conditional_drop = None if normal_tpr == 0 else 1 - victim_tpr / normal_tpr`.
- Devolver as chaves: `n, normal_tpr, victim_tpr, conditional_drop, fooled, already_benign, early_stop_rate, victim_masks_used, normal_masks_used, transfer_full100_tpr, repo_plus_full100_tpr`.

`run_sequential_attack`, espelhando `cascade_attack.py`:

1. `load_sample(...)`, `load_flow_meta`, `load_flow_model`, `policy = POLICIES["iid"]`, `benign = labels == BENIGN_LABEL`, `generator = np.random.default_rng(seed + 4)`.
2. `errors = reconstruction_errors(model, x, ..., n_masks=100, phase=PHASE_ORIGINAL, mask_policy=policy, cost=cost)`; `mean_scores = repo_scores(errors)`.
3. `threshold = threshold_at_fpr(mean_scores[benign], fpr)`; `plus_threshold = threshold_at_fpr(repo_plus_scores(errors)[benign], fpr)`.
4. Selecionar `per_attack` linhas por ataque (`generator.choice`, `sort`), como em `cascade_attack.py:109-116`.
5. Para cada `block` em `validate_blocks(blocks, max_masks)`:
   - `normal = sequential_repo_decisions(errors, threshold=threshold, block=block, min_masks=block, max_masks=max_masks)`.
   - `configs[f"seq{block}"] = {block, min_masks: block, z, max_masks, loss, threshold, repo_plus_threshold, normal_fpr: float(normal.decision[benign].mean()), normal_tpr_macro: _macro_tpr(normal.decision.astype(float), labels, 0.5)["macro"], normal_masks_used_benign, normal_masks_used_attack}`.
   - Para cada ataque: `runner = FlowAttack(model, train_min, train_max, decimal, x.shape[1], real_threshold=block * threshold, attack_threshold=block * threshold, seed=seed, n_masks=block, mask_policy=policy, loss=loss, ci_z=z)`; laço `find_adv`; vítimas = `x[pos]` quando `x_adv is None`; `errors_victim = reconstruction_errors(..., np.stack(victims), n_masks=100, phase=PHASE_VICTIM, sample_indices=indices[rows], mask_policy=policy, cost=cost)`; `victim = sequential_repo_decisions(errors_victim, threshold=threshold, block=block, min_masks=block, max_masks=max_masks)`; `transfer_tpr = float(np.mean(repo_scores(errors_victim) >= threshold))`; `repo_plus_tpr = float(np.mean(repo_plus_scores(errors_victim) >= plus_threshold))`.
   - `payload["attacks"][attack][f"seq{block}"] = sequential_attack_metrics(...) | {"kinds": {...}, "masks_evaluated": int(np.mean([...]))}`; `print` de uma linha no formato do `cascade_attack.py:231-236`.
6. Rodapé: `payload["cost"] = cost.as_dict()`, `payload["runtime_seconds"]`, `schema_version="mask_exp_v1"`, `experiment="sequential_attack"`, `dataset={"name": "cicids2017-flow", "n_benign", "attacks": {name: len(rows)}, "seed", "fpr"}`.

O `FlowAttack` é importado **dentro** de `run_sequential_attack` (não no topo). A escala dos thresholds é `block × τ` porque `FlowAttack.score()` devolve a soma RePO+ sobre `block` máscaras (5 grupos de 1), isto é `block × média`.

- [ ] **Step 4: rodar e ver passar**

Run: `python3 -m pytest code/mask_experiments/tests/test_sequential_attack.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: commit**

```bash
git add code/mask_experiments/sequential_attack.py code/mask_experiments/tests/test_sequential_attack.py
git commit -m "feat: attack the sequential stopping rule"
```

---

### Task 4: Merge dos JSONs por ataque

**Files:**
- Modify: `code/mask_experiments/merge.py:19-25` (`__all__`), `:73-90` (referência), `:162-166` (registro no `main`)
- Test: `code/mask_experiments/tests/test_merge.py`

**Interfaces:**
- Produces: `merge_sequential_attack(directory: Path, out: Path) -> dict[str, Any]` — junta `configs` por chave (`seq5`, `seq10`, …) e `attacks` por ataque/config; `dataset["attacks"] = {attack: next(iter(configs.values()))["n"]}`.

- [ ] **Step 1: escrever o teste que falha**

```python
def test_merge_sequential_attack_combines_blocks(tmp_path):
    for attack, block in (("A", 5), ("B", 10)):
        _write(
            tmp_path / "sequential_attack" / f"{attack}-b{block}.json",
            {
                "dataset": {"name": "cicids2017-flow", "attacks": {attack: 3}},
                "configs": {f"seq{block}": {"block": block}},
                "attacks": {attack: {f"seq{block}": {"n": 3, "victim_tpr": 0.1}}},
            },
        )
    merged = merge_sequential_attack(
        tmp_path / "sequential_attack", tmp_path / "sequential_attack_all.json"
    )
    assert set(merged["configs"]) == {"seq5", "seq10"}
    assert merged["dataset"]["attacks"] == {"A": 3, "B": 3}
    assert merged["attacks"]["A"]["seq5"]["n"] == 3


def test_merge_sequential_attack_merges_blocks_of_the_same_attack(tmp_path):
    for block in (5, 10):
        _write(
            tmp_path / "sequential_attack" / f"A-b{block}.json",
            {
                "dataset": {"name": "cicids2017-flow", "attacks": {"A": 3}},
                "configs": {f"seq{block}": {"block": block}},
                "attacks": {"A": {f"seq{block}": {"n": 3}}},
            },
        )
    merged = merge_sequential_attack(
        tmp_path / "sequential_attack", tmp_path / "out.json"
    )
    assert set(merged["attacks"]["A"]) == {"seq5", "seq10"}
```

- [ ] **Step 2: rodar e ver falhar**

Run: `python3 -m pytest code/mask_experiments/tests/test_merge.py -q`
Expected: FAIL — `ImportError: cannot import name 'merge_sequential_attack'`.

- [ ] **Step 3: implementar e registrar no `main`**

Adicionar a função e a entrada `("sequential_attack", merge_sequential_attack, "sequential_attack_all.json")` na tupla de `merge.main` (linhas 162-166).

- [ ] **Step 4: rodar e ver passar**

Run: `python3 -m pytest code/mask_experiments/tests/test_merge.py -q`
Expected: PASS.

- [ ] **Step 5: commit**

```bash
git add code/mask_experiments/merge.py code/mask_experiments/tests/test_merge.py
git commit -m "feat: merge the sequential attack partials"
```

---

### Task 5: Wiring no cluster

**Files:**
- Modify: `cluster/slurm/job.sh:33-39` (convenção `+` → `,`), `:402` (novo case depois de `mask_cascade_attack`)
- Modify: `cluster/submit.sh:506-565` (`task_mask_attack`), `:600-615` (usage)
- Test: `cluster/tests/test_mask_sequential_scripts.py` (novo)

**Interfaces:**
- Consumes: `python -m mask_experiments.sequential_attack` com `--attacks`, `--blocks`, `--z`, `--loss`, `--fpr`, `--max-masks`, `--per-attack`, `--out` (Task 3).
- Produces: `TASK=mask_sequential_attack`, variáveis `MASK_SEQ_BLOCKS` (lista com `+`), `MASK_SEQ_Z`, `MASK_SEQ_LOSS`; subcomando `bash cluster/submit.sh mask-exp sequential-attack <ataque|all>`; saída `${MASK_RESULTS}/sequential_attack/${ATTACK}-b${MASK_SEQ_BLOCKS//,/-}.json`.

- [ ] **Step 1: escrever o teste que falha**

```python
"""Smoke tests of the sequential-attack SLURM wiring."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_job_sh_is_valid_bash_and_has_the_case():
    text = (REPO_ROOT / "cluster" / "slurm" / "job.sh").read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(REPO_ROOT / "cluster" / "slurm" / "job.sh")], check=True)
    assert "mask_sequential_attack)" in text
    assert "sequential_attack/${ATTACK}" in text
    assert "MASK_SEQ_BLOCKS" in text


def test_submit_sh_is_valid_bash_and_exposes_the_command():
    text = (REPO_ROOT / "cluster" / "submit.sh").read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(REPO_ROOT / "cluster" / "submit.sh")], check=True)
    assert "sequential-attack) task_mask_attack" in text
    assert "MASK_SEQ_BLOCKS" in text
```

- [ ] **Step 2: rodar e ver falhar**

Run: `python3 -m pytest cluster/tests/test_mask_sequential_scripts.py -q`
Expected: FAIL no primeiro `assert` (case ausente).

- [ ] **Step 3: implementar**

`job.sh`: normalizar `MASK_SEQ_BLOCKS="${MASK_SEQ_BLOCKS//+/,}"` no bloco `mask_*` e adicionar um case copiado de `mask_cascade_attack` com:

```
--blocks "${MASK_SEQ_BLOCKS:-5,10,25}" --z "${MASK_SEQ_Z:-1.96}" --loss "${MASK_SEQ_LOSS:-ci}"
--fpr "${MASK_CASCADE_FPR:-0.10}" --max-masks "${MASK_SEQ_MAX_MASKS:-100}"
--per-attack "${MASK_PER_ATTACK:-5000}"
--out "${MASK_RESULTS}/sequential_attack/${ATTACK}-b${MASK_SEQ_BLOCKS//,/-}.json"
```

`submit.sh`: `sequential-attack) task="mask_sequential_attack"` e, em `task_mask_attack`, exportar `MASK_SEQ_BLOCKS` (com `//,/+`), `MASK_SEQ_Z`, `MASK_SEQ_LOSS` quando definidas; adicionar as três linhas na `usage()`.

- [ ] **Step 4: rodar e ver passar**

Run: `python3 -m pytest cluster/tests/test_mask_sequential_scripts.py cluster/tests/test_run_mask_study.py -q`
Expected: PASS (a cadeia existente não muda).

- [ ] **Step 5: commit**

```bash
git add cluster/slurm/job.sh cluster/submit.sh cluster/tests/test_mask_sequential_scripts.py
git commit -m "feat: submit the sequential attack from the cluster scripts"
```

---

### Task 6: Smoke cronometrado e decisão de loss

**Files:** nenhum (verificação operacional).

- [ ] **Step 1: publicar e atualizar o cluster**

```bash
git push origin feat/apuana-cluster
ssh glp3@slurm-client1.cin.ufpe.br 'cd $HOME/Projects/replicating_RePO && git pull --ff-only && python3 -m pytest code/new_data_eval/tests/test_attack_loss.py code/mask_experiments/tests/test_sequential_attack.py -q'
```

Expected: os testes de TF rodam (2 passed) e os puros também.

- [ ] **Step 2: rodar o smoke de 20 amostras no DDoS, bloco 5**

```bash
ssh glp3@slurm-client1.cin.ufpe.br 'cd $HOME/Projects/replicating_RePO && \
  MASK_MAX_BENIGN=200 MASK_MAX_PER_ATTACK=50 MASK_PER_ATTACK=20 \
  bash cluster/submit.sh mask-exp sequential-attack DDoS'
```

Expected: `[submit] mask-exp sequential-attack: array 0-0%2 -> <jobid>`; leva ~10 min.

- [ ] **Step 3: inspecionar o JSON e cronometrar**

```bash
ssh glp3@slurm-client1.cin.ufpe.br 'python3 - <<"PY"
import json, pathlib
p = pathlib.Path.home()/"replicating-repo-results/mask_efficiency/sequential_attack/DDoS-b5.json"
d = json.loads(p.read_text())
print(d["configs"]["seq5"])
print(d["attacks"]["DDoS"]["seq5"])
print("runtime", d["runtime_seconds"], "per sample", d["runtime_seconds"]/d["dataset"]["attacks"]["DDoS"])
PY
```

Expected: `normal_tpr` > 0,4; `victim_tpr` ≤ `normal_tpr`; `early_stop_rate` > 0; nenhum `NaN`; `threshold` na casa de 0,05.

- [ ] **Step 4: decidir a loss e registrar**

Projeção da onda 1 = `t_por_amostra × 1000 × (1·Heartbleed + 1·Infiltration + 1·Hulk + 1·Slowhttptest + 1·GoldenEye + 1·DDoS + 2,5·slowloris + 1,5·FTP + 0,5·PortScan + 0,5·Bot + 0,5·SSH)`.
Se a projeção passar de **9 h**, submeter a onda 1 com `MASK_SEQ_LOSS=sum` e guardar `ci` para a onda 2. Anotar a projeção e a escolha em `docs/MASK_EFFICIENCY_SEQUENTIAL.md` (Task 7).

- [ ] **Step 5: commit (se houver ajuste de default)**

```bash
git add -A && git commit -m "chore: set the sequential attack loss from the smoke timing"
```

---

### Task 7: Submissão das duas ondas, consolidação e documento

**Files:**
- Create: `docs/MASK_EFFICIENCY_SEQUENTIAL.md`
- Modify: `docs/MASK_EFFICIENCY_CASCADE.md` (seção final "Decisão sequencial sob ataque", linkando o doc novo)

- [ ] **Step 1: liberar a GPU e submeter a onda 1 na ordem de prioridade**

```bash
scancel 32059_3 32059_4 32060
cat > $HOME/replicating-repo-results/mask_efficiency/attack-lists/seq-priority.txt <<'EOF'
Heartbleed
Infiltration
DoS Hulk
DoS Slowhttptest
DoS GoldenEye
DDoS
DoS slowloris
FTP-Patator
PortScan
Bot
SSH-Patator
EOF
cd $HOME/Projects/replicating_RePO && \
  MASK_SEQ_BLOCKS=5 MASK_PER_ATTACK=1000 \
  MASK_ATTACK_SET=$HOME/replicating-repo-results/mask_efficiency/attack-lists/seq-priority.txt \
  bash cluster/submit.sh mask-exp sequential-attack all
```

Expected: `[list] 11 ataques`, `array 0-10%2`; os ataques de TPR normal alta vêm primeiro (evadem rápido) e Heartbleed/Infiltration (n=11/36) saem em minutos.

- [ ] **Step 2: enfileirar a onda 2**

```bash
cd $HOME/Projects/replicating_RePO && \
  DEPENDENCY=afterany:<jobid_da_onda_1> MASK_SEQ_BLOCKS=10,25 MASK_PER_ATTACK=1000 \
  MASK_ATTACK_SET=$HOME/replicating-repo-results/mask_efficiency/attack-lists/seq-priority.txt \
  bash cluster/submit.sh mask-exp sequential-attack all
```

Expected: `array 0-10%2` em espera; grava `*-b10-25.json`, sem sobrescrever `*-b5.json`.

- [ ] **Step 3: acompanhar e registrar o que entrou**

```bash
ls -l $HOME/replicating-repo-results/mask_efficiency/sequential_attack/
squeue -u $USER -o "%.8i %.30j %.9T %.8M %.20R"
```

Regra de corte: às 06:00 de 19/11, consolidar **os ataques que existem** e declarar no documento quantos de 11 entraram. Não inventar valor para ataque ausente.

- [ ] **Step 4: merge e doc**

```bash
ssh glp3@slurm-client1.cin.ufpe.br 'cd $HOME/Projects/replicating_RePO && bash cluster/submit.sh mask-exp merge'
ls -l $HOME/replicating-repo-results/mask_efficiency/sequential_attack_all.json  # via ssh
```

Escrever `docs/MASK_EFFICIENCY_SEQUENTIAL.md` com:

- método e algoritmo (bloco, IC, `min_masks`, `max_masks`, o que o atacante otimiza, por que a escala dos thresholds é `block × τ`);
- tabela normal (bloco 5/10/25): TPR macro, FPR, máscaras/amostra no benigno e no ataque, comparada com RePO+ 100 (0,2560) e cascata cheap10 (0,2560; 60,8 máscaras);
- tabela adversarial por ataque: `normal_tpr → victim_tpr`, `early_stop_rate`, `fooled`, máscaras usadas, e as duas colunas de transferência (`transfer_full100_tpr`, `repo_plus_full100_tpr`) — é aqui que cascata e sequencial ficam lado a lado;
- leitura honesta: o sequencial gasta 5,32 máscaras/amostra, mas a parada precoce é um alvo; se o `early_stop_rate` for alto em algum ataque, esse é o preço da economia;
- limitações: 1.000/ataque, sem ataque ao *valor* das máscaras reveladas (só ao escore do primeiro bloco), Heartbleed/Infiltration com n pequeno.

- [ ] **Step 5: commit**

```bash
git add docs/MASK_EFFICIENCY_SEQUENTIAL.md docs/MASK_EFFICIENCY_CASCADE.md
git commit -m "docs: report the sequential rule under attack"
```

---

## Self-review deste plano

- **Cobertura do pedido:** ataque adaptativo aos 11 ataques (Task 3 + 7), mesmo volume do cascade (1.000/ataque, 3 configs nas duas ondas), métricas comparáveis com a base do artigo (transferência para 100 máscaras e RePO+ no mesmo JSON), rodar até quinta de manhã (ondas + ordem de prioridade + decisão de loss por medição).
- **O que não entra:** ataque ao *conjunto de máscaras* revelado (a regra usa máscaras iid fixas por fase; não há escolha de máscara a corromper), avaliação em pacotes, e escala 5.000 — este último é upgrade pós-entrega, como no `cascade_attack`.
- **Ponto aberto declarado:** o cancelamento de A/C é decisão sua e está isolado na seção de agenda.
