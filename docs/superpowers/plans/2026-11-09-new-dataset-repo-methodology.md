# New Dataset RePO Methodology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a reproducible local/Colab evaluation of the new dataset using the article's RePO/RePO+ protocol, with normal and adversarial results per attack class and no Apuana/SLURM work.

**Architecture:** A small Python package `code/new_data_eval/` separates data validation, scoring, attack, metrics, and result persistence. A CLI exposes `validate`, `normal`, `adversarial`, and `smoke`. Notebooks become thin wrappers over the same functions. A shell script extracts the LFS ZIP into `data/new_data/`.

**Tech Stack:** Python 3.10+, TensorFlow/Keras, NumPy, Pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-11-09-new-dataset-repo-methodology-design.md`

## Global Constraints

- Work only in the `feat/new-dataset-eval` worktree; do not modify `cluster/`.
- Package lives in `code/new_data_eval/` but is imported as `new_data_eval` with `PYTHONPATH=code`; never create `code/__init__.py` (it would shadow the stdlib `code` module).
- Dataset: `data/new_data/New-dataset-split-clean.zip`, containing `new_train.csv` and `new_test.csv`.
- Expected test columns: 85, with `column00` at index 0, `Dst Port` at index 5, flow features at indices 8–83, `Label` at index 84.
- Model input: exactly 77 features: `Dst Port` + columns 8–83.
- Labels: `BENIGN`, `xss`, `password`, `injection`, `scanning`, `backdoor`, `ransomware`, `mitm`, `ddos`, `dos`. Compare attack labels case-insensitively after `strip`.
- Protocol: 100 masks per sample, each feature masked with probability 0.75; RePO = mean of 100 errors; RePO+ = min per group of 20, sum of 5 minima.
- Normal evaluation: FPR target 0.01. Adversarial evaluation: FPR target 0.10.
- Seed: 2019.
- Attack loss: sum of 100 per-mask mean squared errors (RePO style). Success and victim TPR: RePO+ aggregate.
- Adversarial sampling: up to 5,000 random samples per attack; use all if fewer.
- Result model type: `new_data_flow`. Variants: `primary`, `sensitivity_first_n`, `sensitivity_paper_0.8`.
- Do not modify `results/flow_based_adversarial_results.json` or `results/packet_based_adversarial_results.json`.
- Do not commit extracted CSVs or `.npz` score files.
- Write JSON atomically; one file per run, never append.
- Target environment for the final run: Python 3.10, TensorFlow 2.21, Keras 3.12.4, NumPy 2.2.6, Pandas 2.2.3. Record the actual versions in every result.

## Review Focus

- A label with different casing, surrounding whitespace, or an unexpected attack name must fail validation with a useful message, never silently produce an empty class.
- A CSV with the wrong column count, wrong column order, or `Label` not last must abort before any scoring.
- Rows containing `inf` or `nan` must be dropped and counted, not crash the run or silently change class counts.
- Threshold calculation must be deterministic with tied scores and small benign sets, and must never divide by zero.
- Changing `--batch-size` or resuming from a checkpoint must not change per-sample scores or duplicate/lose samples.

---

### Task 1: Extract and validate the dataset

**Files:**
- Create: `extract_new_data.sh`
- Create: `code/new_data_eval/__init__.py`
- Create: `code/new_data_eval/data.py`
- Create: `code/new_data_eval/tests/__init__.py`
- Create: `code/new_data_eval/tests/conftest.py`
- Create: `code/new_data_eval/tests/test_data.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `EXPECTED_LABELS: tuple[str, ...]`
  - `ATTACK_LABELS: tuple[str, ...]`
  - `EXPECTED_COLUMNS: tuple[str, ...]` with 85 entries
  - `FEATURE_INDICES: tuple[int, ...]` with 77 entries
  - `N_FEATURES = 77`
  - `LABEL_INDEX = 84`
  - `DatasetInfo` dataclass with `train_sha256`, `test_sha256`, `train_rows`, `test_rows`, `test_valid_rows`, `test_dropped_invalid`, `class_counts`
  - `canonical_label(value: object) -> str`
  - `validate_header(columns: Sequence[str]) -> None`
  - `load_metadata(meta_dir: Path) -> tuple[np.ndarray, np.ndarray]`
  - `normalize_features(x: np.ndarray, train_min: np.ndarray, train_max: np.ndarray) -> np.ndarray`
  - `feature_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
  - `iter_test_chunks(csv_path: Path, chunksize: int = 200_000) -> Iterator[tuple[np.ndarray, np.ndarray]]`
  - `count_classes(csv_path: Path, chunksize: int = 200_000) -> dict[str, int]`
  - `select_attack_samples(csv_path: Path, attack: str, max_samples: int, mode: str, seed: int, chunksize: int = 200_000) -> tuple[np.ndarray, int]`
  - `validate_dataset(train_csv: Path, test_csv: Path, meta_dir: Path) -> DatasetInfo`

- [ ] **Step 1: Write the failing tests**

```python
def test_canonical_label_normalizes_benign_and_attack():
    assert canonical_label("  benign ") == "BENIGN"
    assert canonical_label("DDoS") == "ddos"
    assert canonical_label(" xss ") == "xss"


def test_validate_header_rejects_wrong_column_count():
    with pytest.raises(ValueError, match="85"):
        validate_header(["column00", "Flow ID", "Label"])


def test_validate_header_rejects_wrong_column_order():
    columns = list(EXPECTED_COLUMNS)
    columns[5], columns[6] = columns[6], columns[5]
    with pytest.raises(ValueError, match="column"):
        validate_header(columns)


def test_feature_selection_returns_77_columns(synthetic_project):
    frame = pd.read_csv(synthetic_project.test_csv, nrows=1)
    x, labels, valid = feature_frame(frame)
    assert x.shape == (1, 77)
    assert labels[0] == "BENIGN"
    assert valid[0]


def test_invalid_rows_are_dropped_and_counted(synthetic_project):
    info = validate_dataset(
        synthetic_project.train_csv,
        synthetic_project.test_csv,
        synthetic_project.meta_dir,
    )
    assert info.test_dropped_invalid == 2


def test_validate_dataset_rejects_unknown_label(synthetic_project):
    with pytest.raises(ValueError, match="not_an_attack"):
        validate_dataset(
            synthetic_project.train_csv,
            synthetic_project.unknown_label_csv,
            synthetic_project.meta_dir,
        )


def test_select_attack_samples_is_seeded(synthetic_project):
    a, n_a = select_attack_samples(
        synthetic_project.test_csv, "ddos", max_samples=2,
        mode="random", seed=2019,
    )
    b, n_b = select_attack_samples(
        synthetic_project.test_csv, "ddos", max_samples=2,
        mode="random", seed=2019,
    )
    assert n_a == n_b
    np.testing.assert_allclose(a, b)
```

`synthetic_project` must be a fixture that creates:

- `train_csv` with only `BENIGN`;
- `test_csv` with all 9 attack labels plus `BENIGN`, and two rows containing `inf`/`nan` in the feature range;
- `unknown_label_csv` with the same header but one `not_an_attack` label;
- `meta_dir` with 77-element `train_min.npy` and `train_max.npy`;
- `model_dir` with a tiny SavedModel whose input is `(None, 77)`;
- `results_dir`;
- `normal_repo_plus_json` with `thresholds.fpr_0.10 = 0.123`.

Add the fixture to `conftest.py`; it must also add the repository root to `sys.path`. Add shared fixtures `tiny_model` (a tiny Keras model with 77 inputs) and `meta` (a pair of 77-element normalization arrays) to the same `conftest.py`; Tasks 3 and 6 use them.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest code/new_data_eval/tests/test_data.py -v`  
Expected: FAIL because `new_data_eval.data` does not exist.

- [ ] **Step 3: Implement the data module and extraction script**

`extract_new_data.sh` must:

- fail with a clear message if `git lfs` is unavailable;
- run `git lfs pull --include="data/new_data/New-dataset-split-clean.zip"`;
- unzip into `data/new_data/`;
- verify `new_train.csv` and `new_test.csv` exist and are not LFS pointers;
- print the row count and SHA-256 of each file.

`data.py` must:

- define `EXPECTED_COLUMNS` as the exact 85 names from the `new_test.csv` header, in order (the header is the canonical list);
- select `Dst Port` plus columns 8–83;
- normalize only for scoring, not while loading;
- drop rows with non-finite values in the 77 features and count them;
- reject unknown labels instead of silently ignoring them;
- implement reservoir sampling for `mode="random"` and prefix selection for `mode="first"`.

Add `pytest` to `requirements.txt`. Add `scikit-learn` and `tqdm` only if `preprocess_new_data.py` is kept executable.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest code/new_data_eval/tests/test_data.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extract_new_data.sh code/new_data_eval/__init__.py code/new_data_eval/data.py \
  code/new_data_eval/tests requirements.txt
git commit -m "feat: add new dataset extraction and validation"
```

---

### Task 2: Implement scoring primitives and thresholds

**Files:**
- Create: `code/new_data_eval/scoring.py`
- Create: `code/new_data_eval/tests/test_scoring.py`

**Interfaces:**
- Consumes: `N_FEATURES` from `data.py`.
- Produces:
  - `PHASE_ORIGINAL = "original"`, `PHASE_ATTACK = "attack"`, `PHASE_VICTIM = "victim"`
  - `repo_scores(errors: np.ndarray) -> np.ndarray`
  - `repo_plus_scores(errors: np.ndarray) -> np.ndarray`
  - `score_from_errors(errors: np.ndarray, mode: str) -> np.ndarray`
  - `threshold_at_fpr(benign_scores: np.ndarray, fpr: float, method: str = "ceil") -> float`

- [ ] **Step 1: Write the failing tests**

```python
def test_repo_score_is_mean_of_mask_errors():
    errors = np.array([[1.0, 3.0], [2.0, 4.0]])
    np.testing.assert_allclose(repo_scores(errors), [2.0, 3.0])


def test_repo_plus_uses_five_groups_of_twenty():
    errors = np.arange(100, dtype=float).reshape(1, 100)
    expected = sum(np.min(errors.reshape(1, 5, 20), axis=2).ravel())
    assert repo_plus_scores(errors)[0] == expected


def test_threshold_at_fpr_uses_ceil_and_is_deterministic_with_ties():
    scores = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
    assert threshold_at_fpr(scores, 0.3) == 3.0
    assert threshold_at_fpr(scores, 0.3) == threshold_at_fpr(scores, 0.3)


def test_threshold_at_fpr_rejects_empty_benign():
    with pytest.raises(ValueError, match="empty"):
        threshold_at_fpr(np.array([]), 0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest code/new_data_eval/tests/test_scoring.py -v`  
Expected: FAIL because `new_data_eval.scoring` does not exist.

- [ ] **Step 3: Implement the scoring primitives**

Copy the threshold semantics from `origin/feat/apuana-cluster:cluster/metrics.py` (`threshold_at_fpr`, method `ceil`) so the new results match the Apuana protocol. `repo_plus_scores` must split the second axis into 5 equal groups and require the number of masks to be divisible by 5. Keep these functions NumPy-only; TensorFlow is not needed for this task.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest code/new_data_eval/tests/test_scoring.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/new_data_eval/scoring.py code/new_data_eval/tests/test_scoring.py
git commit -m "feat: add RePO scoring primitives and threshold calibration"
```

---

### Task 3: Add model loading and mask-based scoring

**Files:**
- Modify: `code/new_data_eval/scoring.py`
- Create: `code/new_data_eval/tests/test_model_scoring.py`

**Interfaces:**
- Consumes: `normalize_features`, `load_metadata`, `N_FEATURES` from `data.py`; `repo_scores`, `repo_plus_scores` from Task 2.
- Produces:
  - `load_model(model_dir: Path) -> Any`
  - `validate_model(model: Any, train_min: np.ndarray, train_max: np.ndarray, benign_sample: np.ndarray, seed: int = 2019) -> dict[str, Any]`
  - `reconstruction_errors(model: Any, x_raw: np.ndarray, train_min: np.ndarray, train_max: np.ndarray, *, n_masks: int = 100, batch_size: int = 100, seed: int = 2019, phase: str = PHASE_ORIGINAL, sample_indices: Sequence[int] | None = None, score_repeats: int = 1, paired_seed: int | None = None) -> np.ndarray`
  - `score_samples(model: Any, x_raw: np.ndarray, train_min: np.ndarray, train_max: np.ndarray, *, mode: str, n_masks: int = 100, batch_size: int = 100, seed: int = 2019, phase: str = PHASE_ORIGINAL, sample_indices: Sequence[int] | None = None, score_repeats: int = 1, paired_seed: int | None = None) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

```python
def test_reconstruction_errors_shape_and_finite(tiny_model, meta):
    x = np.zeros((3, 77), dtype=np.float32)
    errors = reconstruction_errors(tiny_model, x, *meta, n_masks=10, batch_size=2)
    assert errors.shape == (3, 10)
    assert np.isfinite(errors).all()


def test_scores_are_batch_size_invariant(tiny_model, meta):
    x = np.random.default_rng(0).random((8, 77)).astype(np.float32)
    a = score_samples(tiny_model, x, *meta, mode="repo_plus", n_masks=20,
                      batch_size=2, seed=2019)
    b = score_samples(tiny_model, x, *meta, mode="repo_plus", n_masks=20,
                      batch_size=8, seed=2019)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_validate_model_rejects_wrong_input_width(tiny_model, meta):
    with pytest.raises(ValueError, match="77"):
        validate_model(tiny_model, *meta, benign_sample=np.zeros((2, 76)),
                       seed=2019)
```

Use a tiny Keras model with 77 inputs that returns a deterministic reconstruction (for example, a Dense layer with identity initialization or a model that returns its input). `meta` is a pair of 77-element arrays.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest code/new_data_eval/tests/test_model_scoring.py -v`  
Expected: FAIL because the functions are missing.

- [ ] **Step 3: Implement model loading and per-sample stateless masks**

Use `tf.keras.models.load_model` through the existing `code/load_model_compat.py` helper if needed. Generate masks with a stateless RNG seeded from `(seed, sample_index, phase, repeat)`; `--batch-size` must not change the masks or scores. Do not use a single mask set for the whole batch.

`validate_model` must check:

- model accepts `(None, 77)`;
- outputs are finite;
- reconstruction error on the benign sample is lower than on uniform random noise.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest code/new_data_eval/tests/test_model_scoring.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/new_data_eval/scoring.py code/new_data_eval/tests/test_model_scoring.py
git commit -m "feat: add model loading and batch-invariant RePO scoring"
```

---

### Task 4: Implement metrics and result persistence

**Files:**
- Create: `code/new_data_eval/metrics.py`
- Create: `code/new_data_eval/results.py`
- Create: `code/new_data_eval/tests/test_metrics.py`
- Create: `code/new_data_eval/tests/test_results.py`

**Interfaces:**
- Consumes: `EXPECTED_LABELS`, `ATTACK_LABELS` from `data.py`.
- Produces:
  - `fpr(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float`
  - `class_rates(scores: np.ndarray, labels: np.ndarray, threshold: float, expected_labels: Sequence[str]) -> dict[str, dict[str, Any]]`
  - `average_attack_tpr(rates: Mapping[str, Mapping[str, Any]], *, low_n_threshold: int = 100) -> dict[str, float]`
  - `conditional_drop(victim_tpr: float, normal_tpr: float) -> float | None`
  - `write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None`
  - `file_sha256(path: Path) -> str`
  - `environment_manifest() -> dict[str, Any]`
  - `config_hash(config: Mapping[str, Any]) -> str`
  - `git_commit() -> str | None`
  - `build_normal_result(**kwargs: Any) -> dict[str, Any]`
  - `build_adversarial_result(**kwargs: Any) -> dict[str, Any]`

- [ ] **Step 1: Write the failing tests**

```python
def test_class_rates_include_counts_and_low_n_flag():
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array(["BENIGN", "ddos", "ddos", "mitm"])
    rates = class_rates(scores, labels, threshold=0.5,
                        expected_labels=["BENIGN", "ddos", "mitm"])
    assert rates["ddos"] == {"n": 2, "tpr": 0.5, "low_n": True}
    assert rates["mitm"]["tpr"] == 0.0


def test_average_tpr_excludes_benign_and_reports_weighted():
    rates = {
        "BENIGN": {"n": 100, "fpr": 0.01},
        "ddos": {"n": 3, "tpr": 1.0, "low_n": True},
        "xss": {"n": 1, "tpr": 0.0, "low_n": True},
    }
    averages = average_attack_tpr(rates, low_n_threshold=100)
    assert averages["macro"] == 0.5
    assert averages["weighted"] == 0.75


def test_conditional_drop_returns_none_for_zero_normal_tpr():
    assert conditional_drop(0.0, 0.0) is None


def test_write_json_atomic_replaces_file(tmp_path):
    path = tmp_path / "result.json"
    write_json_atomic(path, {"a": 1})
    write_json_atomic(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 2}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest code/new_data_eval/tests/test_metrics.py code/new_data_eval/tests/test_results.py -v`  
Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement metrics and result builders**

`class_rates` must include every expected label, with `n` and `tpr` for attacks and `fpr` for `BENIGN`. `average_attack_tpr` must ignore `BENIGN`, compute macro over all 9 attacks, weighted by `n`, and `macro_excluding_low_n` excluding `n < low_n_threshold`.

`build_normal_result` and `build_adversarial_result` must produce the schema fields from the spec. `config_hash` must be stable for equal configs and change when seed, thresholds, sampling, model hash, or data hash changes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest code/new_data_eval/tests/test_metrics.py code/new_data_eval/tests/test_results.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/new_data_eval/metrics.py code/new_data_eval/results.py \
  code/new_data_eval/tests/test_metrics.py code/new_data_eval/tests/test_results.py
git commit -m "feat: add evaluation metrics and atomic result schema"
```

---

### Task 5: Build the CLI and normal evaluation

**Files:**
- Create: `code/new_data_eval/cli.py`
- Create: `code/new_data_eval/__main__.py`
- Create: `code/new_data_eval/tests/test_cli_normal.py`
- Modify: `code/new_data_eval/data.py` if streaming aggregation needs a helper
- Modify: `requirements.txt` if pytest is not already present

**Interfaces:**
- Consumes: all functions from Tasks 1–4.
- Produces:
  - `build_parser() -> argparse.ArgumentParser`
  - `main(argv: Sequence[str] | None = None) -> int`
  - `run_validate(args: argparse.Namespace) -> int`
  - `run_normal(args: argparse.Namespace) -> int`

- [ ] **Step 1: Write the failing tests**

```python
def test_normal_cli_writes_both_results(synthetic_project):
    exit_code = main([
        "normal",
        "--data-dir", str(synthetic_project.data_dir),
        "--model-dir", str(synthetic_project.model_dir),
        "--meta-dir", str(synthetic_project.meta_dir),
        "--results-dir", str(synthetic_project.results_dir),
        "--modes", "repo,repo_plus",
        "--n-masks", "5",
        "--batch-size", "2",
    ])
    assert exit_code == 0
    for mode in ("repo", "repo_plus"):
        payload = json.loads(
            (synthetic_project.results_dir / "normal" / f"{mode}.json").read_text()
        )
        assert payload["experiment"] == "normal"
        assert payload["model_type"] == "new_data_flow"
        assert "thresholds" in payload
        assert "class_rates" in payload


def test_validate_cli_rejects_missing_model(synthetic_project):
    exit_code = main([
        "validate",
        "--data-dir", str(synthetic_project.data_dir),
        "--model-dir", str(synthetic_project.data_dir / "missing"),
        "--meta-dir", str(synthetic_project.meta_dir),
    ])
    assert exit_code != 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest code/new_data_eval/tests/test_cli_normal.py -v`  
Expected: FAIL because `main` does not exist.

- [ ] **Step 3: Implement `validate` and `normal`**

`validate` must run header/label/metadata/model checks and print a concise summary, including the indices of constant features (`max == min`) reported by `constant_feature_indices`. `normal` must:

- stream `new_test.csv` in chunks and never convert the whole CSV to float64;
- pass the global sample offset as `sample_indices` to `score_samples` so masks do not restart at zero in every chunk;
- score `repo` and `repo_plus`;
- compute thresholds 0.01 and 0.10 on `BENIGN`;
- write `results/new_data/normal/repo.json` and `repo_plus.json` (the default
  `--results-dir` is `results/new_data`);
- add `scoring`, `variant="primary"`, hashes, environment, and git commit.

Use `--n-masks` only for tests/smoke; the default remains 100. Accept
`--max-samples` for smoke runs and stop reading once that many valid samples
have been collected; when omitted, evaluate the full test set.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest code/new_data_eval/tests/test_cli_normal.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/new_data_eval/cli.py code/new_data_eval/__main__.py \
  code/new_data_eval/tests/test_cli_normal.py
git commit -m "feat: add validate and normal evaluation CLI"
```

---

### Task 6: Port the adversarial flow attack

**Files:**
- Create: `code/new_data_eval/attack.py`
- Create: `code/new_data_eval/tests/test_attack.py`

**Interfaces:**
- Consumes: `load_model`, `score_samples`, `reconstruction_errors`, `PHASE_*` from `scoring.py`; `repo_plus_scores` from Task 2.
- Uses `score_samples` for all success/victim scores, because it averages per-repeat RePO+ scores. `reconstruction_errors` is only for single-repeat diagnostics; do not combine its repeat-averaged matrix with `repo_plus_scores` when `score_repeats > 1`.
- Produces:
  - `AttackOutcome` dataclass with `kind: Literal["evaded", "backup", "no_change", "not_fooled"]`, `x_adv: np.ndarray | None`, `score: float | None`
  - `FlowAttack` with:
    - `__init__(model, train_min, train_max, decimal_features, num_input, real_threshold, attack_threshold, *, seed: int, n_masks: int = 100)`; require `n_masks % 5 == 0` and use `group_size = n_masks // 5`.
    - `score(sample: np.ndarray, sample_index: int, phase: str) -> float`
    - `find_adv(sample: np.ndarray, sample_index: int) -> AttackOutcome`
    - `optimizer_state() -> dict[str, np.ndarray]`
    - `load_optimizer_state(state: Mapping[str, np.ndarray]) -> None`

- [ ] **Step 1: Write the failing tests**

```python
def test_masks_have_expected_size_and_zeroed_features(tiny_model, meta):
    attack = FlowAttack(tiny_model, *meta, decimal_features=[], num_input=77,
                        real_threshold=0.1, attack_threshold=0.1, seed=2019)
    assert attack.mask_l1.shape == (77,)
    assert attack.mask_l1[0] == 0
    assert attack.mask_l3[76] == 1


def test_find_adv_returns_a_known_outcome(tiny_model, meta):
    attack = FlowAttack(tiny_model, *meta, decimal_features=[], num_input=77,
                        real_threshold=0.10, attack_threshold=0.10, seed=2019)
    outcome = attack.find_adv(np.zeros(77, dtype=np.float32), sample_index=0)
    assert outcome.kind in {"evaded", "backup", "no_change", "not_fooled"}
    if outcome.x_adv is not None:
        assert outcome.x_adv.shape == (77,)
        assert np.isfinite(outcome.x_adv).all()


def test_optimizer_state_round_trip(tiny_model, meta):
    attack = FlowAttack(tiny_model, *meta, decimal_features=[], num_input=77,
                        real_threshold=0.1, attack_threshold=0.1, seed=2019)
    state = attack.optimizer_state()
    attack.load_optimizer_state(state)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest code/new_data_eval/tests/test_attack.py -v`  
Expected: FAIL because `new_data_eval.attack` does not exist.

- [ ] **Step 3: Port `FlowAttack` from the reference runner**

Copy the class from `git show origin/feat/apuana-cluster:cluster/repo_runner.py` (the `FlowAttack` section) and preserve:

- `fixed_mask`, `mask_l1`, `mask_l2`, `mask_l3`;
- duplicate-feature map `61→2`, `63→3`, `62→4`, `64→5`;
- `mins`, `maxs`, `flags`, `aggr_features`;
- Adam learning rates 0.001, 0.01, 0.1, 1.0;
- 30/40/50/60 iterations;
- dependent-feature recomputation formulas;
- loss = sum of 100 per-mask mean squared errors.

Change only:

- RNG seeding: per-sample, stateless, derived from `(seed, sample_index, iteration)`;
- return type: classify `evaded`, `backup`, `no_change`, `not_fooled`;
- `score` takes `sample_index` and `phase` for deterministic masks.

Validate `num_input == 77` and every mask index `< num_input`; raise `ValueError` otherwise.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest code/new_data_eval/tests/test_attack.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/new_data_eval/attack.py code/new_data_eval/tests/test_attack.py
git commit -m "feat: port adversarial flow attack with deterministic masks"
```

---

### Task 7: Add adversarial CLI, sampling, checkpoints, and variants

**Files:**
- Modify: `code/new_data_eval/cli.py`
- Modify: `code/new_data_eval/data.py` if sampling helper is not already there
- Create: `code/new_data_eval/tests/test_cli_adversarial.py`

**Interfaces:**
- Consumes: `FlowAttack`, `select_attack_samples`, result builders, normal result JSON.
- Produces:
  - `run_adversarial(args: argparse.Namespace) -> int`
  - `run_smoke(args: argparse.Namespace) -> int`
  - `load_normal_threshold(results_dir: Path) -> float`
  - `checkpoint_path(results_dir: Path, variant: str, attack: str) -> Path`
  - `result_path(results_dir: Path, variant: str, attack: str) -> Path`

- [ ] **Step 1: Write the failing tests**

```python
def test_adversarial_reads_threshold_from_normal_json(synthetic_project):
    threshold = load_normal_threshold(synthetic_project.results_dir)
    assert threshold == 0.123


def test_adversarial_rejects_unknown_label(synthetic_project):
    exit_code = main([
        "adversarial",
        "--data-dir", str(synthetic_project.data_dir),
        "--model-dir", str(synthetic_project.model_dir),
        "--meta-dir", str(synthetic_project.meta_dir),
        "--results-dir", str(synthetic_project.results_dir),
        "--attack", "not_an_attack",
        "--n-masks", "5",
    ])
    assert exit_code != 0
    assert not list((synthetic_project.results_dir / "adversarial").rglob("*.json"))


def test_attack_label_is_case_insensitive(synthetic_project):
    exit_code = main([
        "adversarial",
        "--data-dir", str(synthetic_project.data_dir),
        "--model-dir", str(synthetic_project.model_dir),
        "--meta-dir", str(synthetic_project.meta_dir),
        "--results-dir", str(synthetic_project.results_dir),
        "--attack", "DDoS",
        "--max-samples", "1",
        "--n-masks", "5",
        "--variant", "primary",
    ])
    assert exit_code == 0
    assert (synthetic_project.results_dir / "adversarial" / "primary" / "ddos.json").exists()


def test_adversarial_writes_variant_directory(synthetic_project):
    exit_code = main([
        "adversarial",
        "--data-dir", str(synthetic_project.data_dir),
        "--model-dir", str(synthetic_project.model_dir),
        "--meta-dir", str(synthetic_project.meta_dir),
        "--results-dir", str(synthetic_project.results_dir),
        "--attack", "ddos",
        "--max-samples", "2",
        "--n-masks", "5",
        "--variant", "sensitivity_first_n",
    ])
    assert exit_code == 0
    assert (synthetic_project.results_dir / "adversarial" / "sensitivity_first_n" / "ddos.json").exists()
    assert not (synthetic_project.results_dir / "adversarial" / "primary" / "ddos.json").exists()


def test_checkpoint_resume_does_not_duplicate_samples(synthetic_project):
    args = [
        "adversarial",
        "--data-dir", str(synthetic_project.data_dir),
        "--model-dir", str(synthetic_project.model_dir),
        "--meta-dir", str(synthetic_project.meta_dir),
        "--results-dir", str(synthetic_project.results_dir),
        "--attack", "ddos",
        "--max-samples", "2",
        "--n-masks", "5",
        "--checkpoint-every", "1",
    ]
    assert main(args) == 0
    result_path = synthetic_project.results_dir / "adversarial" / "primary" / "ddos.json"
    first = json.loads(result_path.read_text())
    assert first["counts"]["n"] == 2

    result_path.unlink()
    assert main(args) == 0
    second = json.loads(result_path.read_text())
    assert second["counts"]["n"] == 2
    assert second["counts"] == first["counts"]


def test_smoke_is_isolated_from_primary(synthetic_project):
    exit_code = main([
        "smoke",
        "--data-dir", str(synthetic_project.data_dir),
        "--model-dir", str(synthetic_project.model_dir),
        "--meta-dir", str(synthetic_project.meta_dir),
        "--results-dir", str(synthetic_project.results_dir),
        "--attack", "ddos",
    ])
    assert exit_code == 0
    assert (synthetic_project.results_dir / "smoke" / "normal" / "repo_plus.json").exists()
    assert (synthetic_project.results_dir / "smoke" / "adversarial" / "ddos.json").exists()
    assert not (synthetic_project.results_dir / "adversarial" / "primary").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest code/new_data_eval/tests/test_cli_adversarial.py -v`  
Expected: FAIL because `run_adversarial` does not exist.

- [ ] **Step 3: Implement adversarial execution**

`adversarial` must:

- accept `--attack all` or a canonical attack label; compare case-insensitively but write the canonical lowercase name;
- select random samples with reservoir sampling (`--sample-mode random`, default) or prefix (`first`);
- read `thresholds.fpr_0.10` from `results/new_data/normal/repo_plus.json`; if missing, recompute and log a warning;
- set `attack_threshold` to `real_threshold` for `calibrated`, or `0.8 * real_threshold` for `paper_0.8`;
- run `FlowAttack` per sample and classify outcomes;
- save checkpoint every 50 samples with index, scores, counts, and optimizer state;
- resume only when `config_hash` matches; otherwise require `--force`;
- write `results/new_data/adversarial/<variant>/<attack>.json` (default
  `--results-dir` is `results/new_data`);
- write `.npz` scores under `results/new_data/scores/`;
- never write to flow/packet result files.

`smoke` must run validate, normal with `--max-samples 20 --n-masks 5`, and
adversarial for one attack with `--max-samples 5 --n-masks 5`, writing only
under `results/new_data/smoke/` so it cannot overwrite `primary`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest code/new_data_eval/tests/test_cli_adversarial.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/new_data_eval/cli.py code/new_data_eval/data.py \
  code/new_data_eval/tests/test_cli_adversarial.py
git commit -m "feat: add adversarial CLI with checkpoints and variants"
```

---

### Task 8: Convert notebooks to thin wrappers and add documentation

**Files:**
- Modify: `code/new_data_RePO_Normal_Test.ipynb`
- Modify: `code/new_data_RePO_Adversarial.ipynb`
- Modify: `code/train_new_data_model.py`
- Create: `docs/NEW_DATASET.md`
- Create: `code/new_data_eval/tests/test_docs_and_notebooks.py`
- Modify: `README.md`
- Modify: `results/.gitignore` or root `.gitignore` for `results/new_data/scores/`
- Modify: `docs/PROJECT_TASKS.md` section 5 after results exist

**Interfaces:**
- Consumes: `new_data_eval.cli.main` and the result paths from Tasks 5 and 7.
- Produces: no new runtime API.

- [ ] **Step 1: Write the failing checks**

```python
def test_new_dataset_doc_has_required_sections():
    text = Path("docs/NEW_DATASET.md").read_text()
    for section in ("Fonte e citação", "Split", "Protocolo", "Comandos"):
        assert section in text


def test_score_files_are_gitignored():
    output = subprocess.run(
        ["git", "check-ignore", "results/new_data/scores/example.npz"],
        capture_output=True, text=True, check=False,
    )
    assert output.returncode == 0


def test_normal_notebook_uses_package_entrypoint():
    notebook = json.loads(
        Path("code/new_data_RePO_Normal_Test.ipynb").read_text()
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "new_data_eval" in source
```

- [ ] **Step 2: Run checks to verify they fail**

Run: `python -m pytest code/new_data_eval/tests/test_docs_and_notebooks.py -v`  
Expected: FAIL because the docs and notebook wrappers do not exist.

- [ ] **Step 3: Replace duplicated notebook logic with calls**

The normal notebook must call the normal CLI or the underlying functions; it must not implement its own scoring, threshold, or metric logic. The adversarial notebook must call the adversarial CLI or functions and must not contain a hardcoded threshold, `attack_type`, or result path. Keep the notebooks useful for inspection on a small subset (`--max-samples`, `--n-masks` for smoke only).

`train_new_data_model.py` must accept `--train-csv`, `--output-dir`, `--seed`, `--epochs`, `--batch-size`; set `tf.keras.utils.set_random_seed(seed)`; assert every label is `BENIGN`; and write `training_manifest.json` with data hash, seed, versions, epochs, final loss, and shape.

`docs/NEW_DATASET.md` must document source (marked pending citation until confirmed), ZIP contents, split rule, class counts generated by the validator, features, constant-feature indices, protocol, commands, and limitations.

- [ ] **Step 4: Run checks to verify they pass**

Run: `python -m pytest code/new_data_eval/tests/test_docs_and_notebooks.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/new_data_RePO_Normal_Test.ipynb code/new_data_RePO_Adversarial.ipynb \
  code/train_new_data_model.py docs/NEW_DATASET.md README.md .gitignore \
  code/new_data_eval/tests/test_docs_and_notebooks.py
git commit -m "docs: wire notebooks and document new dataset methodology"
```

---

### Task 9: End-to-end verification

**Files:**
- No new source files; update docs/results after successful runs.
- Modify: `docs/NEW_DATASET.md` with the first real counts and smoke results.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: verified commands and initial result artifacts.

- [ ] **Step 1: Extract and validate**

Run:

```bash
bash extract_new_data.sh
PYTHONPATH=code python -m new_data_eval validate
```

Expected: both CSVs exist; validation prints 85 columns, 77 features, all 9 attacks, model input 77, and metadata shape 77.

- [ ] **Step 2: Run the test suite**

Run:

```bash
python -m pytest code/new_data_eval/tests -v
```

Expected: PASS.

- [ ] **Step 3: Run normal smoke and adversarial smoke**

Run:

```bash
PYTHONPATH=code python -m new_data_eval normal --modes repo,repo_plus --n-masks 5 --batch-size 2 --max-samples 50
PYTHONPATH=code python -m new_data_eval adversarial --attack ddos --max-samples 5 --n-masks 5 --variant primary
```

Expected: JSON files exist; adversarial JSON has `model_type="new_data_flow"`, `variant="primary"`, thresholds from the normal JSON, and no changes to flow/packet result files.

- [ ] **Step 4: Run the real normal evaluation**

Run:

```bash
PYTHONPATH=code python -m new_data_eval normal --modes repo,repo_plus
```

Expected: `results/new_data/normal/repo.json` and `repo_plus.json` with thresholds, FPR, TPR per class, counts, averages, and runtime. Record the thresholds and class counts in `docs/NEW_DATASET.md`.

- [ ] **Step 5: Run one full attack and then all attacks**

Run:

```bash
PYTHONPATH=code python -m new_data_eval adversarial --attack ddos --variant primary
PYTHONPATH=code python -m new_data_eval adversarial --attack all --variant primary
```

Expected: one JSON per attack; checkpoint files appear; an interrupted run resumes without duplicating samples. Do not run all attacks until the single-attack result has been inspected.

- [ ] **Step 6: Run sensitivities if time allows**

Run:

```bash
PYTHONPATH=code python -m new_data_eval adversarial --attack all --variant sensitivity_first_n --sample-mode first
PYTHONPATH=code python -m new_data_eval adversarial --attack all --variant sensitivity_paper_0.8 --attack-threshold-policy paper_0.8
```

Expected: results in separate directories; `primary` is untouched.

- [ ] **Step 7: Update checklist and commit**

Mark section 5 of `docs/PROJECT_TASKS.md` only for items actually completed. Update `docs/NEW_DATASET.md` with result tables. Commit results JSON and docs; do not commit `.npz` or CSVs.

```bash
git add results/new_data docs/NEW_DATASET.md docs/PROJECT_TASKS.md
git commit -m "docs: record new dataset evaluation results"
```

---

### Task 10: Final review

**Files:**
- No code changes expected.

**Interfaces:**
- Consumes: all tasks.
- Produces: a final verification report.

- [ ] **Step 1: Check result isolation**

Run:

```bash
git status --short
git diff -- results/flow_based_adversarial_results.json results/packet_based_adversarial_results.json
```

Expected: no diff in the flow/packet result files; no extracted CSV or `.npz` staged.

- [ ] **Step 2: Check schema consistency**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
for path in Path("results/new_data").rglob("*.json"):
    data = json.loads(path.read_text())
    assert data["model_type"] == "new_data_flow", path
    assert data["variant"] in {"primary", "sensitivity_first_n", "sensitivity_paper_0.8"}, path
print("ok")
PY
```

Expected: `ok`.

- [ ] **Step 3: Check documentation**

Run:

```bash
grep -n "Fonte e citação\|Split\|Protocolo\|Comandos" docs/NEW_DATASET.md
```

Expected: all four sections present.

- [ ] **Step 4: Commit any final fixes**

```bash
git add docs/NEW_DATASET.md results/new_data
git commit -m "chore: finalize new dataset evaluation"
```
