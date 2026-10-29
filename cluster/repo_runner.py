#!/usr/bin/env python3
"""Runner de avaliação da replicação do RePO para o cluster Apuana.

Subcomandos:
  normal       avalia RePO e/ou RePO+ no conjunto de teste
  adversarial  executa um ataque adversarial com checkpoint/retomada
  smoke        valida dados, modelo e GPU com poucas amostras
  manifest     imprime o manifesto do ambiente

O runner preserva as fórmulas dos notebooks de referência:
  * fluxo: 77 features, 100 máscaras, 5 grupos de 20, mínimo por grupo e soma;
  * pacotes: janelas de 20 pacotes, 29 features, mesma agregação;
  * RePO: média dos erros das 100 máscaras;
  * RePO+: mínimo por grupo e soma dos mínimos.

Diferenças deliberadas em relação aos notebooks:
  * o notebook normal de fluxo pontua dados sem normalizar (bug); aqui a
    normalização é sempre aplicada antes do modelo;
  * o resultado adversarial de pacotes passa a registrar métricas com
    semântica explícita (TPR antes/depois, taxa de sucesso do ataque);
  * seeds são fixas para permitir retomada e repetição.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

CLUSTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = CLUSTER_DIR.parent
if str(CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_DIR))
if str(REPO_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "code"))

import metrics  # noqa: E402

LOG = logging.getLogger("repo-runner")
_TF = None
_STOP = False


def tf():
    global _TF
    if _TF is None:
        import tensorflow as tensorflow

        _TF = tensorflow
    return _TF


# ---------------------------------------------------------------------------
# Infrastructure helpers
# ---------------------------------------------------------------------------


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOG.addHandler(stream)
    LOG.addHandler(file_handler)


def install_signal_handlers() -> None:
    def handler(signum, _frame):
        global _STOP
        _STOP = True
        LOG.warning("sinal %s recebido; checkpoint será salvo no próximo passo", signum)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def stop_requested() -> bool:
    return _STOP


def configure_gpu() -> None:
    t = tf()
    for gpu in t.config.list_physical_devices("GPU"):
        try:
            t.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def resolve_model_dir(args: argparse.Namespace) -> Path:
    if args.model_dir:
        return Path(args.model_dir).expanduser().resolve()
    suffix = "flw_model" if args.model == "flow" else "pkt_model"
    if args.model_source == "retrained":
        return (REPO_ROOT / "models" / "retrained" / suffix).resolve()
    return (REPO_ROOT / "models" / suffix).resolve()


def resolve_norm_dir(args: argparse.Namespace, model_dir: Path) -> Path:
    if args.normalization_dir:
        candidate = Path(args.normalization_dir).expanduser().resolve()
        if (candidate / "train_min.npy").is_file() and (candidate / "train_max.npy").is_file():
            return candidate
        # Se o caminho foi passado pelo job.sh para um modelo pré-treinado,
        # ele pode não ter x_train_meta; cai para os metadados de data/.
    candidate = model_dir / "x_train_meta"
    if (candidate / "train_min.npy").is_file() and (candidate / "train_max.npy").is_file():
        return candidate
    subdir = "flow_based" if args.model == "flow" else "packet_based"
    return (REPO_ROOT / "data" / subdir / "x_train_meta").resolve()


def resolve_data_dir(args: argparse.Namespace) -> Path:
    if args.data_root:
        return Path(args.data_root).expanduser().resolve()
    return (REPO_ROOT / "data").resolve()


def resolve_results_root(args: argparse.Namespace) -> Path:
    if args.results_root:
        return Path(args.results_root).expanduser().resolve()
    env_root = os.environ.get("RESULTS_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (REPO_ROOT / "results" / "cluster").resolve()


def load_meta(norm_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    train_min = np.load(norm_dir / "train_min.npy").astype(np.float32)
    train_max = np.load(norm_dir / "train_max.npy").astype(np.float32)
    return train_min, train_max


def load_saved_model(model_dir: Path):
    if not (model_dir / "saved_model.pb").is_file():
        raise FileNotFoundError(
            f"SavedModel não encontrado em {model_dir}. "
            "Rode o setup ou use --model-source pretrained."
        )
    from load_model_compat import load_model

    return load_model(str(model_dir))


def make_run_id(task: str, model: str, attack: str | None = None) -> str:
    parts = [
        metrics.utc_stamp(),
        f"job{os.environ.get('SLURM_JOB_ID', 'local')}",
        task,
        model,
    ]
    if attack:
        parts.append(metrics.sanitize_name(attack))
    return "_".join(parts)


def model_fingerprint(model_dir: Path) -> dict[str, Any]:
    return metrics.model_fingerprint(model_dir)


def data_fingerprint(norm_dir: Path, data_dir: Path) -> dict[str, Any]:
    meta_files = [norm_dir / "train_min.npy", norm_dir / "train_max.npy"]
    return {
        "normalization_dir": str(norm_dir),
        "normalization_files": {
            path.name: {
                "size": metrics.file_size(path),
                "sha256": metrics.sha256_file(path) if path.is_file() else None,
            }
            for path in meta_files
        },
        "data_dir": str(data_dir),
    }


def result_config(
    args: argparse.Namespace,
    model_dir: Path,
    norm_dir: Path,
    data_dir: Path,
    *,
    experiment: str,
    score_mode: str,
    attack: str | None,
    threshold: float | None,
    max_samples: int | None,
) -> dict[str, Any]:
    payload = {
        "schema_version": metrics.SCHEMA_VERSION,
        "experiment": experiment,
        "model_type": args.model,
        "score_mode": score_mode,
        "attack_type": attack,
        "seed": args.seed,
        "max_samples": max_samples,
        "threshold": threshold,
        "model_fingerprint": model_fingerprint(model_dir),
        "data_fingerprint": data_fingerprint(norm_dir, data_dir),
        "git": metrics.git_info(REPO_ROOT),
    }
    return payload


def write_manifest(
    results_root: Path,
    run_id: str,
    config: Mapping[str, Any],
    *,
    log_path: Path,
    result_path: Path | None,
    started_at: str,
    finished_at: str,
) -> None:
    manifest = {
        "run_id": run_id,
        "config": config,
        "environment": metrics.environment_manifest(),
        "paths": {
            "repo": str(REPO_ROOT),
            "results_root": str(results_root),
            "log": str(log_path),
            "result": str(result_path) if result_path else None,
        },
        "started_at": started_at,
        "finished_at": finished_at,
    }
    metrics.write_json_atomic(results_root / "manifests" / f"{run_id}.json", manifest)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_flow_test(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    test_files = [
        data_dir / "flow_based" / "Tuesday-WH-generate-labeled.csv",
        data_dir / "flow_based" / "Wednesday-WH-generate-labeled.csv",
        data_dir / "flow_based" / "Thursday-WH-generate-labeled.csv",
        data_dir / "flow_based" / "Friday-WH-generate-labeled.csv",
    ]
    missing = [str(path) for path in test_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "CSVs de fluxo ausentes:\n" + "\n".join(missing) + "\n"
            "Rode bash cluster/setup_env.sh."
        )

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for index, path in enumerate(test_files):
        LOG.info("carregando fluxo %s (%s)", path.name, index)
        frame = pd.read_csv(path)
        features = frame.iloc[:, 8:]
        ds_port = frame.iloc[:, 5]
        frame = pd.concat([ds_port, features], axis=1)

        labels = frame.iloc[:, -1].values
        values = frame.iloc[:, :-1].astype(np.float64).values
        known = (np.any(np.isinf(values), axis=1) + np.any(np.isnan(values), axis=1)) == False  # noqa: E712
        x_parts.append(values[known].astype(np.float32))
        y_parts.append(labels[known])
        LOG.info("  linhas=%d válidas=%d", len(frame), int(known.sum()))

    x_test = np.concatenate(x_parts, axis=0).astype(np.float32)
    y_test = np.concatenate(y_parts, axis=0)

    # The adversarial notebook derives the list of integer-valued features from
    # the test set. Keep exactly the same criterion.
    decimal_features = [
        index
        for index in range(x_test.shape[1])
        if np.sum(np.abs(x_test[:, index] - np.round(x_test[:, index]))) == 0
    ]
    LOG.info("fluxo: x=%s labels=%s features_inteiras=%d", x_test.shape, y_test.shape, len(decimal_features))
    return x_test, y_test, decimal_features


def load_packet_day(data_dir: Path, day: str) -> tuple[np.ndarray, np.ndarray]:
    day_dir = data_dir / "packet_based" / day
    files = sorted(day_dir.glob("part_*.npy"))
    if not files:
        raise FileNotFoundError(f"partes de pacotes ausentes em {day_dir}")
    x_parts = []
    for path in files:
        x_parts.append(np.load(path))
    x = np.concatenate(x_parts, axis=0)
    labels_path = day_dir / "labels.npy"
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels.npy ausente em {day_dir}")
    y = np.load(labels_path)
    if len(x) != len(y):
        raise ValueError(f"{day}: x tem {len(x)} pacotes e labels tem {len(y)}")
    LOG.info("pacotes %s: x=%s labels=%s", day, x.shape, y.shape)
    return x, y


def iter_packet_windows(
    data_dir: Path,
    days: Sequence[str],
    timesteps: int,
    max_windows: int | None = None,
):
    """Yield packet windows per day, preserving the original label alignment.

    A generator is used so only one day's windows are kept in memory at a
    time; the full test set would otherwise occupy several gigabytes.
    """

    total = 0
    for day in days:
        x, y = load_packet_day(data_dir, day)
        pad = timesteps - len(x) % timesteps
        x_pad = np.concatenate((x, x[:pad]), axis=0)
        windows = x_pad.reshape(-1, timesteps, x.shape[1])
        pad_y = timesteps - len(y) % timesteps
        y_pad = np.concatenate((y, y[:pad_y]), axis=0)
        labels = y_pad.reshape(-1, timesteps)[:, -1]
        if max_windows is not None:
            remaining = max_windows - total
            if remaining <= 0:
                break
            windows = windows[:remaining]
            labels = labels[:remaining]
        total += len(windows)
        LOG.info("  %s: janelas=%d", day, len(windows))
        yield day, windows, labels


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def reconstruction_errors(
    model: Any,
    x: np.ndarray,
    train_min: np.ndarray,
    train_max: np.ndarray,
    *,
    n_masks: int = 100,
    batch_size: int = 100,
) -> np.ndarray:
    """Return per-mask reconstruction errors with shape (n_samples, n_masks)."""

    t = tf()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim < 2:
        raise ValueError(f"x must have a feature dimension, got shape {x.shape}")
    feature_shape = x.shape[1:]
    errors = np.empty((len(x), n_masks), dtype=np.float64)

    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size]
        x_norm = (batch - train_min) / (train_max - train_min + 1e-6)

        masks = t.random.uniform((n_masks, *feature_shape), dtype=t.float32)
        masks = t.cast(masks > 0.75, t.float32)
        partial = t.expand_dims(x_norm, 1) * masks[None, ...]
        flat = t.reshape(partial, (-1, *feature_shape))
        reconstructed = model(flat, training=False)
        reconstructed = t.reshape(reconstructed, (len(batch), n_masks, *feature_shape))
        diff = t.square(reconstructed - t.expand_dims(x_norm, 1))
        axes = tuple(range(2, 2 + len(feature_shape)))
        batch_errors = t.reduce_mean(diff, axis=axes).numpy()
        errors[start : start + len(batch)] = batch_errors

    return errors


def score_from_errors(errors: np.ndarray, mode: str) -> np.ndarray:
    return metrics.aggregate_mask_errors(errors, mode=mode, groups=5)


def score_samples(
    model: Any,
    x: np.ndarray,
    train_min: np.ndarray,
    train_max: np.ndarray,
    *,
    mode: str,
    n_masks: int = 100,
    batch_size: int = 100,
) -> np.ndarray:
    errors = reconstruction_errors(
        model, x, train_min, train_max, n_masks=n_masks, batch_size=batch_size
    )
    return score_from_errors(errors, mode)


def score_single(
    model: Any,
    sample: np.ndarray,
    train_min: np.ndarray,
    train_max: np.ndarray,
    *,
    mode: str,
    seed: int,
    n_masks: int = 100,
) -> float:
    t = tf()
    t.random.set_seed(int(seed))
    errors = reconstruction_errors(
        model, sample[None], train_min, train_max, n_masks=n_masks, batch_size=1
    )
    return float(score_from_errors(errors, mode)[0])


# ---------------------------------------------------------------------------
# Flow attack (transcribed from Flow_based_RePO_Adversarial.ipynb)
# ---------------------------------------------------------------------------


class FlowAttack:
    def __init__(
        self,
        model: Any,
        train_min: np.ndarray,
        train_max: np.ndarray,
        decimal_features: Sequence[int],
        num_input: int,
        real_threshold: float,
        attack_threshold: float,
        *,
        learning_rates: Sequence[float] = (0.001, 0.01, 0.1, 1.0),
    ) -> None:
        t = tf()
        self.model = model
        self.train_min = train_min
        self.train_max = train_max
        self.decimal_features = list(decimal_features)
        self.num_input = int(num_input)
        self.real_threshold = float(real_threshold)
        self.attack_threshold = float(attack_threshold)

        self.alpha = t.Variable(
            np.zeros((1, self.num_input), dtype=np.float32), name="modifier"
        )
        self.optimizers = [
            t.keras.optimizers.Adam(learning_rate=rate) for rate in learning_rates
        ]
        # Match the reference notebook: initialize optimizer slots once.
        for optimizer in self.optimizers:
            optimizer.apply_gradients(zip([self.alpha], [self.alpha]))

        self.fixed_mask, self.mask_l1, self.mask_l2, self.mask_l3 = self._build_masks()
        self.dup_ht = {61: 2, 63: 3, 62: 4, 64: 5}
        self.mins = [7, 19, 24, 38]
        self.maxs = [6, 18, 23, 39]
        self.flags = [30, 32, 43, 44, 45, 46, 47, 48, 49, 50]
        self.aggr_features = [9, 17, 22, 41, 42, 55, 56, 57, 68, 69, 70, 71, 72, 73, 74, 75, 76]

        self._get_adv = self._make_get_adv(t)
        self._optim = self._make_optim(t)
        self._test_step = self._make_test_step(t)

    def _build_masks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        fixed = np.ones(77, dtype=np.float32)
        for index in (0, 3, 4, 5, 10, 11, 12, 13, 25, 26, 27, 28, 29, 31, 33, 35, 37, 53, 54, 66):
            fixed[index] = 0
        for index in range(58, 65):
            fixed[index] = 0
        for index in (9, 17, 22, 41, 42, 55, 56, 57):
            fixed[index] = 0
        for index in range(68, 77):
            fixed[index] = 0
        for index in (8, 14, 15, 16, 20, 21, 34, 36, 38, 39, 40, 51, 52, 67):
            fixed[index] = 0

        mask_l1 = np.copy(fixed)
        mask_l2 = np.copy(fixed)
        for index in (9, 17, 22, 41, 42):
            mask_l2[index] = 1
        mask_l3 = np.copy(mask_l2)
        for index in (55, 56, 57):
            mask_l3[index] = 1
        for index in range(68, 77):
            mask_l3[index] = 1
        return fixed, mask_l1, mask_l2, mask_l3

    def _make_get_adv(self, t):
        @t.function
        def get_adv(x, adv_mask):
            x_normalized = (x - self.train_min) / (self.train_max - self.train_min + 1e-6)
            alpha_masked = self.alpha * adv_mask
            adv_ex_normalized = x_normalized + alpha_masked
            return adv_ex_normalized * (self.train_max - self.train_min + 1e-6) + self.train_min

        return get_adv

    def _make_optim(self, t):
        @t.function
        def optim(x, adv_mask, optimizer):
            with t.GradientTape() as tape:
                x_normalized = (x - self.train_min) / (self.train_max - self.train_min + 1e-6)
                adv_ex_normalized = x_normalized + self.alpha * adv_mask
                def_mask = t.random.uniform(shape=[100, self.num_input])
                def_mask = t.cast(def_mask > 0.75, t.float32)
                partial_x = def_mask * adv_ex_normalized
                rec_x = self.model(partial_x, training=False)
                score = t.reduce_mean(t.square(rec_x - adv_ex_normalized), axis=1)
                loss = t.reduce_sum(score)
            gradients = tape.gradient(loss, [self.alpha])
            optimizer.apply_gradients(zip(gradients, [self.alpha]))

        return optim

    def _make_test_step(self, t):
        @t.function
        def test_step(x):
            def_mask = t.random.uniform(shape=[100, self.num_input])
            def_mask = t.cast(def_mask > 0.75, t.float32)
            x_normalized = (x - self.train_min) / (self.train_max - self.train_min + 1e-6)
            partial_x = def_mask * x_normalized
            rec_x = self.model(partial_x, training=False)
            score = t.reduce_mean(t.square(rec_x - x_normalized), axis=1)
            score = t.reduce_min(t.reshape(score, [5, 20]), axis=-1)
            return t.reduce_sum(score)

        return test_step

    def score(self, sample: np.ndarray, *, seed: int) -> float:
        t = tf()
        t.random.set_seed(int(seed))
        return float(self._test_step(np.asarray(sample, dtype=np.float32)[None]).numpy())

    def find_adv(self, sample: np.ndarray) -> np.ndarray | str | None:
        t = tf()
        self.alpha.assign(np.zeros((1, self.num_input), dtype=np.float32))
        orig_sample = np.copy(np.asarray(sample, dtype=np.float32)[None])

        adv_ex = self._get_adv(orig_sample, self.mask_l1)
        if float(self._test_step(adv_ex).numpy()) < self.attack_threshold:
            return "no change"

        backup = [None]

        def optimize(optimizer, total_iter, mask_v):
            self.alpha.assign(np.zeros((1, self.num_input), dtype=np.float32))
            for _ in range(total_iter):
                self._optim(orig_sample, mask_v, optimizer)
                adv_ex = self._get_adv(orig_sample, mask_v)
                adv_ex_np = adv_ex.numpy()

                for key, source in self.dup_ht.items():
                    adv_ex_np[0, key] = adv_ex_np[0, source]
                adv_ex_np[0, self.mins] = np.maximum(0, adv_ex_np[0, self.mins])
                adv_ex_np[0, self.mins] = np.minimum(orig_sample[0, self.mins], adv_ex_np[0, self.mins])
                adv_ex_np[0, self.maxs] = np.maximum(orig_sample[0, self.maxs], adv_ex_np[0, self.maxs])
                adv_ex_np[0, self.flags] = np.maximum(orig_sample[0, self.flags], adv_ex_np[0, self.flags])
                flags_max_changed = np.max(adv_ex_np[0, self.flags] - orig_sample[0, self.flags])
                adv_ex_np[0, self.aggr_features] = np.maximum(0, adv_ex_np[0, self.aggr_features])
                adv_ex_np[0, 2] = np.maximum(orig_sample[0, 2] + flags_max_changed, adv_ex_np[0, 2])
                adv_ex_np[0, 65] = np.maximum(0, adv_ex_np[0, 65])

                adv_ex_np[0, self.decimal_features] = np.round(adv_ex_np[0, self.decimal_features])

                adv_ex_np[0, 4] = (
                    adv_ex_np[0, 4]
                    + (adv_ex_np[0, 6] != orig_sample[0, 6]) * adv_ex_np[0, 6]
                    + (adv_ex_np[0, 7] != orig_sample[0, 7]) * adv_ex_np[0, 7]
                )
                adv_ex_np[0, 8] = adv_ex_np[0, 4] / adv_ex_np[0, 2]
                adv_ex_np[0, 14] = (adv_ex_np[0, 4] + adv_ex_np[0, 5]) / adv_ex_np[0, 1] * 1e6
                adv_ex_np[0, 15] = (adv_ex_np[0, 2] + adv_ex_np[0, 3]) / adv_ex_np[0, 1] * 1e6
                adv_ex_np[0, 16] = adv_ex_np[0, 1] / (adv_ex_np[0, 2] + adv_ex_np[0, 3] - 1)

                adv_ex_np[0, 21] = adv_ex_np[0, 20] / (adv_ex_np[0, 2] - 1)
                adv_ex_np[0, 34] = adv_ex_np[0, 34] + 20 * (adv_ex_np[0, 2] - orig_sample[0, 2])
                adv_ex_np[0, 36] = adv_ex_np[0, 2] / adv_ex_np[0, 1] * 1e6

                adv_ex_np[0, 38] = np.minimum(adv_ex_np[0, 38], adv_ex_np[0, 7])
                adv_ex_np[0, 39] = np.maximum(adv_ex_np[0, 39], adv_ex_np[0, 6])

                adv_ex_np[0, 40] = (adv_ex_np[0, 4] + adv_ex_np[0, 5]) / (adv_ex_np[0, 3] + adv_ex_np[0, 2] + 1)
                adv_ex_np[0, 51] = adv_ex_np[0, 3] / adv_ex_np[0, 2]
                adv_ex_np[0, 52] = (adv_ex_np[0, 4] + adv_ex_np[0, 5]) / (adv_ex_np[0, 3] + adv_ex_np[0, 2])
                adv_ex_np[0, 53] = adv_ex_np[0, 8]
                adv_ex_np[0, 67] = (
                    adv_ex_np[0, 67]
                    + (adv_ex_np[0, 6] != orig_sample[0, 6]) * 1
                    + (adv_ex_np[0, 7] != orig_sample[0, 7]) * 1
                )

                adv_ex_np[np.isinf(adv_ex_np)] = 0
                adv_ex_np[np.isnan(adv_ex_np)] = 0
                score = float(self._test_step(adv_ex_np).numpy())
                if score < self.attack_threshold:
                    return adv_ex_np
                if score < self.real_threshold and backup[0] is None:
                    backup[0] = adv_ex_np
            return None

        result: Any = None
        for mask in (self.mask_l1, self.mask_l2, self.mask_l3):
            result = optimize(self.optimizers[0], 30, mask)
            if result is None:
                result = optimize(self.optimizers[1], 40, mask)
                if result is None:
                    result = optimize(self.optimizers[2], 50, mask)
                    if result is None:
                        result = optimize(self.optimizers[3], 60, mask)
            if result is not None:
                break

        if result is None:
            return backup[0]
        return result

    def optimizer_state(self) -> dict[str, np.ndarray]:
        state: dict[str, np.ndarray] = {}
        for index, optimizer in enumerate(self.optimizers):
            for weight_index, variable in enumerate(optimizer.variables):
                state[f"flow_opt_{index}_{weight_index}"] = np.asarray(variable.numpy())
        return state

    def load_optimizer_state(self, state: Mapping[str, np.ndarray]) -> None:
        for index, optimizer in enumerate(self.optimizers):
            weights = []
            weight_index = 0
            while f"flow_opt_{index}_{weight_index}" in state:
                weights.append(state[f"flow_opt_{index}_{weight_index}"])
                weight_index += 1
            if not weights:
                continue
            variables = list(optimizer.variables)
            if len(weights) != len(variables):
                LOG.warning(
                    "estado do optimizer %d tem %d tensores, esperado %d; ignorando",
                    index, len(weights), len(variables),
                )
                continue
            for variable, value in zip(variables, weights):
                variable.assign(value)


# ---------------------------------------------------------------------------
# Packet attack (transcribed from Packet_based_RePO_Adversarial.ipynb)
# ---------------------------------------------------------------------------


class PacketAttack:
    def __init__(
        self,
        model: Any,
        train_min: np.ndarray,
        train_max: np.ndarray,
        *,
        timesteps: int = 20,
        num_input: int = 29,
        threshold: float,
    ) -> None:
        t = tf()
        self.model = model
        self.train_min = train_min
        self.train_max = train_max
        self.timesteps = int(timesteps)
        self.num_input = int(num_input)
        self.threshold = float(threshold)

        self.alpha_delay = t.Variable(
            np.zeros((1, 1, self.num_input), dtype=np.float32), name="delay"
        )
        self.alpha_split = t.Variable(np.zeros((1,), dtype=np.float32), name="split")
        self.alpha_inject = t.Variable(
            np.zeros((1, 1, self.num_input), dtype=np.float32), name="modifier"
        )
        self.optimizer_split = t.keras.optimizers.Adam(learning_rate=0.1)
        self.optimizer_inject = t.keras.optimizers.Adam(learning_rate=0.1)
        # Inicializa os slots dos otimizadores para que o estado possa ser
        # salvo/restaurado em checkpoints.
        self.optimizer_split.apply_gradients(
            zip(
                [t.zeros_like(self.alpha_delay), t.zeros_like(self.alpha_split)],
                [self.alpha_delay, self.alpha_split],
            )
        )
        self.optimizer_inject.apply_gradients(
            zip([t.zeros_like(self.alpha_inject)], [self.alpha_inject])
        )

        self.tcp_mask = np.array([1] * 8 + [1] * 16 + [0] * 3 + [0] * 1 + [1] * 1, dtype=np.float32)
        self.udp_mask = np.array([1] * 8 + [0] * 16 + [1] * 3 + [0] * 1 + [1] * 16, dtype=np.float32)
        self.packet_mins = np.array(
            [0, 60, 20, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1],
            dtype=np.float32,
        )
        self.packet_maxs = np.array(
            [15, 2**16, 20, 2**16, 1, 1, 1, 255, 2**16, 2**16, 2**32, 2**32, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2**16, 2**16, 2**16, 2**16, 2**16, 0, 1],
            dtype=np.float32,
        )
        self.fake_packets: list[np.ndarray] = []

        self._test_step = self._make_test_step(t)
        self._get_delayed = self._make_get_delayed(t)
        self._delay_optim = self._make_delay_optim(t)
        self._get_injected = self._make_get_injected(t)
        self._inject_optim = self._make_inject_optim(t)

    def _make_test_step(self, t):
        @t.function
        def test_step(x):
            def_mask = t.random.uniform(shape=[100, self.timesteps, self.num_input])
            def_mask = t.cast(def_mask > 0.75, t.float32)
            x_normalized = (x - self.train_min) / (self.train_max - self.train_min + 1e-6)
            partial_x = def_mask * x_normalized
            rec_x = self.model(partial_x, training=False)
            score = t.reduce_mean(t.square(rec_x - x_normalized), axis=[1, 2])
            score = t.reduce_min(t.reshape(score, [5, 20]), axis=-1)
            return t.reduce_sum(score)

        return test_step

    def _make_get_delayed(self, t):
        @t.function
        def get_delayed(x, p_len):
            alpha_split_2 = t.zeros((1, 1, self.num_input)) + self.alpha_split
            alpha_split_2 = t.minimum(alpha_split_2, 0.0)
            alpha_split_2 = t.maximum(alpha_split_2, -p_len + np.float32(61))
            alpha_delay_2 = t.maximum(self.alpha_delay, 0.0)
            alpha_delay_2 = t.minimum(alpha_delay_2, 15.0)
            masked_alpha = alpha_delay_2 * np.ones((1, 1, self.num_input), dtype=np.float32)
            mask_split = np.zeros((1, 1, self.num_input), dtype=np.float32)
            mask_split[0, 0, 1] = mask_split[0, 0, 3] = 1
            mask_split = t.cast(mask_split, t.bool)
            alpha_final = t.where(mask_split, alpha_split_2, masked_alpha)
            last_ts_modified = x[0, -1] + alpha_final
            return t.concat((x[:, :19], last_ts_modified), axis=1)

        return get_delayed

    def _make_delay_optim(self, t):
        @t.function
        def delay_split_optim(x, p_len):
            with t.GradientTape() as tape:
                adv_x = self._get_delayed(x, p_len)
                adv_x_normalized = (adv_x - self.train_min) / (self.train_max - self.train_min + 1e-6)
                rand_mask = t.random.uniform(shape=[100, self.timesteps, self.num_input])
                rand_mask = t.cast(rand_mask > 0.75, t.float32)
                partial = adv_x_normalized * rand_mask
                rec = self.model(partial, training=False)
                score = t.reduce_mean(t.square(rec - adv_x_normalized), axis=[1, 2])
                loss = t.reduce_sum(score)
            gradients = tape.gradient(loss, [self.alpha_delay, self.alpha_split])
            self.optimizer_split.apply_gradients(
                zip(gradients, [self.alpha_delay, self.alpha_split])
            )

        return delay_split_optim

    def _make_get_injected(self, t):
        @t.function
        def get_injected(x, inject_mask):
            alpha_inject_2 = t.minimum(self.alpha_inject, self.packet_maxs)
            alpha_inject_2 = t.maximum(alpha_inject_2, self.packet_mins)
            alpha_inject_masked = alpha_inject_2 * inject_mask
            adv_x1 = t.concat((x[:, :19], alpha_inject_masked), axis=1)
            adv_x2 = t.concat((x[:, 1:19], alpha_inject_masked, x[:, 19:]), axis=1)
            return adv_x1, adv_x2

        return get_injected

    def _make_inject_optim(self, t):
        @t.function
        def inject_optim(x, inject_mask):
            with t.GradientTape() as tape:
                adv_x1, adv_x2 = self._get_injected(x, inject_mask)
                adv_x1_normalized = (adv_x1 - self.train_min) / (self.train_max - self.train_min + 1e-6)
                adv_x2_normalized = (adv_x2 - self.train_min) / (self.train_max - self.train_min + 1e-6)
                rand_mask1 = t.random.uniform(shape=[100, self.timesteps, self.num_input])
                rand_mask1 = t.cast(rand_mask1 > 0.75, t.float32)
                rand_mask2 = t.random.uniform(shape=[100, self.timesteps, self.num_input])
                rand_mask2 = t.cast(rand_mask2 > 0.75, t.float32)
                rec1 = self.model(adv_x1_normalized * rand_mask1, training=False)
                rec2 = self.model(adv_x2_normalized * rand_mask2, training=False)
                score1 = t.reduce_sum(t.reduce_mean(t.square(rec1 - adv_x1_normalized), axis=[1, 2]))
                score2 = t.reduce_sum(t.reduce_mean(t.square(rec2 - adv_x2_normalized), axis=[1, 2]))
                loss = score1 + score2
            gradients = tape.gradient(loss, [self.alpha_inject])
            self.optimizer_inject.apply_gradients(zip(gradients, [self.alpha_inject]))

        return inject_optim

    def score(self, sample: np.ndarray, *, seed: int) -> float:
        t = tf()
        t.random.set_seed(int(seed))
        return float(self._test_step(np.asarray(sample, dtype=np.float32)[None]).numpy())

    def find_adv(self, x: np.ndarray) -> Any:
        score = float(self._test_step(x).numpy())
        if score < self.threshold:
            return "cons_as_ben"
        if x[0, -1, -1] == 2:
            return self.inject(np.copy(x))
        result = self.delay_and_split(np.copy(x))
        if result:
            return ("split", result)
        return self.inject(np.copy(x))

    def delay_and_split(self, x: np.ndarray) -> list[np.ndarray] | None:
        self.alpha_delay.assign(np.zeros((1, 1, self.num_input), dtype=np.float32))
        self.alpha_split.assign(np.zeros((1,), dtype=np.float32))
        len_last = x[0, -1, 1]
        adv_x = self._get_delayed(x, len_last).numpy()
        if float(self._test_step(adv_x).numpy()) < self.threshold:
            return [adv_x]

        result: list[np.ndarray] = []
        for _ in range(300):
            self._delay_optim(x, len_last)
            adv_x = self._get_delayed(x, len_last).numpy()
            adv_x[0, -1, 1:4] = np.round(adv_x[0, -1, 1:4])
            if float(self._test_step(adv_x).numpy()) < self.threshold:
                first_part = np.copy(adv_x)
                diff = len_last - adv_x[0, -1, 1]
                if diff > 0:
                    adv_x[0, -1, 1] = diff + 60
                    adv_x[0, -1, 0] = 0
                    adv_x[0, -1, 3] = diff + 60 - 14
                    second_part = self.delay_and_split(adv_x)
                    if second_part is None:
                        return None
                    result.append(first_part)
                    result.extend(second_part)
                else:
                    result.append(first_part)
                break
        if not result:
            return None
        return result

    def inject(self, x: np.ndarray, mask_type: str = "tcp") -> Any:
        self.alpha_inject.assign(np.zeros((1, 1, self.num_input), dtype=np.float32))
        current_mask = self.tcp_mask if mask_type == "tcp" else self.udp_mask
        for _ in range(300):
            self._inject_optim(x, current_mask)
            adv_x1, adv_x2 = self._get_injected(x, current_mask)
            adv_x1, adv_x2 = adv_x1.numpy(), adv_x2.numpy()
            adv_x1[0, :, 1:] = np.round(adv_x1[0, :, 1:])
            score1 = float(self._test_step(adv_x1).numpy())
            adv_x2[0, :, 1:] = np.round(adv_x2[0, :, 1:])
            score2 = float(self._test_step(adv_x2).numpy())
            if score1 < self.threshold and score2 < self.threshold:
                self.fake_packets.append(adv_x1[0, -1])
                return ("inject", adv_x1[0, -1])
        return None

    def optimizer_state(self) -> dict[str, np.ndarray]:
        state: dict[str, np.ndarray] = {}
        for name, optimizer in (
            ("split", self.optimizer_split),
            ("inject", self.optimizer_inject),
        ):
            for index, variable in enumerate(optimizer.variables):
                state[f"packet_opt_{name}_{index}"] = np.asarray(variable.numpy())
        return state

    def load_optimizer_state(self, state: Mapping[str, np.ndarray]) -> None:
        for name, optimizer in (
            ("split", self.optimizer_split),
            ("inject", self.optimizer_inject),
        ):
            weights = []
            index = 0
            while f"packet_opt_{name}_{index}" in state:
                weights.append(state[f"packet_opt_{name}_{index}"])
                index += 1
            if not weights:
                continue
            variables = list(optimizer.variables)
            if len(weights) != len(variables):
                LOG.warning(
                    "estado do optimizer %s tem %d tensores, esperado %d; ignorando",
                    name, len(weights), len(variables),
                )
                continue
            for variable, value in zip(variables, weights):
                variable.assign(value)


# ---------------------------------------------------------------------------
# Normal evaluation
# ---------------------------------------------------------------------------


def run_normal_flow(args: argparse.Namespace, paths: dict[str, Path]) -> int:
    started = metrics.iso_now()
    wall_start = time.time()
    run_id = make_run_id("normal", "flow")
    log_path = paths["results_root"] / "logs" / f"{run_id}.log"
    setup_logging(log_path)
    configure_gpu()
    t = tf()
    t.keras.utils.set_random_seed(args.seed)

    data_dir = paths["data_dir"]
    model_dir = paths["model_dir"]
    norm_dir = paths["norm_dir"]
    output_dir = paths["results_root"] / "flow" / "normal"
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("run_id=%s", run_id)
    LOG.info("modelo=%s", model_dir)
    x, labels, _decimal = load_flow_test(data_dir)
    if args.max_samples:
        x = x[: args.max_samples]
        labels = labels[: args.max_samples]
        LOG.info("amostras limitadas a %d", len(x))

    train_min, train_max = load_meta(norm_dir)
    model = load_saved_model(model_dir)

    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    errors_chunks: list[np.ndarray] = []
    for start in range(0, len(x), args.batch_size):
        batch = x[start : start + args.batch_size]
        errors_chunks.append(
            reconstruction_errors(model, batch, train_min, train_max, batch_size=args.batch_size)
        )
        LOG.info("pontuado %d/%d", min(start + args.batch_size, len(x)), len(x))
    errors = np.concatenate(errors_chunks, axis=0)

    for mode in modes:
        scores = score_from_errors(errors, mode)
        threshold = metrics.threshold_at_fpr(
            scores[labels == "BENIGN"], 0.01, method="ceil"
        )
        rates = metrics.compute_class_rates(scores, labels, metrics.FLOW_LABELS, threshold)
        average = metrics.average_attack_tpr(rates)
        paper_threshold = metrics.PAPER_THRESHOLDS["flow"]["fpr_0.1"]
        benign_mask = labels == "BENIGN"
        fpr_at_paper_threshold = (
            float(np.mean(scores[benign_mask] >= paper_threshold)) if benign_mask.any() else None
        )
        runtime = time.time() - wall_start

        score_path = output_dir / f"scores_{mode}.npz"
        np.savez_compressed(score_path, scores=scores, labels=labels, threshold=threshold)

        result = {
            "schema_version": metrics.SCHEMA_VERSION,
            "status": "completed",
            "run_id": run_id,
            "experiment": "normal",
            "model_type": "flow",
            "score_mode": mode,
            "attack_type": None,
            "threshold": threshold,
            "threshold_fpr": 0.01,
            "threshold_method": "ceil",
            "fpr": rates["BENIGN"]["rate"],
            "paper_threshold": paper_threshold,
            "fpr_at_paper_threshold": fpr_at_paper_threshold,
            "class_rates": rates,
            "average_tpr": average,
            "paper_average_tpr": metrics.PAPER_AVERAGES.get(("flow", "normal", mode)),
            "n_samples": int(len(scores)),
            "scores_file": str(score_path),
            "runtime_seconds": runtime,
            "config": result_config(
                args, model_dir, norm_dir, data_dir,
                experiment="normal", score_mode=mode, attack=None,
                threshold=threshold, max_samples=args.max_samples,
            ),
            "environment": metrics.environment_manifest(),
            "started_at": started,
            "finished_at": metrics.iso_now(),
        }
        result["config_hash"] = metrics.config_hash(result["config"])
        result_path = output_dir / f"{mode}.json"
        metrics.write_json_atomic(result_path, result)
        LOG.info(
            "%s: threshold=%.6g fpr=%s avg_tpr=%s -> %s",
            mode, threshold, result["fpr"], average, result_path,
        )
        write_manifest(
            paths["results_root"], f"{run_id}_{mode}", result["config"],
            log_path=log_path, result_path=result_path,
            started_at=started, finished_at=metrics.iso_now(),
        )
    return 0


def run_normal_packet(args: argparse.Namespace, paths: dict[str, Path]) -> int:
    started = metrics.iso_now()
    wall_start = time.time()
    run_id = make_run_id("normal", "packet")
    log_path = paths["results_root"] / "logs" / f"{run_id}.log"
    setup_logging(log_path)
    configure_gpu()
    t = tf()
    t.keras.utils.set_random_seed(args.seed)

    data_dir = paths["data_dir"]
    model_dir = paths["model_dir"]
    norm_dir = paths["norm_dir"]
    output_dir = paths["results_root"] / "packet" / "normal"
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("run_id=%s", run_id)
    LOG.info("modelo=%s", model_dir)
    days = ("tuesday", "wednesday", "thursday", "friday")
    train_min, train_max = load_meta(norm_dir)
    model = load_saved_model(model_dir)

    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    all_errors: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for day, windows, labels in iter_packet_windows(
        data_dir, days, timesteps=20, max_windows=args.max_samples
    ):
        LOG.info("pontuando %s (%d janelas)", day, len(windows))
        for start in range(0, len(windows), args.batch_size):
            batch = windows[start : start + args.batch_size]
            all_errors.append(
                reconstruction_errors(model, batch, train_min, train_max, batch_size=args.batch_size)
            )
        all_labels.append(labels)
        LOG.info("  %s concluído", day)
    errors = np.concatenate(all_errors, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    valid = labels != -1
    errors = errors[valid]
    labels = labels[valid]
    LOG.info("janelas válidas=%d", len(labels))

    for mode in modes:
        scores = score_from_errors(errors, mode)
        threshold = metrics.threshold_at_fpr(
            scores[labels == 0], 0.01, method="round"
        )
        rates = metrics.compute_class_rates(scores, labels, metrics.PACKET_LABELS, threshold)
        average = metrics.average_attack_tpr(rates)
        paper_threshold = metrics.PAPER_THRESHOLDS["packet"]["fpr_0.1"]
        benign_mask = labels == 0
        fpr_at_paper_threshold = (
            float(np.mean(scores[benign_mask] >= paper_threshold)) if benign_mask.any() else None
        )
        runtime = time.time() - wall_start

        score_path = output_dir / f"scores_{mode}.npz"
        np.savez_compressed(score_path, scores=scores, labels=labels, threshold=threshold)

        result = {
            "schema_version": metrics.SCHEMA_VERSION,
            "status": "completed",
            "run_id": run_id,
            "experiment": "normal",
            "model_type": "packet",
            "score_mode": mode,
            "attack_type": None,
            "threshold": threshold,
            "threshold_fpr": 0.01,
            "threshold_method": "round",
            "fpr": rates["Benign"]["rate"],
            "paper_threshold": paper_threshold,
            "fpr_at_paper_threshold": fpr_at_paper_threshold,
            "class_rates": rates,
            "average_tpr": average,
            "paper_average_tpr": metrics.PAPER_AVERAGES.get(("packet", "normal", mode)),
            "n_samples": int(len(scores)),
            "scores_file": str(score_path),
            "runtime_seconds": runtime,
            "config": result_config(
                args, model_dir, norm_dir, data_dir,
                experiment="normal", score_mode=mode, attack=None,
                threshold=threshold, max_samples=args.max_samples,
            ),
            "environment": metrics.environment_manifest(),
            "started_at": started,
            "finished_at": metrics.iso_now(),
        }
        result["config_hash"] = metrics.config_hash(result["config"])
        result_path = output_dir / f"{mode}.json"
        metrics.write_json_atomic(result_path, result)
        LOG.info(
            "%s: threshold=%.6g fpr=%s avg_tpr=%s -> %s",
            mode, threshold, result["fpr"], average, result_path,
        )
        write_manifest(
            paths["results_root"], f"{run_id}_{mode}", result["config"],
            log_path=log_path, result_path=result_path,
            started_at=started, finished_at=metrics.iso_now(),
        )
    return 0


def run_normal(args: argparse.Namespace, paths: dict[str, Path]) -> int:
    if args.model == "flow":
        return run_normal_flow(args, paths)
    return run_normal_packet(args, paths)


# ---------------------------------------------------------------------------
# Adversarial evaluation
# ---------------------------------------------------------------------------


def _checkpoint_path(output_dir: Path, attack: str) -> Path:
    return output_dir / "checkpoints" / f"{metrics.sanitize_name(attack)}.npz"


def _result_path(output_dir: Path, attack: str) -> Path:
    return output_dir / f"{metrics.sanitize_name(attack)}.json"


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as file:
        np.savez_compressed(file, **payload)
    os.replace(tmp, path)


def _completed_result(path: Path, config_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        result = metrics.read_json(path)
    except Exception:
        return False
    return result.get("status") == "completed" and result.get("config_hash") == config_hash


def run_adversarial_flow(args: argparse.Namespace, paths: dict[str, Path]) -> int:
    started = metrics.iso_now()
    attack_name = args.attack
    run_id = make_run_id("adversarial", "flow", attack_name)
    log_path = paths["results_root"] / "logs" / f"{run_id}.log"
    setup_logging(log_path)
    configure_gpu()
    t = tf()
    t.keras.utils.set_random_seed(args.seed)

    data_dir = paths["data_dir"]
    model_dir = paths["model_dir"]
    norm_dir = paths["norm_dir"]
    output_dir = paths["results_root"] / "flow" / "adversarial"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = _result_path(output_dir, attack_name)
    checkpoint_path = _checkpoint_path(output_dir, attack_name)
    scores_path = output_dir / f"scores_{metrics.sanitize_name(attack_name)}.npz"

    train_min, train_max = load_meta(norm_dir)
    real_threshold = float(args.real_threshold or metrics.PAPER_THRESHOLDS["flow"]["fpr_0.1"])
    attack_threshold = float(args.threshold or metrics.PAPER_THRESHOLDS["flow"]["attack"])
    max_samples = args.max_samples or 5000

    config = result_config(
        args, model_dir, norm_dir, data_dir,
        experiment="adversarial", score_mode="repo_plus", attack=attack_name,
        threshold=attack_threshold, max_samples=max_samples,
    )
    config["real_threshold"] = real_threshold
    config["attack_threshold"] = attack_threshold
    config_hash = metrics.config_hash(config)

    if _completed_result(result_path, config_hash) and not args.force:
        LOG.info("resultado já concluído para %s; nada a fazer", attack_name)
        return 0
    if args.force:
        for path in (result_path, checkpoint_path, scores_path):
            if path.exists():
                path.unlink()

    LOG.info("run_id=%s ataque=%s", run_id, attack_name)
    LOG.info("threshold_real=%.9g threshold_ataque=%.9g", real_threshold, attack_threshold)
    x_all, y_all, decimal_features = load_flow_test(data_dir)
    x_mal = x_all[y_all == attack_name]
    if len(x_mal) == 0:
        raise ValueError(f"nenhuma amostra de fluxo para o ataque {attack_name!r}")
    x_mal = x_mal[:max_samples].astype(np.float32)
    LOG.info("amostras do ataque=%d", len(x_mal))

    model = load_saved_model(model_dir)
    attack = FlowAttack(
        model, train_min, train_max, decimal_features, x_all.shape[1],
        real_threshold, attack_threshold,
    )

    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint is not None:
        saved_hash = str(checkpoint.get("config_hash", ""))
        if saved_hash != config_hash:
            if not args.force:
                raise RuntimeError(
                    "checkpoint encontrado com configuração diferente; use --force "
                    f"({checkpoint_path})"
                )
            checkpoint = None
    if checkpoint is None:
        scores_before = np.zeros(len(x_mal), dtype=np.float64)
        LOG.info("pontuando amostras originais...")
        for index in range(len(x_mal)):
            scores_before[index] = attack.score(x_mal[index], seed=args.seed + index)
        scores_after = np.zeros(len(x_mal), dtype=np.float64)
        x_adv = np.zeros_like(x_mal)
        counts = {"cons_as_mal": 0, "cons_as_ben": 0, "fooled": 0}
        next_index = 0
        elapsed = 0.0
    else:
        next_index = int(checkpoint["next_index"])
        scores_before = checkpoint["scores_before"]
        scores_after = checkpoint["scores_after"]
        x_adv = checkpoint["x_test_adv"]
        counts = {
            "cons_as_mal": int(checkpoint["cons_as_mal"]),
            "cons_as_ben": int(checkpoint["cons_as_ben"]),
            "fooled": int(checkpoint["fooled"]),
        }
        elapsed = float(checkpoint["elapsed"])
        attack.load_optimizer_state(checkpoint)
        LOG.info("retomando do checkpoint em %d/%d", next_index, len(x_mal))

    original_scores = scores_before.copy()
    loop_started = time.time()
    for index in range(next_index, len(x_mal)):
        if stop_requested():
            _save_checkpoint(
                checkpoint_path,
                {
                    **attack.optimizer_state(),
                    "next_index": index,
                    "scores_before": scores_before,
                    "scores_after": scores_after,
                    "x_test_adv": x_adv,
                    "cons_as_mal": counts["cons_as_mal"],
                    "cons_as_ben": counts["cons_as_ben"],
                    "fooled": counts["fooled"],
                    "elapsed": elapsed + (time.time() - loop_started),
                    "config_hash": config_hash,
                },
            )
            LOG.warning("interrompido em %d/%d; checkpoint salvo", index, len(x_mal))
            return 143

        t.random.set_seed(args.seed + 10_000 + index)
        result = attack.find_adv(x_mal[index])
        if isinstance(result, str) and result == "no change":
            counts["cons_as_ben"] += 1
            x_adv[index] = np.copy(x_mal[index])
        elif result is None:
            counts["cons_as_mal"] += 1
            x_adv[index] = np.copy(x_mal[index])
        else:
            counts["fooled"] += 1
            x_adv[index] = result

        scores_after[index] = attack.score(
            x_adv[index], seed=args.seed + 20_000 + index
        )

        if index % args.log_every == 0:
            LOG.info(
                "%d/%d t=%.1fs mal=%d ben=%d fooled=%d",
                index, len(x_mal), elapsed + (time.time() - loop_started),
                counts["cons_as_mal"], counts["cons_as_ben"], counts["fooled"],
            )
        if args.checkpoint_every and (index + 1) % args.checkpoint_every == 0:
            _save_checkpoint(
                checkpoint_path,
                {
                    **attack.optimizer_state(),
                    "next_index": index + 1,
                    "scores_before": scores_before,
                    "scores_after": scores_after,
                    "x_test_adv": x_adv,
                    "cons_as_mal": counts["cons_as_mal"],
                    "cons_as_ben": counts["cons_as_ben"],
                    "fooled": counts["fooled"],
                    "elapsed": elapsed + (time.time() - loop_started),
                    "config_hash": config_hash,
                },
            )
            LOG.info("checkpoint salvo em %d", index + 1)

    elapsed += time.time() - loop_started
    victim_tpr = float(np.mean(scores_after >= attack_threshold))
    normal_tpr_paper = float(np.mean(original_scores >= real_threshold))
    normal_tpr_attack = float(np.mean(original_scores >= attack_threshold))
    result = {
        "schema_version": metrics.SCHEMA_VERSION,
        "status": "completed",
        "run_id": run_id,
        "experiment": "adversarial",
        "model_type": "flow",
        "score_mode": "repo_plus",
        "attack_type": attack_name,
        "threshold": attack_threshold,
        "threshold_mode": "paper_fpr_0.1",
        "real_threshold": real_threshold,
        "attack_threshold": attack_threshold,
        "normal_tpr": normal_tpr_paper,
        "normal_tpr_at_real_threshold": normal_tpr_paper,
        "normal_tpr_at_attack_threshold": normal_tpr_attack,
        "victim_tpr": victim_tpr,
        "attack_success_rate": 1.0 - victim_tpr,
        "fooled_rate": counts["fooled"] / len(x_mal),
        "counts": {**counts, "n": int(len(x_mal))},
        "n_samples": int(len(x_mal)),
        "scores_file": str(scores_path),
        "runtime_seconds": elapsed,
        "config": config,
        "config_hash": config_hash,
        "environment": metrics.environment_manifest(),
        "started_at": started,
        "finished_at": metrics.iso_now(),
    }
    np.savez_compressed(
        scores_path,
        scores_before=original_scores,
        scores_after=scores_after,
        threshold=np.array([attack_threshold]),
        real_threshold=np.array([real_threshold]),
    )
    metrics.write_json_atomic(result_path, result)
    write_manifest(
        paths["results_root"], run_id, config,
        log_path=log_path, result_path=result_path,
        started_at=started, finished_at=metrics.iso_now(),
    )
    LOG.info(
        "concluído: victim_tpr=%.4f attack_success=%.4f normal_tpr=%.4f -> %s",
        victim_tpr, result["attack_success_rate"], normal_tpr_paper, result_path,
    )
    return 0


def run_adversarial_packet(args: argparse.Namespace, paths: dict[str, Path]) -> int:
    started = metrics.iso_now()
    attack_name = args.attack
    run_id = make_run_id("adversarial", "packet", attack_name)
    log_path = paths["results_root"] / "logs" / f"{run_id}.log"
    setup_logging(log_path)
    configure_gpu()
    t = tf()
    t.keras.utils.set_random_seed(args.seed)

    data_dir = paths["data_dir"]
    model_dir = paths["model_dir"]
    norm_dir = paths["norm_dir"]
    output_dir = paths["results_root"] / "packet" / "adversarial"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = _result_path(output_dir, attack_name)
    checkpoint_path = _checkpoint_path(output_dir, attack_name)
    scores_path = output_dir / f"scores_{metrics.sanitize_name(attack_name)}.npz"

    label = metrics.PACKET_ATTACK_TO_LABEL.get(attack_name)
    if label is None:
        raise ValueError(f"ataque de pacotes desconhecido: {attack_name!r}")
    day = packet_day_for_attack(attack_name)
    threshold = float(args.threshold or metrics.PAPER_THRESHOLDS["packet"]["fpr_0.1"])
    max_samples = args.max_samples or 25000
    timesteps = 20
    num_input = 29

    config = result_config(
        args, model_dir, norm_dir, data_dir,
        experiment="adversarial", score_mode="repo_plus", attack=attack_name,
        threshold=threshold, max_samples=max_samples,
    )
    config.update({"day": day, "label": label, "timesteps": timesteps})
    config_hash = metrics.config_hash(config)

    if _completed_result(result_path, config_hash) and not args.force:
        LOG.info("resultado já concluído para %s; nada a fazer", attack_name)
        return 0
    if args.force:
        for path in (result_path, checkpoint_path, scores_path):
            if path.exists():
                path.unlink()

    LOG.info("run_id=%s ataque=%s dia=%s threshold=%.9g", run_id, attack_name, day, threshold)
    x_day, y_day = load_packet_day(data_dir, day)
    x_mal = x_day[y_day == label]
    if len(x_mal) == 0:
        raise ValueError(f"nenhum pacote para {attack_name!r} no dia {day}")
    x_mal = np.concatenate(
        (np.zeros((timesteps - 1, num_input), dtype=x_mal.dtype), x_mal), axis=0
    )
    x_mal = x_mal[:max_samples].astype(np.float32)
    LOG.info("pacotes do ataque=%d", len(x_mal))

    train_min, train_max = load_meta(norm_dir)
    model = load_saved_model(model_dir)
    attack = PacketAttack(model, train_min, train_max, threshold=threshold)

    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint is not None:
        saved_hash = str(checkpoint.get("config_hash", ""))
        if saved_hash != config_hash:
            if not args.force:
                raise RuntimeError(
                    "checkpoint encontrado com configuração diferente; use --force "
                    f"({checkpoint_path})"
                )
            checkpoint = None

    if checkpoint is None:
        LOG.info("pontuando pacotes originais...")
        score_np = np.zeros(len(x_mal), dtype=np.float64)
        start_index = timesteps - 1
        for index in range(len(x_mal) - timesteps):
            score_np[start_index + index] = attack.score(
                x_mal[index : index + timesteps], seed=args.seed + index
            )
            if index % args.log_every == 0:
                LOG.info("normal %d/%d", index, len(x_mal) - timesteps)
        normal_scores = score_np[timesteps:]
        stream = [x_mal[index] for index in range(timesteps - 1)]
        next_index = timesteps - 1
        counts = {"cons_as_mal": 0, "cons_as_ben": 0, "fooled": 0}
        elapsed = 0.0
    else:
        next_index = int(checkpoint["next_index"])
        normal_scores = checkpoint["normal_scores"]
        stream = [np.asarray(packet) for packet in checkpoint["stream"]]
        counts = {
            "cons_as_mal": int(checkpoint["cons_as_mal"]),
            "cons_as_ben": int(checkpoint["cons_as_ben"]),
            "fooled": int(checkpoint["fooled"]),
        }
        elapsed = float(checkpoint["elapsed"])
        attack.load_optimizer_state(checkpoint)
        LOG.info("retomando do checkpoint em %d/%d", next_index, len(x_mal))

    normal_tpr = float(np.mean(normal_scores >= threshold))
    loop_started = time.time()
    for index in range(next_index, len(x_mal)):
        if stop_requested():
            _save_checkpoint(
                checkpoint_path,
                {
                    **attack.optimizer_state(),
                    "next_index": index,
                    "normal_scores": normal_scores,
                    "stream": np.asarray(stream),
                    "cons_as_mal": counts["cons_as_mal"],
                    "cons_as_ben": counts["cons_as_ben"],
                    "fooled": counts["fooled"],
                    "elapsed": elapsed + (time.time() - loop_started),
                    "config_hash": config_hash,
                },
            )
            LOG.warning("interrompido em %d/%d; checkpoint salvo", index, len(x_mal))
            return 143

        t.random.set_seed(args.seed + 10_000 + index)
        sample = np.zeros((1, timesteps, num_input), dtype=np.float32)
        sample[0, : timesteps - 1] = np.asarray(stream[-timesteps + 1 :])
        sample[0, timesteps - 1] = x_mal[index]
        outcome = attack.find_adv(np.copy(sample))

        if outcome is None:
            stream.append(x_mal[index])
            counts["cons_as_mal"] += 1
        elif isinstance(outcome, str) and outcome == "cons_as_ben":
            stream.append(x_mal[index])
            counts["cons_as_ben"] += 1
        elif outcome[0] == "split":
            counts["fooled"] += 1
            for packet in outcome[1]:
                stream.append(packet[0, -1])
        elif outcome[0] == "inject":
            counts["fooled"] += 1
            stream.append(outcome[1])
            stream.append(x_mal[index])
        else:
            raise RuntimeError(f"resultado inesperado do ataque: {outcome!r}")

        if index % args.log_every == 0:
            LOG.info(
                "# %d %.2fmin mal=%d ben=%d fooled=%d",
                index, (elapsed + time.time() - loop_started) / 60.0,
                counts["cons_as_mal"], counts["cons_as_ben"], counts["fooled"],
            )
        if args.checkpoint_every and (index + 1) % args.checkpoint_every == 0:
            _save_checkpoint(
                checkpoint_path,
                {
                    **attack.optimizer_state(),
                    "next_index": index + 1,
                    "normal_scores": normal_scores,
                    "stream": np.asarray(stream),
                    "cons_as_mal": counts["cons_as_mal"],
                    "cons_as_ben": counts["cons_as_ben"],
                    "fooled": counts["fooled"],
                    "elapsed": elapsed + (time.time() - loop_started),
                    "config_hash": config_hash,
                },
            )
            LOG.info("checkpoint salvo em %d", index + 1)

    elapsed += time.time() - loop_started
    n = len(x_mal)
    victim_tpr = counts["cons_as_mal"] / n
    processed = max(1, n - (timesteps - 1))
    result = {
        "schema_version": metrics.SCHEMA_VERSION,
        "status": "completed",
        "run_id": run_id,
        "experiment": "adversarial",
        "model_type": "packet",
        "score_mode": "repo_plus",
        "attack_type": attack_name,
        "threshold": threshold,
        "threshold_mode": "paper_fpr_0.1",
        "normal_tpr": normal_tpr,
        "victim_tpr": victim_tpr,
        "victim_tpr_processed": counts["cons_as_mal"] / processed,
        "attack_success_rate": 1.0 - victim_tpr,
        "fooled_rate": counts["fooled"] / n,
        "counts": {**counts, "n": n, "n_processed": processed},
        "n_samples": n,
        "scores_file": str(scores_path),
        "runtime_seconds": elapsed,
        "config": config,
        "config_hash": config_hash,
        "environment": metrics.environment_manifest(),
        "started_at": started,
        "finished_at": metrics.iso_now(),
    }
    np.savez_compressed(
        scores_path,
        normal_scores=normal_scores,
        threshold=np.array([threshold]),
    )
    metrics.write_json_atomic(result_path, result)
    write_manifest(
        paths["results_root"], run_id, config,
        log_path=log_path, result_path=result_path,
        started_at=started, finished_at=metrics.iso_now(),
    )
    LOG.info(
        "concluído: victim_tpr=%.4f attack_success=%.4f normal_tpr=%.4f -> %s",
        victim_tpr, result["attack_success_rate"], normal_tpr, result_path,
    )
    return 0


def packet_day_for_attack(attack: str) -> str:
    if attack in {"FTP-Patator", "SSH-Patator"}:
        return "tuesday"
    if attack in {"Slowloris", "Slowhttptest", "Hulk", "GoldenEye", "Heartbleed"}:
        return "wednesday"
    if attack in {"Web-Attack", "Infiltration"}:
        return "thursday"
    return "friday"


def run_adversarial(args: argparse.Namespace, paths: dict[str, Path]) -> int:
    if args.model == "flow":
        return run_adversarial_flow(args, paths)
    return run_adversarial_packet(args, paths)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def run_smoke(args: argparse.Namespace, paths: dict[str, Path]) -> int:
    LOG.info("smoke test: modelo=%s", args.model)
    if args.model == "flow":
        normal_args = argparse.Namespace(**vars(args))
        normal_args.modes = "repo_plus"
        normal_args.max_samples = 128
        run_normal_flow(normal_args, paths)
        adv_args = argparse.Namespace(**vars(args))
        adv_args.attack = "DoS Slowhttptest"
        adv_args.max_samples = 1
        adv_args.checkpoint_every = 1
        adv_args.log_every = 1
        adv_args.real_threshold = None
        adv_args.threshold = None
        run_adversarial_flow(adv_args, paths)
    else:
        normal_args = argparse.Namespace(**vars(args))
        normal_args.modes = "repo_plus"
        normal_args.max_samples = 128
        run_normal_packet(normal_args, paths)
        adv_args = argparse.Namespace(**vars(args))
        adv_args.attack = "FTP-Patator"
        adv_args.max_samples = 64
        adv_args.checkpoint_every = 1
        adv_args.log_every = 1
        adv_args.threshold = None
        run_adversarial_packet(adv_args, paths)
    LOG.info("smoke test concluído")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=("flow", "packet"), required=True)
    parser.add_argument("--model-source", choices=("pretrained", "retrained"), default=os.environ.get("MODEL_SOURCE", "pretrained"))
    parser.add_argument("--model-dir")
    parser.add_argument("--normalization-dir")
    parser.add_argument("--data-root")
    parser.add_argument("--results-root")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", metrics.DEFAULT_SEED)))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    normal = subparsers.add_parser("normal", help="avalia RePO/RePO+ no conjunto de teste")
    add_common_arguments(normal)
    normal.add_argument("--modes", default="repo,repo_plus", help="lista separada por vírgula")

    adversarial = subparsers.add_parser("adversarial", help="executa um ataque adversarial")
    add_common_arguments(adversarial)
    adversarial.add_argument("--attack", required=True)
    adversarial.add_argument("--threshold", type=float, default=None, help="threshold de ataque (default: paper)")
    adversarial.add_argument("--real-threshold", type=float, default=None, help="threshold real do fluxo (default: paper)")
    adversarial.add_argument("--checkpoint-every", type=int, default=50)
    adversarial.add_argument("--log-every", type=int, default=50)

    smoke = subparsers.add_parser("smoke", help="teste rápido de dados/modelo/GPU")
    add_common_arguments(smoke)
    smoke.add_argument("--threshold", type=float, default=None)
    smoke.add_argument("--real-threshold", type=float, default=None)
    smoke.add_argument("--checkpoint-every", type=int, default=1)
    smoke.add_argument("--log-every", type=int, default=1)
    smoke.add_argument("--modes", default="repo_plus")

    subparsers.add_parser("manifest", help="imprime o manifesto do ambiente")
    return parser


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    model_dir = resolve_model_dir(args)
    norm_dir = resolve_norm_dir(args, model_dir)
    data_dir = resolve_data_dir(args)
    results_root = resolve_results_root(args)
    results_root.mkdir(parents=True, exist_ok=True)
    return {
        "model_dir": model_dir,
        "norm_dir": norm_dir,
        "data_dir": data_dir,
        "results_root": results_root,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "manifest":
        import json

        print(json.dumps(metrics.environment_manifest(), indent=2, sort_keys=True))
        return 0

    install_signal_handlers()
    paths = resolve_paths(args)

    if args.command == "normal":
        return run_normal(args, paths)
    if args.command == "adversarial":
        return run_adversarial(args, paths)
    if args.command == "smoke":
        return run_smoke(args, paths)
    parser.error(f"comando desconhecido: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
