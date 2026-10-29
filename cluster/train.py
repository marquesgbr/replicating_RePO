#!/usr/bin/env python3
"""Treino dos modelos RePO de fluxo e de pacotes no cluster Apuana.

Os hiperparâmetros seguem o artigo e os scripts de referência:
  * fluxo: Monday, 5 épocas, batch 256, Adam 1e-3, máscara 0.75;
  * pacotes: Monday, 30.000 batches (10.000 por LR), batch 512,
    Adam 1e-3/1e-4/1e-5, máscara 0.75.

O modelo é exportado como SavedModel compatível com code/load_model_compat.py
e o min/max de normalização é salvo junto ao modelo.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

CLUSTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = CLUSTER_DIR.parent
if str(CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_DIR))
if str(REPO_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "code"))

import metrics  # noqa: E402

LOG = logging.getLogger("repo-train")


def tf():
    import tensorflow as tensorflow

    return tensorflow


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


def configure_gpu() -> None:
    t = tf()
    for gpu in t.config.list_physical_devices("GPU"):
        try:
            t.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def load_flow_training(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = data_dir / "flow_based" / "Monday-WH-generate-labeled.csv"
    if not path.is_file():
        raise FileNotFoundError(f"CSV de treino ausente: {path}. Rode o setup.")
    LOG.info("carregando treino de fluxo: %s", path)
    frame = pd.read_csv(path)
    features = frame.iloc[:, 8:]
    ds_port = frame.iloc[:, 5]
    frame = pd.concat([ds_port, features], axis=1)
    values = frame.iloc[:, :-1].astype(np.float64).values
    known = (np.any(np.isinf(values), axis=1) + np.any(np.isnan(values), axis=1)) == False  # noqa: E712
    x_train = values[known].astype(np.float32)
    train_min = np.min(x_train, axis=0)
    train_max = np.max(x_train, axis=0)
    normalized = (x_train - train_min) / (train_max - train_min + 1e-6)
    LOG.info("fluxo: x=%s min=%s max=%s", normalized.shape, train_min.shape, train_max.shape)
    return normalized, train_min, train_max


def load_packet_training(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    day_dir = data_dir / "packet_based" / "monday"
    files = sorted(day_dir.glob("part_*.npy"))
    if not files:
        raise FileNotFoundError(f"partes de treino ausentes em {day_dir}. Rode o setup.")
    LOG.info("carregando treino de pacotes: %d arquivos", len(files))
    parts = [np.load(path) for path in files]
    x_train = np.concatenate(parts, axis=0).astype(np.float32)
    train_min = np.min(x_train, axis=0)
    train_max = np.max(x_train, axis=0)
    normalized = (x_train - train_min) / (train_max - train_min + 1e-6)
    LOG.info("pacotes: x=%s min=%s max=%s", normalized.shape, train_min.shape, train_max.shape)
    return normalized, train_min, train_max


def make_partial(dataset: np.ndarray, seed: int, masked_percent: float = 0.75) -> np.ndarray:
    rng = np.random.RandomState(seed)
    mask = (rng.random_sample(dataset.shape) > masked_percent).astype(np.float32)
    return dataset * mask


def export_model(t: Any, model: Any, output_dir: Path, input_shape: tuple[int, ...]) -> None:
    if output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    # Keras 3 exige que o modelo tenha sido chamado ao menos uma vez para
    # inferir a assinatura de entrada do SavedModel.
    model(t.zeros((1, *input_shape), dtype=t.float32), training=False)
    try:
        if hasattr(model, "export"):
            model.export(str(output_dir))
        else:  # pragma: no cover - fallback for older Keras
            t.saved_model.save(model, str(output_dir))
    except Exception as error:
        raise RuntimeError(
            f"falha ao exportar SavedModel em {output_dir}: {error}"
        ) from error

    # Valida o SavedModel com o mesmo adaptador usado pelos notebooks.
    from load_model_compat import load_model

    loaded = load_model(str(output_dir))
    dummy = t.zeros((1, *input_shape), dtype=t.float32)
    output = loaded(dummy, training=False)
    LOG.info("SavedModel validado: saída=%s", getattr(output, "shape", None))


def save_training_metadata(
    output_dir: Path,
    train_min: np.ndarray,
    train_max: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    meta_dir = output_dir / "x_train_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    np.save(meta_dir / "train_min.npy", train_min)
    np.save(meta_dir / "train_max.npy", train_max)
    metrics.write_json_atomic(output_dir / "training_manifest.json", metadata)


def train_flow(args: argparse.Namespace) -> int:
    t = tf()
    t.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    configure_gpu()

    from train_flow_based_model import FlowModel

    data_dir = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir in {(REPO_ROOT / "models" / "flw_model").resolve(), (REPO_ROOT / "models" / "pkt_model").resolve()}:
        raise SystemExit(f"recusando sobrescrever o modelo dos autores: {output_dir}")
    x_train, train_min, train_max = load_flow_training(data_dir)
    input_size = x_train.shape[1]

    model = FlowModel(input_size)
    model.build((None, input_size))
    train_loss = t.keras.metrics.Mean(name="train_loss")

    @t.function
    def train_step(partial_x, x, optimizer):
        with t.GradientTape() as tape:
            reconstructed = model(partial_x, training=True)
            loss = t.reduce_mean(t.square(reconstructed - x))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss.update_state(loss)

    optimizer = t.keras.optimizers.Adam(learning_rate=args.learning_rate)
    rng = np.random.RandomState(args.seed)
    epochs = 1 if args.smoke else args.epochs
    batch_size = args.batch_size
    total_batches = len(x_train) // batch_size
    if len(x_train) % batch_size:
        total_batches += 1
    if args.smoke:
        total_batches = min(total_batches, 10)

    started = time.time()
    for epoch in range(epochs):
        partial = make_partial(x_train, args.seed + epoch)
        indices = rng.permutation(len(x_train))
        partial = partial[indices]
        original = x_train[indices]
        train_loss.reset_state()
        for batch_index in range(total_batches):
            start = batch_index * batch_size
            stop = start + batch_size
            train_step(partial[start:stop], original[start:stop], optimizer)
            if batch_index % 100 == 0:
                LOG.info(
                    "época %d/%d batch %d/%d loss=%.6f t=%.1fs",
                    epoch + 1, epochs, batch_index, total_batches,
                    float(train_loss.result()), time.time() - started,
                )
        LOG.info("época %d concluída: loss=%.6f", epoch + 1, float(train_loss.result()))

    runtime = time.time() - started
    export_model(t, model, output_dir, (input_size,))
    metadata = {
        "schema_version": metrics.SCHEMA_VERSION,
        "model_type": "flow",
        "status": "completed",
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": args.learning_rate,
            "masked_percent": 0.75,
            "seed": args.seed,
            "smoke": args.smoke,
        },
        "n_samples": int(len(x_train)),
        "final_loss": float(train_loss.result()),
        "runtime_seconds": runtime,
        "output_dir": str(output_dir),
        "environment": metrics.environment_manifest(),
        "git": metrics.git_info(REPO_ROOT),
        "finished_at": metrics.iso_now(),
    }
    save_training_metadata(output_dir, train_min, train_max, metadata)
    metrics.write_json_atomic(
        Path(args.results_root).expanduser().resolve() / "training" / "flow.json",
        metadata,
    )
    LOG.info("modelo de fluxo salvo em %s", output_dir)
    return 0


def train_packet(args: argparse.Namespace) -> int:
    t = tf()
    t.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    configure_gpu()

    from train_packet_based_model import PacketModel

    data_dir = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir in {(REPO_ROOT / "models" / "flw_model").resolve(), (REPO_ROOT / "models" / "pkt_model").resolve()}:
        raise SystemExit(f"recusando sobrescrever o modelo dos autores: {output_dir}")
    x_train, train_min, train_max = load_packet_training(data_dir)
    timesteps = 20
    num_input = 29
    if x_train.shape[1] != num_input:
        raise ValueError(f"esperado {num_input} features, encontrado {x_train.shape[1]}")

    model = PacketModel(timesteps, num_input)
    model.build((None, timesteps, num_input))
    train_loss = t.keras.metrics.Mean(name="train_loss")

    @t.function
    def train_step(partial_x, x, optimizer):
        with t.GradientTape() as tape:
            reconstructed = model(partial_x, training=True)
            loss = t.reduce_mean(t.square(reconstructed - x))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss.update_state(loss)

    iterations = 10 if args.smoke else args.iterations
    batch_size = args.batch_size
    last_valid_index = len(x_train) - timesteps - 1
    started = time.time()

    for learning_rate_index, learning_rate in enumerate(args.learning_rates):
        partial = make_partial(x_train, args.seed + learning_rate_index)
        optimizer = t.keras.optimizers.Adam(learning_rate=learning_rate)
        zero_gradients = [
            np.zeros(tuple(variable.shape), dtype=np.float32)
            for variable in model.trainable_variables
        ]
        optimizer.apply_gradients(zip(zero_gradients, model.trainable_variables))
        train_loss.reset_state()

        for iteration in range(iterations):
            x_batch = np.zeros((batch_size, timesteps, num_input), dtype=np.float32)
            y_batch = np.zeros((batch_size, timesteps, num_input), dtype=np.float32)
            position = 0
            while position < batch_size:
                index = np.random.randint(0, last_valid_index)
                if np.sum(x_train[index + timesteps - 1]) == 0:
                    continue
                x_batch[position] = partial[index : index + timesteps]
                y_batch[position] = x_train[index : index + timesteps]
                position += 1
            train_step(x_batch, y_batch, optimizer)
            if iteration % 1000 == 0:
                LOG.info(
                    "lr=%g iteração %d/%d loss=%.6f t=%.1fs",
                    learning_rate, iteration, iterations,
                    float(train_loss.result()), time.time() - started,
                )
        LOG.info("lr=%g concluído: loss=%.6f", learning_rate, float(train_loss.result()))

    runtime = time.time() - started
    export_model(t, model, output_dir, (timesteps, num_input))
    metadata = {
        "schema_version": metrics.SCHEMA_VERSION,
        "model_type": "packet",
        "status": "completed",
        "hyperparameters": {
            "iterations_per_learning_rate": iterations,
            "learning_rates": list(args.learning_rates),
            "batch_size": batch_size,
            "timesteps": timesteps,
            "num_input": num_input,
            "masked_percent": 0.75,
            "seed": args.seed,
            "smoke": args.smoke,
        },
        "n_samples": int(len(x_train)),
        "final_loss": float(train_loss.result()),
        "runtime_seconds": runtime,
        "output_dir": str(output_dir),
        "environment": metrics.environment_manifest(),
        "git": metrics.git_info(REPO_ROOT),
        "finished_at": metrics.iso_now(),
    }
    save_training_metadata(output_dir, train_min, train_max, metadata)
    metrics.write_json_atomic(
        Path(args.results_root).expanduser().resolve() / "training" / "packet.json",
        metadata,
    )
    LOG.info("modelo de pacotes salvo em %s", output_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("flow", "packet"), required=True)
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--output-dir")
    parser.add_argument("--results-root", default=os.environ.get("RESULTS_ROOT", str(REPO_ROOT / "results" / "cluster")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", metrics.DEFAULT_SEED)))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.001, 0.0001, 0.00001])
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size is None:
        args.batch_size = 256 if args.model == "flow" else 512
    if args.output_dir is None:
        suffix = "flw_model" if args.model == "flow" else "pkt_model"
        args.output_dir = str(REPO_ROOT / "models" / "retrained" / suffix)
    output_dir = Path(args.output_dir).expanduser().resolve()
    protected = {
        (REPO_ROOT / "models" / "flw_model").resolve(),
        (REPO_ROOT / "models" / "pkt_model").resolve(),
    }
    if output_dir in protected:
        raise SystemExit(f"recusando sobrescrever o modelo dos autores: {output_dir}")
    results_root = Path(args.results_root).expanduser().resolve()
    log_path = results_root / "logs" / f"train_{args.model}_{metrics.utc_stamp()}.log"
    setup_logging(log_path)
    LOG.info("treino: modelo=%s output=%s smoke=%s", args.model, args.output_dir, args.smoke)
    if args.force and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    if args.model == "flow":
        return train_flow(args)
    return train_packet(args)


if __name__ == "__main__":
    raise SystemExit(main())
