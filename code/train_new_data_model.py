"""Train the RePO autoencoder on the benign split of the new dataset.

The script keeps the architecture and training behaviour of the reference
prototype (five epochs, batch size 256, 75% of the features masked) but makes
the run reproducible and auditable:

* the CSV and output directory are command line arguments;
* ``tf.keras.utils.set_random_seed`` seeds Python, NumPy and TensorFlow;
* the training CSV must contain ``BENIGN`` rows only;
* ``training_manifest.json`` records the data hash, seed, library versions,
  epochs, batch size, final loss and the training matrix shape.

Run it from the repository root with::

    python code/train_new_data_model.py \
        --train-csv data/new_data/new_train.csv \
        --output-dir models/new_data_model \
        --seed 2019 --epochs 5 --batch-size 256
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Model

from new_data_eval.results import environment_manifest, write_json_atomic

#: Repository root (``code/train_new_data_model.py`` -> ``<repo>``).
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Default CSV and output paths of the new dataset model.
DEFAULT_TRAIN_CSV = REPO_ROOT / "data" / "new_data" / "new_train.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models" / "new_data_model"

#: Values of the reference training script.
DEFAULT_SEED = 2019
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 256

#: Fraction of every training row that the RePO mask zeroes.
MASKED_PERCENT = 0.75


class FlowModel(Model):
    """RePO autoencoder of the flow based replication."""

    def __init__(self, input_size):
        super(FlowModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(256, activation="relu")
        self.d2 = Dense(256, activation="relu")
        self.d3 = Dense(256, activation="relu")
        self.drop1 = Dropout(0.5)
        self.d4 = Dense(256, activation="relu")
        self.d5 = Dense(256, activation="relu")
        self.drop2 = Dropout(0.5)
        self.d6 = Dense(256, activation="relu")
        self.d7 = Dense(input_size, activation="relu")
        self.input_size = input_size

        # MUST preserve the order of implemented layers for building the model
        self._layers_to_build = [
            self.d1,
            self.d2,
            self.d3,
            self.drop1,
            self.d4,
            self.d5,
            self.drop2,
            self.d6,
            self.d7,
        ]

    def build(self, input_shape):
        current_shape = tf.TensorShape(input_shape)
        for layer in self._layers_to_build:
            layer.build(current_shape)
            current_shape = tf.TensorShape(layer.compute_output_shape(current_shape))
        super().build(input_shape)

    def call(self, x, training=None):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.drop1(x, training=training)
        x = self.d4(x)
        x = self.d5(x)
        x = self.drop2(x, training=training)
        x = self.d6(x)
        x = self.d7(x)
        x = tf.clip_by_value(x, 0.0, 1.0)
        return x


def assert_benign_labels(labels: Sequence[Any]) -> None:
    """Raise ``ValueError`` unless every training label is ``BENIGN``.

    Labels are stripped and upper-cased before the comparison, matching the
    canonicalisation of the evaluation package, but anything other than
    ``BENIGN`` is rejected: training on mixed labels would silently change the
    methodology.
    """
    values = [str(label).strip().upper() for label in labels]
    unexpected = sorted({value for value in values if value != "BENIGN"})
    if unexpected:
        raise ValueError(
            "The training CSV must contain BENIGN rows only; "
            f"found unexpected labels: {unexpected}"
        )


def get_train_ds(train_csv: str | Path = DEFAULT_TRAIN_CSV) -> np.ndarray:
    """Load, validate and normalize the benign training matrix."""
    x_train, _labels = load_training_data(train_csv)
    return x_train


def load_training_data(
    train_csv: str | Path = DEFAULT_TRAIN_CSV,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the normalized training matrix and its labels.

    ``Dst Port`` (column 5) plus columns 8-83 are the 77 model features, the
    same selection used by the evaluation package.  Rows with non-finite
    features are dropped before the min/max normalization.
    """
    train_csv = Path(train_csv)
    if not train_csv.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")

    frame = pd.read_csv(train_csv)
    feats = frame.iloc[:, 8:]
    ds_port = frame.iloc[:, 5]
    frame = pd.concat([ds_port, feats], axis=1)

    labels = frame.iloc[:, -1].to_numpy()
    assert_benign_labels(labels)

    all_feats = frame.iloc[:, :-1].astype(np.float32).to_numpy()
    known_data_ids = np.isfinite(all_feats).all(axis=1)
    x_train = all_feats[known_data_ids]
    y_train = labels[known_data_ids]

    if x_train.shape[0] == 0:
        raise ValueError(f"No finite training rows found in {train_csv}")

    train_min = np.min(x_train, axis=0)
    train_max = np.max(x_train, axis=0)
    x_train = (x_train - train_min) / (train_max - train_min + 1e-6)
    return x_train, y_train


def make_partial(dset: np.ndarray, masked_percent: float = MASKED_PERCENT) -> np.ndarray:
    """Zero a random fraction of every row (the RePO mask of the training)."""
    mask = np.random.random(dset.shape)
    mask = (mask > masked_percent) * 1.0
    mask = mask.astype(np.float32)
    return dset * mask


@tf.function
def repo_loss(rec_x, x):
    """Mean squared reconstruction error of a masked batch."""
    mse_loss = (rec_x - x) ** 2
    mse_loss = tf.reduce_mean(mse_loss)
    return mse_loss


@tf.function
def train_step(partial_x, x, optimizer):
    """Apply one gradient step and record its loss."""
    with tf.GradientTape() as tape:
        rec_x = model(partial_x, training=True)
        loss = repo_loss(rec_x=rec_x, x=x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)


def train_model(
    x_train: np.ndarray,
    *,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[float]:
    """Train ``model`` and return the mean training loss of every epoch."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    total_batch = len(x_train) // batch_size
    if len(x_train) % batch_size != 0:
        total_batch += 1

    losses: list[float] = []
    start_time = time.time()
    for ep in range(epochs):
        train_loss.reset_state()
        x_train_partial = make_partial(x_train)
        inds = rng.permutation(x_train_partial.shape[0])
        x_train_partial_perm = x_train_partial[inds]
        x_train_perm = x_train[inds]
        for i in range(total_batch):
            x_batch = x_train_partial_perm[i * batch_size : (i + 1) * batch_size]
            y_batch = x_train_perm[i * batch_size : (i + 1) * batch_size]
            train_step(x_batch, y_batch, optimizer)
        epoch_loss = float(train_loss.result().numpy())
        losses.append(epoch_loss)
        print("trained time", time.time() - start_time, ep, "loss:", epoch_loss)
    print("total time", time.time() - start_time)
    return losses


def _training_versions() -> dict[str, Any]:
    """Return the environment manifest with the imported TensorFlow version."""
    versions = environment_manifest()
    if not versions.get("tensorflow"):
        versions["tensorflow"] = getattr(tf, "__version__", None)
    if not versions.get("keras"):
        versions["keras"] = getattr(tf.keras, "__version__", None)
    return versions


def build_training_manifest(
    train_csv: str | Path,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    final_loss: float,
    shape: Sequence[int],
    versions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the provenance document written next to the trained model."""
    import hashlib

    path = Path(train_csv)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    if versions is None:
        versions = _training_versions()
    else:
        versions = dict(versions)

    return {
        "train_csv": str(path),
        "data_sha256": digest.hexdigest(),
        "seed": int(seed),
        "versions": versions,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "final_loss": float(final_loss),
        "shape": [int(value) for value in shape],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _positive_int(value: str) -> int:
    """Return ``value`` as a positive integer for an argparse option."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from exc
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {number}")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the training script."""
    parser = argparse.ArgumentParser(
        description="Train the new dataset RePO autoencoder on BENIGN rows."
    )
    parser.add_argument(
        "--train-csv",
        default=str(DEFAULT_TRAIN_CSV),
        help="CSV holding the benign training rows",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="destination of the exported model and training_manifest.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="seed of Python, NumPy, TensorFlow and the batch permutation",
    )
    parser.add_argument(
        "--epochs",
        type=_positive_int,
        default=DEFAULT_EPOCHS,
        help="number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=DEFAULT_BATCH_SIZE,
        help="training batch size",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the arguments, train the model and write the manifest."""
    global model, rng, train_loss

    args = build_parser().parse_args(argv)
    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    x_train = get_train_ds(args.train_csv)
    input_size = x_train.shape[1]

    model = FlowModel(input_size)
    model.build((None, input_size))
    train_loss = tf.keras.metrics.Mean(name="train_loss")

    losses = train_model(
        x_train, epochs=args.epochs, batch_size=args.batch_size
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    model.export(str(output_dir))

    manifest = build_training_manifest(
        args.train_csv,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        final_loss=losses[-1],
        shape=x_train.shape,
    )
    manifest_path = output_dir / "training_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(f"training manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
