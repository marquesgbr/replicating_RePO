"""Streaming sampler of the CICIDS2017 flow CSVs for the mask experiments.

The reference runner loads the four test days at once (~2.3 M rows).  The
variance study only needs a random subset (100 k benign + a few thousand rows
per attack), so this module streams the CSVs in chunks and keeps, for every
label, the rows with the smallest random priorities: a uniform random sample
without replacement, deterministic given the seed.

Sampling *uniformly* (instead of taking the first N rows) matters because the
CICIDS2017 captures are ordered in time and the classes concentrate in time
windows -- the bias documented in ``docs/REPLICATION_ANALYSIS.md``.

The 77 features follow the reference layout: ``Dst Port`` (CSV column 5) plus
the 76 flow features (CSV columns 8-83).  The last column is the label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

__all__ = [
    "BENIGN_LABEL",
    "FLOW_ATTACKS",
    "FLOW_FEATURE_COLUMNS",
    "FLOW_TEST_DAYS",
    "WEB_ATTACK_LABEL",
    "load_flow_meta",
    "normalize_label",
    "sample_flow_test",
]

#: Label of the benign class in CICIDS2017.
BENIGN_LABEL = "BENIGN"

#: Attack labels of the reference flow evaluation (Web Attack excluded).
FLOW_ATTACKS: tuple[str, ...] = (
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "DoS Hulk",
    "DoS GoldenEye",
    "Heartbleed",
    "Infiltration",
    "Bot",
    "PortScan",
    "DDoS",
)

#: Label that aggregates the three web attack variants (excluded from means).
WEB_ATTACK_LABEL = "Web Attack"

#: Test days of the reference flow evaluation.
FLOW_TEST_DAYS: tuple[str, ...] = (
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
)

#: Number of features consumed by the flow model.
N_FEATURES = 77


def normalize_label(value: object) -> str | None:
    """Return the canonical CICIDS2017 label, or ``None`` for junk rows.

    The raw CSVs contain a few malformed rows (a repeated header becomes the
    label ``"Label"`` and a missing label becomes ``"nan"``) and three web
    attack variants that the reference paper reports as a single, excluded
    class.  Normalising here keeps the class rates comparable with the
    reference runner.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "label"}:
        return None
    if raw.upper() == BENIGN_LABEL:
        return BENIGN_LABEL
    if raw.startswith("Web Attack"):
        return WEB_ATTACK_LABEL
    return raw


def _feature_columns(frame: pd.DataFrame) -> np.ndarray:
    """Return the ``(rows, 77)`` feature matrix of one chunk.

    ``Dst Port`` comes from column 5 and the flow features from columns
    8..83; the label column (the last one) is excluded.
    """
    port = frame.iloc[:, 5].to_numpy(dtype=np.float64).reshape(-1, 1)
    features = frame.iloc[:, 8:-1].to_numpy(dtype=np.float64)
    values = np.concatenate([port, features], axis=1)
    if values.shape[1] != N_FEATURES:
        raise ValueError(
            f"expected {N_FEATURES} features, got {values.shape[1]}; "
            "the CSV does not follow the CICIDS2017 flow layout"
        )
    return values


def _keep_rows_by_priority(
    rows: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    priorities: np.ndarray,
    *,
    caps: dict[str, int],
    state: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """Merge one chunk into the per-label reservoir of smallest priorities."""
    for label in np.unique(labels):
        cap = caps.get(str(label))
        if cap is None or cap <= 0:
            continue
        mask = labels == label
        candidate_rows = rows[mask]
        candidate_idx = indices[mask]
        candidate_prio = priorities[mask]

        if label in state:
            old_rows, old_idx, old_prio = state[label]
            candidate_rows = np.concatenate([old_rows, candidate_rows], axis=0)
            candidate_idx = np.concatenate([old_idx, candidate_idx], axis=0)
            candidate_prio = np.concatenate([old_prio, candidate_prio], axis=0)

        if len(candidate_prio) > cap:
            selected = np.argpartition(candidate_prio, cap - 1)[:cap]
        else:
            selected = np.arange(len(candidate_prio))
        state[label] = (
            candidate_rows[selected],
            candidate_idx[selected],
            candidate_prio[selected],
        )


def sample_flow_test(
    data_dir: str | Path,
    *,
    max_benign: int,
    max_per_attack: int,
    seed: int = 2019,
    chunksize: int = 200_000,
    days: Iterable[str] = FLOW_TEST_DAYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a uniform random sample of the flow test set.

    Returns ``(x_raw, labels, sample_indices)``.  ``sample_indices`` are the
    global row indices in the concatenated CSV stream and are the key of the
    stateless masks, so the score of a row never depends on the sample size.
    ``max_benign`` caps the benign rows and ``max_per_attack`` caps every other
    label (a label with fewer rows is kept entirely).
    """
    root = Path(data_dir) / "flow_based"
    files = [root / f"{day}-WH-generate-labeled.csv" for day in days]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "CSVs de fluxo ausentes:\n" + "\n".join(missing)
            + "\nRode bash extract_flow_data.sh."
        )

    caps = {BENIGN_LABEL: int(max_benign)}
    state: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    generator = np.random.default_rng(seed)
    offset = 0

    for path in files:
        for frame in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            rows = _feature_columns(frame)
            raw_labels = frame.iloc[:, -1].to_numpy()
            labels = np.asarray([normalize_label(value) for value in raw_labels])
            valid = np.isfinite(rows).all(axis=1) & np.array(
                [value is not None for value in labels]
            )
            rows = rows[valid]
            labels = labels[valid]
            labels = labels.astype(str)
            indices = offset + np.flatnonzero(valid)
            priorities = generator.random(len(rows))

            # The benign cap comes from the argument; every attack label gets
            # the same cap unless it was configured before.
            for label in np.unique(labels):
                name = str(label)
                caps.setdefault(
                    name,
                    int(max_per_attack) if name != BENIGN_LABEL else int(max_benign),
                )

            _keep_rows_by_priority(
                rows,
                labels,
                indices,
                priorities,
                caps=caps,
                state=state,
            )
            offset += len(frame)

    if not state:
        raise ValueError("no valid flow rows were read")

    x_parts = []
    label_parts = []
    index_parts = []
    # BENIGN first for a stable order; the other labels sorted.
    for label in sorted(state, key=lambda name: (name != BENIGN_LABEL, name)):
        rows, indices, priorities = state[label]
        order = np.argsort(priorities)
        x_parts.append(rows[order].astype(np.float32))
        label_parts.append(np.full(len(order), label, dtype=object))
        index_parts.append(indices[order].astype(np.int64))

    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(label_parts, axis=0),
        np.concatenate(index_parts, axis=0),
    )


def load_flow_meta(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(train_min, train_max)`` normalization arrays."""
    root = Path(data_dir) / "flow_based" / "x_train_meta"
    train_min = np.load(root / "train_min.npy").astype(np.float32)
    train_max = np.load(root / "train_max.npy").astype(np.float32)
    if train_min.shape != (N_FEATURES,) or train_max.shape != (N_FEATURES,):
        raise ValueError(
            f"normalization arrays must have {N_FEATURES} values, got "
            f"{train_min.shape} and {train_max.shape}"
        )
    return train_min, train_max
