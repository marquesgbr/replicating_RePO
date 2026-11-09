"""Loading, validation and sampling helpers for the new dataset.

The dataset is the processed CIC-ToN-IoT split distributed as
``data/new_data/New-dataset-split-clean.zip`` (Git LFS).  The archive contains
two CSVs with the very same 85 column layout:

===========  ==========================================================
index        content
===========  ==========================================================
0            ``column00`` (row counter added while preprocessing)
1-7          identifiers and metadata (``Flow ID``, IPs, ports, ...)
8-83         76 numeric flow features
84           ``Label``
===========  ==========================================================

The autoencoder consumes 77 features: ``Dst Port`` (index 5) plus the
columns 8-83.  Features are returned **raw** (not normalized); normalization
is applied only when scoring, using the ``train_min``/``train_max`` arrays
committed under ``data/new_data/x_train_meta``.

Rows whose feature block contains ``inf``/``nan`` are dropped and counted;
labels outside the expected set raise :class:`ValueError` instead of being
silently ignored.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "ATTACK_LABELS",
    "EXPECTED_COLUMNS",
    "EXPECTED_LABELS",
    "FEATURE_INDICES",
    "FEATURE_NAMES",
    "LABEL_INDEX",
    "N_FEATURES",
    "Chunk",
    "DatasetInfo",
    "canonical_label",
    "constant_feature_indices",
    "count_classes",
    "feature_frame",
    "integer_feature_indices",
    "iter_test_chunks",
    "iter_test_chunks_detailed",
    "load_metadata",
    "normalize_features",
    "read_header",
    "select_attack_samples",
    "sha256_file",
    "validate_dataset",
    "validate_header",
]


EXPECTED_COLUMNS: tuple[str, ...] = (
    "column00",
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Max",
    "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean",
    "Fwd Pkt Len Std",
    "Bwd Pkt Len Max",
    "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean",
    "Bwd Pkt Len Std",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Tot",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Tot",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Len",
    "Bwd Header Len",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
    "Pkt Len Min",
    "Pkt Len Max",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Len Var",
    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "URG Flag Cnt",
    "CWE Flag Count",
    "ECE Flag Cnt",
    "Down/Up Ratio",
    "Pkt Size Avg",
    "Fwd Seg Size Avg",
    "Bwd Seg Size Avg",
    "Fwd Byts/b Avg",
    "Fwd Pkts/b Avg",
    "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg",
    "Bwd Pkts/b Avg",
    "Bwd Blk Rate Avg",
    "Subflow Fwd Pkts",
    "Subflow Fwd Byts",
    "Subflow Bwd Pkts",
    "Subflow Bwd Byts",
    "Init Fwd Win Byts",
    "Init Bwd Win Byts",
    "Fwd Act Data Pkts",
    "Fwd Seg Size Min",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "Label",
)

#: Index of the ``Label`` column (the last one).
LABEL_INDEX = 84

#: Index of the ``Dst Port`` column, the first model feature.
DST_PORT_INDEX = 5

#: Indices of the 76 numeric flow features (columns 8-83).
FLOW_FEATURE_INDICES: tuple[int, ...] = tuple(range(8, 84))

#: The 77 features consumed by the model: ``Dst Port`` + columns 8-83.
FEATURE_INDICES: tuple[int, ...] = (DST_PORT_INDEX,) + FLOW_FEATURE_INDICES

#: Number of model features.
N_FEATURES = 77

#: Names of the 77 model features, in the order used by the model.
FEATURE_NAMES: tuple[str, ...] = tuple(EXPECTED_COLUMNS[index] for index in FEATURE_INDICES)

#: Every label accepted by the pipeline.
EXPECTED_LABELS: tuple[str, ...] = (
    "BENIGN",
    "xss",
    "password",
    "injection",
    "scanning",
    "backdoor",
    "ransomware",
    "mitm",
    "ddos",
    "dos",
)

#: The nine malicious classes evaluated by the methodology.
ATTACK_LABELS: tuple[str, ...] = EXPECTED_LABELS[1:]

#: Label used for benign traffic.
BENIGN_LABEL = "BENIGN"

#: Default number of rows read per chunk.
DEFAULT_CHUNK_SIZE = 200_000

#: Epsilon used by the normalization formula.
NORMALIZATION_EPS = 1e-6


@dataclass(frozen=True)
class Chunk:
    """A slice of a CSV file.

    ``features`` holds the raw (not normalized) 77 model features, ``labels``
    the canonical labels, ``valid`` the mask of rows with finite features and
    ``indices`` the global row index of every row inside the file (used to
    derive stateless per-sample masks).
    """

    features: np.ndarray
    labels: np.ndarray
    valid: np.ndarray
    indices: np.ndarray

    @property
    def valid_features(self) -> np.ndarray:
        return self.features[self.valid]

    @property
    def valid_labels(self) -> np.ndarray:
        return self.labels[self.valid]


@dataclass(frozen=True)
class DatasetInfo:
    """Summary of a validated dataset pair.

    ``constant_features`` holds the indices of the features that are constant
    in the training split (``train_max == train_min``).  The normalization of
    those features divides by ``1e-6`` only, so their scaled values explode;
    the indices are reported so the validation output can make them visible.
    """

    train_sha256: str
    test_sha256: str
    train_rows: int
    test_rows: int
    test_valid_rows: int
    test_dropped_invalid: int
    class_counts: dict[str, int]
    constant_features: tuple[int, ...] = ()


def canonical_label(value: object) -> str:
    """Return the canonical name of ``value``.

    ``BENIGN`` is normalized to upper case and the attack names to lower case;
    surrounding whitespace is ignored.  Unknown labels raise
    :class:`ValueError` -- an unexpected attack name must never silently
    produce an empty class.
    """
    if value is None:
        raise ValueError(f"Unknown label None; expected one of {list(EXPECTED_LABELS)}")
    if isinstance(value, float) and math.isnan(value):
        raise ValueError(f"Unknown label nan; expected one of {list(EXPECTED_LABELS)}")

    raw = str(value).strip()
    if not raw:
        raise ValueError(f"Empty label; expected one of {list(EXPECTED_LABELS)}")
    if raw.upper() == BENIGN_LABEL:
        return BENIGN_LABEL

    lowered = raw.lower()
    if lowered in ATTACK_LABELS:
        return lowered

    raise ValueError(
        f"Unknown label {raw!r}; expected one of {list(EXPECTED_LABELS)}"
    )


def validate_header(columns: Sequence[str]) -> None:
    """Validate the CSV header against the canonical 85 column layout."""
    names = [str(column) for column in columns]
    if len(names) != len(EXPECTED_COLUMNS):
        raise ValueError(
            f"Expected {len(EXPECTED_COLUMNS)} columns, found {len(names)}: {names[:5]}"
        )
    for index, (expected, found) in enumerate(zip(EXPECTED_COLUMNS, names)):
        if expected != found:
            raise ValueError(
                f"Unexpected column at index {index}: expected {expected!r}, found {found!r}"
            )


def read_header(csv_path: Path) -> list[str]:
    """Return the header of ``csv_path`` without reading the data rows."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    columns = pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns
    return [str(column) for column in columns]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hexadecimal digest of ``path``."""
    path = Path(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_metadata(meta_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the ``train_min``/``train_max`` normalization arrays.

    Both arrays must hold one finite value per model feature.  Features that
    are constant in the training split (``max == min``) are accepted -- the
    committed metadata contains eight of them -- but ``max < min`` is an
    error.
    """
    meta_dir = Path(meta_dir)
    min_path = meta_dir / "train_min.npy"
    max_path = meta_dir / "train_max.npy"
    if not min_path.is_file():
        raise FileNotFoundError(f"Normalization metadata not found: {min_path}")
    if not max_path.is_file():
        raise FileNotFoundError(f"Normalization metadata not found: {max_path}")

    train_min = np.load(min_path).astype(np.float32, copy=False).ravel()
    train_max = np.load(max_path).astype(np.float32, copy=False).ravel()

    if train_min.shape != (N_FEATURES,) or train_max.shape != (N_FEATURES,):
        raise ValueError(
            f"Normalization metadata must have shape ({N_FEATURES},); "
            f"found {train_min.shape} and {train_max.shape}"
        )
    if not np.isfinite(train_min).all() or not np.isfinite(train_max).all():
        raise ValueError("Normalization metadata must be finite")
    if (train_max < train_min).any():
        raise ValueError("train_max must be greater than or equal to train_min")

    return train_min, train_max


def constant_feature_indices(
    train_min: np.ndarray, train_max: np.ndarray
) -> list[int]:
    """Indices of the features that are constant in the training split.

    ``validate_dataset`` reports them in ``DatasetInfo.constant_features`` so
    the validation output can make them visible (the normalization of those
    features divides by the ``1e-6`` epsilon only).
    """
    train_min = np.asarray(train_min).ravel()
    train_max = np.asarray(train_max).ravel()
    return [int(index) for index in np.flatnonzero(train_max == train_min)]


def normalize_features(
    x: np.ndarray, train_min: np.ndarray, train_max: np.ndarray
) -> np.ndarray:
    """Scale ``x`` with the training bounds: ``(x - min) / (max - min + 1e-6)``."""
    values = np.asarray(x, dtype=np.float32)
    train_min = np.asarray(train_min, dtype=np.float32).ravel()
    train_max = np.asarray(train_max, dtype=np.float32).ravel()

    if values.ndim != 2 or values.shape[1] != N_FEATURES:
        raise ValueError(
            f"Expected a ({values.shape[0]}, {N_FEATURES}) feature matrix, "
            f"got {values.shape}"
        )
    if train_min.shape != (N_FEATURES,) or train_max.shape != (N_FEATURES,):
        raise ValueError(
            f"Expected ({N_FEATURES},) normalization arrays, got "
            f"{train_min.shape} and {train_max.shape}"
        )

    return (values - train_min) / (train_max - train_min + NORMALIZATION_EPS)


def feature_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the model features, the canonical labels and a validity mask.

    ``frame`` may either use the full canonical header (validated in order) or
    contain only the columns required by the model (selected by name, which is
    what the chunked reader does).  The returned features are **raw**: the
    normalization is applied only when scoring.
    """
    columns = [str(column) for column in frame.columns]

    if len(columns) == len(EXPECTED_COLUMNS):
        validate_header(columns)
        feature_block = frame.iloc[:, list(FEATURE_INDICES)]
        label_block = frame.iloc[:, LABEL_INDEX]
    elif len(columns) < len(EXPECTED_COLUMNS):
        missing = [name for name in FEATURE_NAMES if name not in columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        if "Label" not in columns:
            raise ValueError("Missing label column 'Label'")
        feature_block = frame.loc[:, list(FEATURE_NAMES)]
        label_block = frame["Label"]
    else:
        raise ValueError(
            f"Expected at most {len(EXPECTED_COLUMNS)} columns, found "
            f"{len(columns)}"
        )

    if any(
        not pd.api.types.is_numeric_dtype(dtype) for dtype in feature_block.dtypes
    ):
        # A corrupt cell becomes NaN, so the row is dropped and counted
        # instead of aborting the whole run.
        feature_block = feature_block.apply(pd.to_numeric, errors="coerce")
    values = feature_block.to_numpy(dtype=np.float32, copy=True)
    raw_labels = label_block.to_numpy()
    labels = np.asarray(
        [canonical_label(value) for value in raw_labels], dtype=np.str_
    )
    valid = np.isfinite(values).all(axis=1)
    return values, labels, valid


def iter_test_chunks_detailed(
    csv_path: Path, chunksize: int = DEFAULT_CHUNK_SIZE
) -> Iterator[Chunk]:
    """Yield one :class:`Chunk` per block of rows of ``csv_path``.

    The header is validated before any row is read.  Invalid rows (non finite
    features) are flagged, not dropped, so callers can count them.
    """
    csv_path = Path(csv_path)
    validate_header(read_header(csv_path))

    usecols = list(FEATURE_NAMES) + ["Label"]
    reader = pd.read_csv(
        csv_path, usecols=usecols, chunksize=chunksize, encoding="utf-8-sig"
    )
    offset = 0
    try:
        for frame in reader:
            features, labels, valid = feature_frame(frame)
            n_rows = len(frame)
            indices = np.arange(offset, offset + n_rows, dtype=np.int64)
            offset += n_rows
            yield Chunk(
                features=features, labels=labels, valid=valid, indices=indices
            )
    finally:
        reader.close()


def iter_test_chunks(
    csv_path: Path, chunksize: int = DEFAULT_CHUNK_SIZE
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(features, labels)`` for the valid rows of ``csv_path``.

    Rows with non finite features are dropped here; use
    :func:`iter_test_chunks_detailed` to also count them.
    """
    for chunk in iter_test_chunks_detailed(csv_path, chunksize):
        yield chunk.valid_features, chunk.valid_labels


def count_classes(
    csv_path: Path, chunksize: int = DEFAULT_CHUNK_SIZE
) -> dict[str, int]:
    """Count the valid rows per canonical label."""
    counts: dict[str, int] = {}
    for chunk in iter_test_chunks_detailed(csv_path, chunksize):
        labels = chunk.valid_labels
        if len(labels) == 0:
            continue
        values, numbers = np.unique(labels, return_counts=True)
        for value, number in zip(values, numbers):
            key = str(value)
            counts[key] = counts.get(key, 0) + int(number)
    return dict(sorted(counts.items()))


def integer_feature_indices(
    csv_path: Path, chunksize: int = DEFAULT_CHUNK_SIZE
) -> list[int]:
    """Return the model features that are integer valued in every valid row.

    The reference protocol derives this list once from the concatenated,
    valid test matrix, independently of the attacked class.  The CSV is
    scanned in chunks and only the per-feature ``sum(abs(x - round(x)))`` is
    accumulated; a feature is returned when that sum is exactly zero.  This
    mirrors the reference ``sum(abs(x - round(x))) == 0`` test without
    materializing the whole file.
    """
    deviation = np.zeros(N_FEATURES, dtype=np.float64)
    for chunk in iter_test_chunks_detailed(csv_path, chunksize):
        if not chunk.valid.any():
            continue
        values = chunk.valid_features.astype(np.float64, copy=False)
        deviation += np.abs(values - np.round(values)).sum(axis=0)
    return [index for index in range(N_FEATURES) if deviation[index] == 0.0]


def _scan_csv(
    csv_path: Path, chunksize: int = DEFAULT_CHUNK_SIZE
) -> tuple[int, int, dict[str, int]]:
    """Return ``(rows, valid_rows, class_counts)`` for ``csv_path``."""
    rows = 0
    valid_rows = 0
    counts: dict[str, int] = {}
    for chunk in iter_test_chunks_detailed(csv_path, chunksize):
        rows += len(chunk.labels)
        valid_rows += int(chunk.valid.sum())
        if len(chunk.labels) == 0:
            continue
        values, numbers = np.unique(chunk.valid_labels, return_counts=True)
        for value, number in zip(values, numbers):
            key = str(value)
            counts[key] = counts.get(key, 0) + int(number)
    return rows, valid_rows, dict(sorted(counts.items()))


def _apply_reservoir_draws(
    reservoir_x: np.ndarray,
    reservoir_idx: np.ndarray,
    draws: np.ndarray,
    rows: np.ndarray,
    indices: np.ndarray,
) -> None:
    """Apply the Algorithm R replacements, keeping the last draw per slot.

    ``draws[j]`` is the reservoir slot chosen by the ``j``-th candidate.  When
    two candidates choose the same slot the later one wins, which is the
    sequential behaviour of the algorithm.  The last draw of every slot is
    selected explicitly (reverse + ``np.unique``) instead of relying on the
    order of repeated indices in NumPy advanced assignment, which is not
    documented and therefore must not be part of the correctness argument.
    """
    if len(draws) == 0:
        return
    slots, first_positions = np.unique(draws[::-1], return_index=True)
    reservoir_x[slots] = rows[::-1][first_positions]
    reservoir_idx[slots] = indices[::-1][first_positions]


def select_attack_samples(
    csv_path: Path,
    attack: str,
    max_samples: int,
    mode: str,
    seed: int,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> tuple[np.ndarray, int]:
    """Select samples of a single attack class.

    ``mode="random"`` uses reservoir sampling (algorithm R) with a seeded
    generator, so the selection depends only on ``seed`` and on the order of
    the rows in the file -- never on ``chunksize``.  ``mode="first"`` keeps the
    first ``max_samples`` rows of the class (the paper's biased protocol,
    kept as a sensitivity analysis).

    Returns the selected **raw** features (ordered by their position in the
    file) and the total number of valid rows available for the class.
    """
    target = canonical_label(attack)
    if target not in ATTACK_LABELS:
        raise ValueError(
            f"Unknown attack label {target!r}; expected one of {list(ATTACK_LABELS)}"
        )
    if mode not in ("random", "first"):
        raise ValueError(f"Unknown sampling mode {mode!r}; expected 'random' or 'first'")
    if max_samples < 0:
        raise ValueError(f"max_samples must not be negative, got {max_samples}")

    rng = np.random.default_rng(seed)
    reservoir_x = np.empty((0, N_FEATURES), dtype=np.float32)
    reservoir_idx = np.empty((0,), dtype=np.int64)
    prefix_x = np.empty((0, N_FEATURES), dtype=np.float32)
    prefix_idx = np.empty((0,), dtype=np.int64)
    available = 0
    seen = 0

    for chunk in iter_test_chunks_detailed(csv_path, chunksize):
        mask = chunk.valid & (chunk.labels == target)
        rows = chunk.features[mask]
        indices = chunk.indices[mask]
        n_rows = len(rows)
        if n_rows == 0:
            continue

        available += n_rows
        ranks = np.arange(seen + 1, seen + n_rows + 1, dtype=np.int64)
        seen += n_rows

        if mode == "first":
            room = max_samples - len(prefix_x)
            if room > 0:
                take = min(room, n_rows)
                prefix_x = np.concatenate([prefix_x, rows[:take]])
                prefix_idx = np.concatenate([prefix_idx, indices[:take]])
            continue

        if len(reservoir_x) < max_samples:
            room = max_samples - len(reservoir_x)
            take = min(room, n_rows)
            reservoir_x = np.concatenate([reservoir_x, rows[:take]])
            reservoir_idx = np.concatenate([reservoir_idx, indices[:take]])
            rows = rows[take:]
            indices = indices[take:]
            ranks = ranks[take:]
            if len(rows) == 0:
                continue

        if max_samples > 0 and len(rows) > 0:
            # Algorithm R: the i-th item replaces a random slot with
            # probability k/i.
            draws = rng.integers(0, ranks)
            keep = draws < max_samples
            if keep.any():
                _apply_reservoir_draws(
                    reservoir_x,
                    reservoir_idx,
                    draws[keep],
                    rows[keep],
                    indices[keep],
                )

    if mode == "first":
        selected_x, selected_idx = prefix_x, prefix_idx
    else:
        selected_x, selected_idx = reservoir_x, reservoir_idx

    order = np.argsort(selected_idx, kind="stable")
    return selected_x[order], available


def validate_dataset(
    train_csv: Path, test_csv: Path, meta_dir: Path
) -> DatasetInfo:
    """Validate the dataset pair and return a summary.

    The checks are:

    * both CSVs exist and use the canonical 85 column header;
    * the normalization metadata holds 77 finite values per array;
    * the training CSV contains ``BENIGN`` rows only;
    * the test CSV contains ``BENIGN`` and the nine attack classes;
    * labels outside the expected set abort the validation.

    The summary also reports the features that are constant in the training
    split (``constant_features``), because their normalization divides by the
    ``1e-6`` epsilon only.
    """
    train_csv = Path(train_csv)
    test_csv = Path(test_csv)
    meta_dir = Path(meta_dir)

    for path in (train_csv, test_csv):
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")

    validate_header(read_header(train_csv))
    validate_header(read_header(test_csv))
    train_min, train_max = load_metadata(meta_dir)

    train_rows, _train_valid, train_counts = _scan_csv(train_csv)
    if BENIGN_LABEL not in train_counts:
        raise ValueError(f"{train_csv} contains no BENIGN rows")
    unexpected = sorted(label for label in train_counts if label != BENIGN_LABEL)
    if unexpected:
        raise ValueError(
            f"{train_csv} must contain BENIGN rows only; found {unexpected}"
        )

    test_rows, test_valid_rows, class_counts = _scan_csv(test_csv)
    missing = [
        label
        for label in (BENIGN_LABEL,) + ATTACK_LABELS
        if label not in class_counts
    ]
    if missing:
        raise ValueError(f"{test_csv} is missing expected classes: {missing}")

    return DatasetInfo(
        train_sha256=sha256_file(train_csv),
        test_sha256=sha256_file(test_csv),
        train_rows=train_rows,
        test_rows=test_rows,
        test_valid_rows=test_valid_rows,
        test_dropped_invalid=test_rows - test_valid_rows,
        class_counts=class_counts,
        constant_features=tuple(constant_feature_indices(train_min, train_max)),
    )
