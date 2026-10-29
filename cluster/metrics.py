"""Pure metrics and metadata helpers for the Apuana reproduction runner.

This module deliberately avoids importing TensorFlow so that the result
aggregation and the unit tests can run on a login node or on a laptop
without a GPU environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

# Attack names are the labels used by the original flow notebook.
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

FLOW_LABELS: tuple[str, ...] = (
    "BENIGN",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "DoS Hulk",
    "DoS GoldenEye",
    "Heartbleed",
    "Web Attack",
    "Infiltration",
    "Bot",
    "PortScan",
    "DDoS",
)

# Attack names are the labels used by the original packet notebook.
# Web-Attack is intentionally excluded from the paper comparison because the
# packet features only contain headers and the paper excludes web attacks.
PACKET_ATTACKS: tuple[str, ...] = (
    "FTP-Patator",
    "SSH-Patator",
    "Slowloris",
    "Slowhttptest",
    "Hulk",
    "GoldenEye",
    "Heartbleed",
    "Infiltration",
    "Botnet",
    "PortScan",
    "DDoS",
)

PACKET_LABELS: tuple[str, ...] = (
    "Benign",
    "FTP-Patator",
    "SSH-Patator",
    "Slowloris",
    "Slowhttptest",
    "Hulk",
    "GoldenEye",
    "Heartbleed",
    "Web-Attack",
    "Infiltration",
    "Botnet",
    "PortScan",
    "DDoS",
)

PACKET_ATTACK_TO_LABEL: dict[str, int] = {
    "FTP-Patator": 1,
    "SSH-Patator": 2,
    "Slowloris": 3,
    "Slowhttptest": 4,
    "Hulk": 5,
    "GoldenEye": 6,
    "Heartbleed": 7,
    "Web-Attack": 8,
    "Infiltration": 9,
    "Botnet": 10,
    "PortScan": 11,
    "DDoS": 12,
}

EXPECTED_ATTACKS: dict[str, tuple[str, ...]] = {
    "flow": FLOW_ATTACKS,
    "packet": PACKET_ATTACKS,
}

# Thresholds copied from the reference notebooks. The paper states that the
# normal evaluation uses FPR=0.01 and the adversarial evaluation uses FPR=0.1.
# The flow notebook applies a 0.8 factor to the FPR=0.1 threshold to
# compensate for the effect of the group-4 features, as described in
# "Towards Evaluation of NIDSs in Adversarial Setting".
PAPER_THRESHOLDS: dict[str, dict[str, float]] = {
    "flow": {
        "fpr_0.1": 0.016235964372754097,
        "attack": 0.016235964372754097 * 0.8,
    },
    "packet": {
        "fpr_0.1": 0.08106673508882523,
        "attack": 0.08106673508882523,
    },
}

# Average TPRs reported in the paper (Section C and D). These are used only
# for a quick sanity check in the generated summary.
PAPER_AVERAGES: dict[tuple[str, str, str], float] = {
    ("flow", "normal", "repo"): 0.2161,
    ("flow", "normal", "repo_plus"): 0.2549,
    ("packet", "normal", "repo"): 0.2765,
    ("packet", "normal", "repo_plus"): 0.3477,
    ("packet", "adversarial", "repo_plus"): 0.6207,
}

DEFAULT_SEED = 2019
SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Naming / serialization helpers
# ---------------------------------------------------------------------------


def sanitize_name(value: str) -> str:
    """Return a filesystem-safe, lowercase token."""

    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    token = token.strip("-._")
    return token or "item"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically so a killed job cannot leave a half file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True, default=_json_default)
            file.write("\n")
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def config_hash(payload: Mapping[str, Any]) -> str:
    """Stable hash used to decide whether a checkpoint/result can be resumed."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Thresholds and metrics
# ---------------------------------------------------------------------------


def threshold_at_fpr(
    scores: Sequence[float] | np.ndarray,
    fpr: float,
    method: str = "ceil",
) -> float:
    """Reproduce the threshold selection used by the reference notebooks.

    Flow notebooks use ``ceil(n * fpr)``; the packet notebook uses
    ``round(n * fpr)``.  Both choose the index from the sorted benign scores
    counting from the end.
    """

    values = np.asarray(scores, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("cannot compute a threshold from an empty score array")
    if not 0.0 < fpr < 1.0:
        raise ValueError(f"fpr must be in (0, 1), got {fpr!r}")

    if method == "ceil":
        index = int(math.ceil(values.size * fpr))
    elif method == "round":
        index = int(np.round(values.size * fpr))
    else:
        raise ValueError(f"unknown threshold method: {method!r}")

    index = max(1, min(index, values.size))
    return float(np.sort(values)[-index])


def aggregate_mask_errors(
    errors: np.ndarray,
    mode: str = "repo_plus",
    groups: int = 5,
) -> np.ndarray:
    """Aggregate per-mask reconstruction errors into one score per sample.

    ``errors`` must have shape ``(n_samples, n_masks)``.  ``repo`` is the mean
    over all masks (the original RePO score); ``repo_plus`` splits the masks
    into ``groups`` groups, takes the minimum in each group and sums the
    minima (the original RePO+ score).
    """

    array = np.asarray(errors, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"errors must be 2-D, got shape {array.shape}")
    if array.shape[1] == 0:
        raise ValueError("errors must contain at least one mask")

    if mode == "repo":
        return array.mean(axis=1)
    if mode in {"repo_plus", "plus"}:
        if array.shape[1] % groups != 0:
            raise ValueError(
                f"number of masks ({array.shape[1]}) is not divisible by groups ({groups})"
            )
        grouped = array.reshape(array.shape[0], groups, -1)
        return grouped.min(axis=2).sum(axis=1)
    raise ValueError(f"unknown score mode: {mode!r}")


def compute_class_rates(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[Any] | np.ndarray,
    label_names: Sequence[str],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    """Compute FPR (first label) and TPR (remaining labels)."""

    score_array = np.asarray(scores, dtype=np.float64).ravel()
    label_array = np.asarray(labels)
    if score_array.shape[0] != label_array.shape[0]:
        raise ValueError(
            f"scores and labels have different lengths: "
            f"{score_array.shape[0]} != {label_array.shape[0]}"
        )

    rates: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(label_names):
        mask = label_array == (index if np.issubdtype(label_array.dtype, np.integer) else name)
        count = int(mask.sum())
        if count == 0:
            rates[name] = {"rate": None, "count": 0}
            continue
        rate = float(np.mean(score_array[mask] >= threshold))
        rates[name] = {"rate": rate, "count": count}
    return rates


def average_attack_tpr(
    class_rates: Mapping[str, Mapping[str, Any]],
    excluded: Iterable[str] = ("BENIGN", "Benign", "Web Attack", "Web-Attack"),
) -> float | None:
    excluded_set = set(excluded)
    values = [
        float(item["rate"])
        for name, item in class_rates.items()
        if name not in excluded_set and item.get("rate") is not None
    ]
    if not values:
        return None
    return float(np.mean(values))


def normal_tpr(
    scores: Sequence[float] | np.ndarray,
    threshold: float,
) -> float:
    values = np.asarray(scores, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("cannot compute TPR from an empty score array")
    return float(np.mean(values >= threshold))


# ---------------------------------------------------------------------------
# Environment / provenance
# ---------------------------------------------------------------------------


def git_info(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)

    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=str(root), text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }


def environment_manifest() -> dict[str, Any]:
    """Collect the versions that matter for reproducibility.

    TensorFlow is imported lazily so this helper can also be used in a
    lightweight aggregation environment.
    """

    manifest: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_JOB_PARTITION",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_NODELIST",
            )
        },
    }

    try:
        import pandas
        import tensorflow as tf

        manifest["tensorflow"] = tf.__version__
        manifest["keras"] = getattr(tf.keras, "__version__", None)
        manifest["numpy"] = np.__version__
        manifest["pandas"] = pandas.__version__
        try:
            manifest["gpus"] = [
                {
                    "name": device.name,
                    "type": device.device_type,
                }
                for device in tf.config.list_physical_devices("GPU")
            ]
        except Exception as error:  # pragma: no cover - defensive
            manifest["gpus_error"] = repr(error)
    except Exception as error:  # pragma: no cover - depends on the environment
        manifest["tensorflow_error"] = repr(error)

    try:
        manifest["nvidia_smi"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=20,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        manifest["nvidia_smi"] = None

    return manifest


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(model_dir: str | Path) -> dict[str, Any]:
    """Hash the SavedModel files without loading TensorFlow."""

    root = Path(model_dir)
    if not root.exists():
        return {"path": str(root), "exists": False}
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return {
        "path": str(root),
        "exists": True,
        "files": len(files),
        "sha256": digest.hexdigest(),
    }


def expected_attacks(model_type: str) -> tuple[str, ...]:
    try:
        return EXPECTED_ATTACKS[model_type]
    except KeyError as error:
        raise ValueError(f"unknown model type: {model_type!r}") from error


def file_size(path: str | Path) -> int | None:
    try:
        return Path(path).stat().st_size
    except OSError:
        return None


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
