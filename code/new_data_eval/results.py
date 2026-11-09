"""Result schemas, atomic JSON persistence and provenance helpers.

Every evaluation run writes exactly one JSON document through
:func:`write_json_atomic` (a temporary file is written in the destination
directory and then ``os.replace``d over the target), so a killed job can never
leave a half written result and no run ever appends to a previous one.

The builders assemble the schema of the design document
(``docs/superpowers/specs/2026-11-09-new-dataset-repo-methodology-design.md``,
section 9) and compute the metrics from the scores.  They deliberately accept
keyword arguments so the normal and adversarial CLIs can pass only the pieces
they have; every missing required piece raises :class:`TypeError`.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from new_data_eval.data import EXPECTED_LABELS
from new_data_eval.metrics import (
    average_attack_tpr,
    class_rates,
    conditional_drop,
    fpr,
)

__all__ = [
    "MODEL_TYPE",
    "SCHEMA_VERSION",
    "build_adversarial_result",
    "build_normal_result",
    "config_hash",
    "environment_manifest",
    "file_sha256",
    "git_commit",
    "write_json_atomic",
]


#: Schema version stored in every result document.
SCHEMA_VERSION = "new_data_v1"

#: Model type that distinguishes the new dataset from the flow/packet runs.
MODEL_TYPE = "new_data_flow"

#: Default variant of a result document (the main result, not a sensitivity).
DEFAULT_VARIANT = "primary"

#: The two thresholds of the normal result.
NORMAL_THRESHOLD_NAMES: tuple[str, ...] = ("fpr_0.01", "fpr_0.10")

_MISSING = object()
_NO_DEFAULT = object()


def _json_default(value: Any) -> Any:
    """Serialize NumPy/Path values that are not JSON native."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically, replacing any previous file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hexadecimal digest of ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    """Return the installed version of ``name`` without importing it."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment_manifest() -> dict[str, Any]:
    """Return the runtime versions relevant for reproducibility."""
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "tensorflow": _package_version("tensorflow"),
        "keras": _package_version("keras"),
    }


def config_hash(config: Mapping[str, Any]) -> str:
    """Return the stable SHA-256 of ``config``.

    The mapping is serialized with sorted keys and compact separators, so the
    hash depends on the values (seed, thresholds, sampling, model hash, data
    hash, ...) and not on the insertion order of the dictionary.
    """
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), default=_json_default
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _repository_root() -> Path:
    """Return the repository root that contains this module."""
    return Path(__file__).resolve().parents[2]


def git_commit() -> str | None:
    """Return the current git commit of the repository, or ``None``."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_repository_root()),
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:  # pragma: no cover - depends on the host
        return None
    commit = completed.stdout.strip()
    if completed.returncode != 0 or not commit:
        return None
    return commit


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _take(
    kwargs: dict[str, Any], name: str, default: Any = _NO_DEFAULT, *, function: str
) -> Any:
    """Pop a keyword argument, raising a clear error when it is required."""
    if name in kwargs:
        return kwargs.pop(name)
    if default is not _NO_DEFAULT:
        return default
    raise TypeError(
        f"{function}() missing required keyword argument: {name!r}"
    )


def _reject_extra(kwargs: Mapping[str, Any], *, function: str) -> None:
    """Reject keyword arguments that the builder does not understand."""
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"{function}() got unexpected keyword arguments: {names}")


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    """Return a shallow ``dict`` copy of a mapping argument."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value)!r}")
    return dict(value)


def _field(mapping: Mapping[str, Any], key: str, *names: str) -> Any:
    """Return ``mapping[key]``, raising a ``ValueError`` when it is absent."""
    try:
        return mapping[key]
    except KeyError as exc:
        label = ".".join((*names, key)) if names else key
        raise ValueError(f"missing required field: {label}") from exc


def _timestamp(value: Any) -> str:
    """Return an ISO 8601 timestamp, using the current UTC time for ``None``."""
    if value is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    if isinstance(value, datetime):
        moment = value
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat(timespec="seconds")
    return str(value)


def _runtime(value: Any) -> float:
    """Return a non-negative finite runtime in seconds."""
    seconds = float(value)
    if not np.isfinite(seconds) or seconds < 0.0:
        raise ValueError(f"runtime_seconds must be finite and non-negative, got {value!r}")
    return seconds


def _environment(value: Any) -> dict[str, Any]:
    """Return the environment manifest, computing it when not supplied."""
    if value is None:
        return environment_manifest()
    return _as_mapping(value, "environment")


def _commit(value: Any) -> str | None:
    """Return the commit to store; look it up in git when not supplied."""
    if value is _MISSING:
        return git_commit()
    return None if value is None else str(value)


def _normal_dataset(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Return the dataset block of a normal result."""
    return {
        "name": str(_field(dataset, "name", "dataset")),
        "test_csv": str(_field(dataset, "test_csv", "dataset")),
        "test_csv_sha256": str(_field(dataset, "test_csv_sha256", "dataset")),
        "n_samples": int(_field(dataset, "n_samples", "dataset")),
        "n_dropped_invalid": int(_field(dataset, "n_dropped_invalid", "dataset")),
    }


def _adversarial_dataset(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Return the dataset sampling block of an adversarial result."""
    return {
        "n_available": int(_field(dataset, "n_available", "dataset")),
        "n_samples": int(_field(dataset, "n_samples", "dataset")),
        "sample_mode": str(_field(dataset, "sample_mode", "dataset")),
        "seed": int(_field(dataset, "seed", "dataset")),
    }


def _normal_thresholds(thresholds: Mapping[str, Any]) -> dict[str, float]:
    """Return the two normal thresholds as floats."""
    return {
        name: float(_field(thresholds, name, "thresholds"))
        for name in NORMAL_THRESHOLD_NAMES
    }


def _adversarial_thresholds(thresholds: Mapping[str, Any]) -> dict[str, Any]:
    """Return the adversarial threshold block."""
    return {
        "fpr_0.10": float(_field(thresholds, "fpr_0.10", "thresholds")),
        "attack": float(_field(thresholds, "attack", "thresholds")),
        "policy": str(_field(thresholds, "policy", "thresholds")),
    }


def _normal_scoring(scoring: Mapping[str, Any]) -> dict[str, Any]:
    """Return the scoring block of a normal result."""
    return {
        "n_masks": int(_field(scoring, "n_masks", "scoring")),
        "score_repeats": int(_field(scoring, "score_repeats", "scoring")),
        "paired_scoring": bool(_field(scoring, "paired_scoring", "scoring")),
    }


def _adversarial_scoring(
    score_mode: str, scoring: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the scoring block of an adversarial result."""
    return {
        "score_mode": str(
            _field(scoring, "score_mode", "scoring")
            if "score_mode" in scoring
            else score_mode
        ),
        "n_masks": int(_field(scoring, "n_masks", "scoring")),
        "score_repeats": int(_field(scoring, "score_repeats", "scoring")),
        "paired_scoring": bool(_field(scoring, "paired_scoring", "scoring")),
    }


def _counts(counts: Mapping[str, Any]) -> dict[str, int]:
    """Return the adversarial count block with native ints."""
    payload = {
        key: int(_field(counts, key, "counts"))
        for key in ("n", "fooled", "evaded", "backup", "cons_as_mal", "cons_as_ben")
    }
    if payload["n"] < 0:
        raise ValueError(f"counts['n'] must not be negative, got {payload['n']}")
    return payload


def _rate_or_compute(
    kwargs: dict[str, Any],
    name: str,
    scores_name: str,
    threshold: float,
    expected_n: int | None,
) -> float:
    """Return a precomputed rate or compute it from the supplied scores."""
    if name in kwargs:
        return float(kwargs.pop(name))
    if scores_name not in kwargs:
        raise TypeError(
            "build_adversarial_result() requires either "
            f"{name!r} or {scores_name!r}"
        )
    scores = np.asarray(kwargs.pop(scores_name), dtype=np.float64).ravel()
    if expected_n is not None and scores.shape[0] != expected_n:
        raise ValueError(
            f"{scores_name} has {scores.shape[0]} values but counts['n'] is "
            f"{expected_n}"
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"{scores_name} must be finite")
    if scores.size == 0:
        return 0.0
    return float(np.mean(scores >= threshold))


# ---------------------------------------------------------------------------
# Result builders
# ---------------------------------------------------------------------------


def build_normal_result(**kwargs: Any) -> dict[str, Any]:
    """Build the JSON document of a normal evaluation.

    Required keyword arguments:

    ``score_mode``
        ``"repo"`` or ``"repo_plus"``.
    ``thresholds``
        Mapping with ``fpr_0.01`` and ``fpr_0.10``.
    ``scores`` / ``labels``
        The scores and canonical labels of the evaluated samples.
    ``dataset``
        Mapping with ``name``, ``test_csv``, ``test_csv_sha256``, ``n_samples``
        and ``n_dropped_invalid``.
    ``scoring``
        Mapping with ``n_masks``, ``score_repeats`` and ``paired_scoring``.
    ``config``
        The run configuration; it is hashed into ``config_hash``.
    ``runtime_seconds``
        Wall clock runtime of the run.

    Optional keyword arguments: ``variant`` (default ``"primary"``),
    ``status`` (default ``"completed"``), ``started_at``, ``finished_at``,
    ``environment`` (computed when absent) and ``git_commit`` (looked up when
    absent).
    """
    function = "build_normal_result"
    score_mode = str(_take(kwargs, "score_mode", function=function))
    thresholds = _as_mapping(
        _take(kwargs, "thresholds", function=function), "thresholds"
    )
    scores = _take(kwargs, "scores", function=function)
    labels = _take(kwargs, "labels", function=function)
    dataset = _as_mapping(_take(kwargs, "dataset", function=function), "dataset")
    scoring = _as_mapping(_take(kwargs, "scoring", function=function), "scoring")
    config = _as_mapping(_take(kwargs, "config", function=function), "config")
    runtime_seconds = _take(kwargs, "runtime_seconds", function=function)

    variant = str(_take(kwargs, "variant", DEFAULT_VARIANT, function=function))
    status = str(_take(kwargs, "status", "completed", function=function))
    started_at = _take(kwargs, "started_at", None, function=function)
    finished_at = _take(kwargs, "finished_at", None, function=function)
    environment = _take(kwargs, "environment", None, function=function)
    commit = _take(kwargs, "git_commit", _MISSING, function=function)
    _reject_extra(kwargs, function=function)

    score_array = np.asarray(scores, dtype=np.float64).ravel()
    label_array = np.asarray(labels).ravel()
    if score_array.shape[0] != label_array.shape[0]:
        raise ValueError(
            "scores and labels have different lengths: "
            f"{score_array.shape[0]} != {label_array.shape[0]}"
        )
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite")

    threshold_payload = _normal_thresholds(thresholds)
    rates = {
        f"at_{name}": class_rates(
            score_array, label_array, threshold, EXPECTED_LABELS
        )
        for name, threshold in threshold_payload.items()
    }
    fpr_payload = {
        f"at_{name}_threshold": fpr(score_array, label_array, threshold)
        for name, threshold in threshold_payload.items()
    }
    averages = {
        f"at_{name}": average_attack_tpr(rates[f"at_{name}"])
        for name in threshold_payload
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "experiment": "normal",
        "model_type": MODEL_TYPE,
        "score_mode": score_mode,
        "dataset": _normal_dataset(dataset),
        "thresholds": threshold_payload,
        "fpr": fpr_payload,
        "class_rates": rates,
        "average_tpr": averages,
        "scoring": _normal_scoring(scoring),
        "variant": variant,
        "runtime_seconds": _runtime(runtime_seconds),
        "config": config,
        "environment": _environment(environment),
        "config_hash": config_hash(config),
        "git_commit": _commit(commit),
        "started_at": _timestamp(started_at),
        "finished_at": _timestamp(finished_at),
    }


def build_adversarial_result(**kwargs: Any) -> dict[str, Any]:
    """Build the JSON document of one adversarial attack.

    Required keyword arguments:

    ``attack_type`` / ``variant`` / ``score_mode``
        Attack name, variant directory and aggregate score mode.
    ``dataset``
        Sampling block with ``n_available``, ``n_samples``, ``sample_mode``
        and ``seed``.
    ``thresholds``
        Mapping with ``fpr_0.10``, ``attack`` and ``policy``.
    ``scoring``
        Mapping with ``n_masks``, ``score_repeats`` and ``paired_scoring``
        (``score_mode`` defaults to the top-level argument).
    ``counts``
        Mapping with ``n``, ``fooled``, ``evaded``, ``backup``,
        ``cons_as_mal`` and ``cons_as_ben``.
    ``config`` / ``runtime_seconds``
        Run configuration (hashed) and wall clock runtime.

    The victim and normal TPRs, the FPR at the attack threshold and the
    rates are computed from the supplied score arrays.  For convenience the
    rates may instead be passed precomputed as ``fpr_at_attack_threshold``,
    ``normal_tpr_at_attack_threshold`` and ``victim_tpr``.  The two forms are
    mutually exclusive per metric.

    Optional keyword arguments: ``status`` (default ``"completed"``),
    ``started_at``, ``finished_at``, ``environment`` (computed when absent)
    and ``git_commit`` (looked up when absent).
    """
    function = "build_adversarial_result"
    attack_type = str(_take(kwargs, "attack_type", function=function))
    variant = str(_take(kwargs, "variant", DEFAULT_VARIANT, function=function))
    score_mode = str(_take(kwargs, "score_mode", function=function))
    dataset = _as_mapping(_take(kwargs, "dataset", function=function), "dataset")
    thresholds = _as_mapping(
        _take(kwargs, "thresholds", function=function), "thresholds"
    )
    scoring = _as_mapping(_take(kwargs, "scoring", function=function), "scoring")
    counts = _as_mapping(_take(kwargs, "counts", function=function), "counts")
    config = _as_mapping(_take(kwargs, "config", function=function), "config")
    runtime_seconds = _take(kwargs, "runtime_seconds", function=function)

    status = str(_take(kwargs, "status", "completed", function=function))
    started_at = _take(kwargs, "started_at", None, function=function)
    finished_at = _take(kwargs, "finished_at", None, function=function)
    environment = _take(kwargs, "environment", None, function=function)
    commit = _take(kwargs, "git_commit", _MISSING, function=function)

    threshold_payload = _adversarial_thresholds(thresholds)
    attack_threshold = threshold_payload["attack"]
    count_payload = _counts(counts)
    n_samples = count_payload["n"]

    fpr_at_attack = _rate_or_compute(
        kwargs, "fpr_at_attack_threshold", "benign_scores", attack_threshold, None
    )
    normal_tpr = _rate_or_compute(
        kwargs,
        "normal_tpr_at_attack_threshold",
        "normal_scores",
        attack_threshold,
        n_samples,
    )
    victim_tpr = _rate_or_compute(
        kwargs, "victim_tpr", "victim_scores", attack_threshold, n_samples
    )
    _reject_extra(kwargs, function=function)

    if n_samples:
        fooled_rate = count_payload["fooled"] / n_samples
        evaded_rate = count_payload["evaded"] / n_samples
        backup_rate = count_payload["backup"] / n_samples
    else:
        fooled_rate = evaded_rate = backup_rate = 0.0

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "experiment": "adversarial",
        "model_type": MODEL_TYPE,
        "score_mode": score_mode,
        "attack_type": attack_type,
        "variant": variant,
        "dataset": _adversarial_dataset(dataset),
        "thresholds": threshold_payload,
        "scoring": _adversarial_scoring(score_mode, scoring),
        "fpr_at_attack_threshold": float(fpr_at_attack),
        "normal_tpr_at_attack_threshold": float(normal_tpr),
        "victim_tpr": float(victim_tpr),
        "attack_success_rate": 1.0 - float(victim_tpr),
        "conditional_drop": conditional_drop(victim_tpr, normal_tpr),
        "fooled_rate": float(fooled_rate),
        "evaded_rate": float(evaded_rate),
        "backup_rate": float(backup_rate),
        "counts": count_payload,
        "runtime_seconds": _runtime(runtime_seconds),
        "config": config,
        "environment": _environment(environment),
        "config_hash": config_hash(config),
        "git_commit": _commit(commit),
        "started_at": _timestamp(started_at),
        "finished_at": _timestamp(finished_at),
    }
