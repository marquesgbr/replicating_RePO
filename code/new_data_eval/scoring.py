"""RePO/RePO+ scoring primitives, model loading and mask based scoring.

The scoring follows the reference implementation of the Apuana runner
(``cluster/metrics.py``) so the results of the new dataset evaluation use the
very same protocol as the flow/packet replications:

* **RePO** -- the mean of the per-mask reconstruction errors;
* **RePO+** -- the per-mask errors are split into five groups, the minimum of
  every group is taken and the five minima are summed (100 masks -> five
  groups of 20);
* the decision threshold is the ``ceil(n * fpr)``-th largest benign score,
  counted from the end of the sorted benign scores.

The module also implements the model side of the protocol (the "normal"
notebook of the reference): every sample is reconstructed from 100 random
masks -- each feature is masked (zeroed) with probability 0.75 -- and the
reconstruction error of a mask is the mean squared error between the
reconstruction of the masked sample and the **original** normalized sample.

The masks are drawn with a *stateless* RNG seeded from
``(seed, sample_index, phase, repeat)``, so the score of a sample depends
only on its global index: neither ``--batch-size`` nor the chunking of the
CSV file can change a score.

The aggregation stays NumPy-only: TensorFlow is never imported at module
level (the model is injected by the caller), and it is imported lazily by
:func:`load_model` only.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from new_data_eval.data import N_FEATURES, normalize_features

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_N_MASKS",
    "MASK_PROBABILITY",
    "N_FEATURES",
    "PHASES",
    "PHASE_ATTACK",
    "PHASE_ORIGINAL",
    "PHASE_VICTIM",
    "REPO_PLUS_GROUPS",
    "load_model",
    "reconstruction_errors",
    "repo_plus_scores",
    "repo_scores",
    "score_from_errors",
    "score_samples",
    "threshold_at_fpr",
    "validate_model",
]


#: Phase of the score computed on the untouched (original) sample.
PHASE_ORIGINAL = "original"

#: Phase of the score computed during the attack optimization.
PHASE_ATTACK = "attack"

#: Phase of the score computed on the adversarial (victim) sample.
PHASE_VICTIM = "victim"

#: Every phase used to derive the stateless mask seeds.
PHASES: tuple[str, ...] = (PHASE_ORIGINAL, PHASE_ATTACK, PHASE_VICTIM)

#: Number of groups used by the RePO+ aggregate (100 masks -> 5 groups of 20).
REPO_PLUS_GROUPS = 5

#: Probability of masking (zeroing) a feature in every random mask.
MASK_PROBABILITY = 0.75

#: Default number of masks drawn for every sample.
DEFAULT_N_MASKS = 100

#: Default number of model rows (masked samples) per forward pass.
DEFAULT_BATCH_SIZE = 100


def _as_error_matrix(errors: np.ndarray) -> np.ndarray:
    """Return ``errors`` as a finite ``(n_samples, n_masks)`` float64 matrix."""
    array = np.asarray(errors, dtype=np.float64)
    if array.ndim == 1:
        # A single sample: one row of per-mask errors.
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"errors must be 2-D, got shape {array.shape}")
    if array.shape[1] == 0:
        raise ValueError("errors must contain at least one mask")
    if not np.isfinite(array).all():
        raise ValueError("errors must be finite")
    return array


def repo_scores(errors: np.ndarray) -> np.ndarray:
    """Return the RePO score of every sample (the mean of its mask errors).

    ``errors`` has shape ``(n_samples, n_masks)``; the result has shape
    ``(n_samples,)``.
    """
    return _as_error_matrix(errors).mean(axis=1)


def repo_plus_scores(errors: np.ndarray) -> np.ndarray:
    """Return the RePO+ score of every sample.

    The masks are split into :data:`REPO_PLUS_GROUPS` equally sized groups,
    the minimum error of every group is taken and the minima are summed
    (100 masks -> five groups of 20).  The number of masks must be divisible
    by the number of groups, otherwise the split is ambiguous and a
    :class:`ValueError` is raised.
    """
    array = _as_error_matrix(errors)
    n_masks = array.shape[1]
    if n_masks % REPO_PLUS_GROUPS != 0:
        raise ValueError(
            f"number of masks ({n_masks}) is not divisible by groups "
            f"({REPO_PLUS_GROUPS})"
        )
    grouped = array.reshape(array.shape[0], REPO_PLUS_GROUPS, -1)
    return grouped.min(axis=2).sum(axis=1)


def score_from_errors(errors: np.ndarray, mode: str) -> np.ndarray:
    """Return the scores of ``errors`` for the requested ``mode``.

    ``mode="repo"`` returns the RePO scores and ``mode="repo_plus"`` the
    RePO+ scores; every other mode raises a :class:`ValueError`.
    """
    if mode == "repo":
        return repo_scores(errors)
    if mode == "repo_plus":
        return repo_plus_scores(errors)
    raise ValueError(f"unknown score mode: {mode!r}")


def threshold_at_fpr(
    benign_scores: np.ndarray, fpr: float, method: str = "ceil"
) -> float:
    """Return the decision threshold of the benign scores at ``fpr``.

    The threshold is the ``ceil(n * fpr)``-th largest benign score, counted
    from the end of the sorted scores -- the same rule the flow notebook and
    the Apuana runner use.  Only ``method="ceil"`` is supported.  The index is
    clamped to ``[1, n]``, so the threshold always exists and is one of the
    benign scores.  Non-finite benign scores are rejected: they would
    otherwise silently poison the threshold.
    """
    values = np.asarray(benign_scores, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("cannot compute a threshold from an empty score array")
    if not 0.0 < fpr < 1.0:
        raise ValueError(f"fpr must be in (0, 1), got {fpr!r}")
    if not np.isfinite(values).all():
        raise ValueError("benign scores must be finite")
    if method != "ceil":
        raise ValueError(f"unknown threshold method: {method!r}")

    index = int(math.ceil(values.size * fpr))
    index = max(1, min(index, values.size))
    return float(np.sort(values)[-index])


# ---------------------------------------------------------------------------
# Model loading and mask based scoring
# ---------------------------------------------------------------------------


def load_model(model_dir: Path) -> Any:
    """Load the trained model of the evaluation.

    ``model_dir`` may be a SavedModel directory (the format produced by
    ``model.export()`` and used by ``models/new_data_model``) or a Keras
    file.  The loading is delegated to ``code/load_model_compat.py``, which
    handles both the legacy SavedModel and the Keras 3 formats.

    TensorFlow is imported lazily (inside the function) so that this module
    stays importable without a deep learning stack.
    """
    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    try:
        from load_model_compat import load_model as _load_compat_model
    except ImportError as exc:  # pragma: no cover - depends on sys.path
        raise ImportError(
            "load_model_compat could not be imported; the 'code' directory of "
            "the repository must be on sys.path (PYTHONPATH=code)"
        ) from exc

    return _load_compat_model(str(path))


def _derive_seed(*parts: object) -> int:
    """Return a stable 64 bit seed derived from ``parts``.

    The derivation uses BLAKE2b instead of :func:`hash` so the seeds do not
    depend on the interpreter hash randomization -- the same seed must always
    produce the same masks.
    """
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _mask_seed(
    seed: int,
    sample_index: int,
    phase: str,
    repeat: int,
    paired_seed: int | None,
) -> int:
    """Return the seed of the masks of one ``(sample, phase, repeat)`` triple.

    Without ``paired_seed`` the phase takes part in the derivation, so every
    phase uses its own masks.  With ``paired_seed`` the phase is dropped and
    the seed of the paired run is used instead: the before/after phases of
    the paired sensitivity analysis then share the very same masks.
    """
    if paired_seed is None:
        return _derive_seed("mask", seed, sample_index, phase, repeat)
    return _derive_seed("mask-paired", paired_seed, sample_index, repeat)


def _sample_masks(
    seed: int,
    sample_index: int,
    phase: str,
    repeat: int,
    n_masks: int,
    paired_seed: int | None,
) -> np.ndarray:
    """Return the ``(n_masks, N_FEATURES)`` masks of one sample.

    A mask keeps a feature with probability ``1 - MASK_PROBABILITY`` (the
    reference notebooks keep a feature when ``uniform > 0.75``) and zeroes it
    otherwise.
    """
    generator = np.random.default_rng(
        _mask_seed(seed, sample_index, phase, repeat, paired_seed)
    )
    uniforms = generator.random((n_masks, N_FEATURES))
    return (uniforms > MASK_PROBABILITY).astype(np.float32)


def _accepts_training(model: Any) -> bool:
    """Return whether ``model`` accepts a ``training`` keyword argument."""
    try:
        parameters = inspect.signature(model).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return True
    if "training" in parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _call_model(model: Any, inputs: np.ndarray, supports_training: bool) -> Any:
    """Call ``model`` with ``inputs`` and return the raw output."""
    if supports_training:
        return model(inputs, training=False)
    return model(inputs)


def _predict(
    model: Any, inputs: np.ndarray, supports_training: bool
) -> np.ndarray:
    """Call ``model`` and return the reconstruction as a finite float32 array."""
    outputs = _call_model(model, inputs, supports_training)
    array = np.asarray(outputs, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != N_FEATURES:
        raise ValueError(
            f"The model must return {N_FEATURES} features per row; "
            f"got shape {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("The model produced non-finite outputs")
    return array


def _declared_input_width(model: Any) -> int | None:
    """Return the declared input width of ``model`` (``None`` if unknown)."""
    shape = getattr(model, "input_shape", None)
    if shape is None:
        return None
    if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
        entries = list(shape)
    else:
        entries = [shape]
    widths = set()
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or not entry:
            return None
        widths.add(int(entry[-1]))
    if len(widths) != 1:
        return None
    return widths.pop()


def _check_dynamic_batch(model: Any, supports_training: bool) -> None:
    """Check that ``model`` accepts a batch of any size (``None`` batch axis)."""
    for rows in (1, 2):
        probe = np.zeros((rows, N_FEATURES), dtype=np.float32)
        try:
            outputs = _call_model(model, probe, supports_training)
        except Exception as exc:
            raise ValueError(
                f"The model must accept a (None, {N_FEATURES}) input; the call "
                f"with {rows} row(s) failed: {exc}"
            ) from exc
        array = np.asarray(outputs, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != N_FEATURES:
            raise ValueError(
                f"The model must return {N_FEATURES} features per row; "
                f"got shape {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("The model produced non-finite outputs")


def _resolve_sample_indices(
    sample_indices: Sequence[int] | None, n_samples: int
) -> np.ndarray:
    """Return the global index of every sample of the current block."""
    if sample_indices is None:
        return np.arange(n_samples, dtype=np.int64)
    indices = np.asarray(list(sample_indices), dtype=np.int64)
    if indices.ndim != 1 or indices.shape[0] != n_samples:
        raise ValueError(
            f"sample_indices must hold one index per sample ({n_samples}); "
            f"got {indices.shape}"
        )
    if (indices < 0).any():
        raise ValueError("sample_indices must not be negative")
    return indices


def _prepare_scoring(
    x_raw: np.ndarray,
    train_min: np.ndarray,
    train_max: np.ndarray,
    *,
    n_masks: int,
    batch_size: int,
    score_repeats: int,
    phase: str,
    sample_indices: Sequence[int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the arguments and return the normalized samples and indices."""
    x = np.asarray(x_raw, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(
            f"Expected a 2-D feature matrix with {N_FEATURES} columns, "
            f"got shape {x.shape}"
        )
    if x.shape[1] != N_FEATURES:
        raise ValueError(
            f"Expected a ({x.shape[0]}, {N_FEATURES}) feature matrix, "
            f"got shape {x.shape}"
        )
    if n_masks <= 0:
        raise ValueError(f"n_masks must be positive, got {n_masks}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if score_repeats <= 0:
        raise ValueError(f"score_repeats must be positive, got {score_repeats}")
    if phase not in PHASES:
        raise ValueError(
            f"unknown phase: {phase!r}; expected one of {list(PHASES)}"
        )

    indices = _resolve_sample_indices(sample_indices, x.shape[0])
    x_norm = normalize_features(x, train_min, train_max)
    return x_norm, indices


def _store_errors(
    errors: np.ndarray,
    inputs: np.ndarray,
    targets: np.ndarray,
    slots: np.ndarray,
    filled: int,
    model: Any,
    supports_training: bool,
) -> None:
    """Reconstruct one batch of masked samples and store the mask errors."""
    reconstructions = _predict(model, inputs[:filled], supports_training)
    residuals = reconstructions.astype(np.float64) - targets[:filled].astype(np.float64)
    mse = np.mean(np.square(residuals), axis=1)
    errors[slots[:filled, 0], slots[:filled, 1]] = mse


def _reconstruction_errors_normalized(
    model: Any,
    x_norm: np.ndarray,
    indices: np.ndarray,
    *,
    n_masks: int,
    batch_size: int,
    seed: int,
    phase: str,
    score_repeats: int,
    paired_seed: int | None,
) -> np.ndarray:
    """Return the per-mask errors of every sample and repeat.

    The result has shape ``(score_repeats, n_samples, n_masks)``.  The masks
    are drawn per sample (never shared by a batch) and the reconstruction
    error of a mask is the mean squared error between the reconstruction of
    the masked sample and the original normalized sample.
    """
    n_samples = x_norm.shape[0]
    errors = np.empty((score_repeats, n_samples, n_masks), dtype=np.float64)

    inputs = np.empty((batch_size, N_FEATURES), dtype=np.float32)
    targets = np.empty((batch_size, N_FEATURES), dtype=np.float32)
    slots = np.empty((batch_size, 2), dtype=np.int64)
    mask_indices = np.arange(n_masks, dtype=np.int64)
    supports_training = _accepts_training(model)

    for repeat in range(score_repeats):
        filled = 0
        for row in range(n_samples):
            target = x_norm[row]
            masks = _sample_masks(
                seed, int(indices[row]), phase, repeat, n_masks, paired_seed
            )
            block = target * masks
            start = 0
            while start < n_masks:
                take = min(batch_size - filled, n_masks - start)
                inputs[filled : filled + take] = block[start : start + take]
                targets[filled : filled + take] = target
                slots[filled : filled + take, 0] = row
                slots[filled : filled + take, 1] = mask_indices[start : start + take]
                filled += take
                start += take
                if filled == batch_size:
                    _store_errors(
                        errors[repeat],
                        inputs,
                        targets,
                        slots,
                        filled,
                        model,
                        supports_training,
                    )
                    filled = 0
        if filled:
            _store_errors(
                errors[repeat],
                inputs,
                targets,
                slots,
                filled,
                model,
                supports_training,
            )

    return errors


def reconstruction_errors(
    model: Any,
    x_raw: np.ndarray,
    train_min: np.ndarray,
    train_max: np.ndarray,
    *,
    n_masks: int = DEFAULT_N_MASKS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 2019,
    phase: str = PHASE_ORIGINAL,
    sample_indices: Sequence[int] | None = None,
    score_repeats: int = 1,
    paired_seed: int | None = None,
) -> np.ndarray:
    """Return the reconstruction error of every sample and mask.

    ``x_raw`` holds the **raw** (not normalized) features; the samples are
    normalized with ``train_min``/``train_max`` before they are fed to the
    model.  The result has shape ``(n_samples, n_masks)`` -- the mean squared
    error between the reconstruction of the masked sample and the original
    normalized sample -- and is the mean over the ``score_repeats`` repeats.

    ``sample_indices`` holds the **global** index of every sample of the
    block: the masks are derived from it, so a sample scored alone and a
    sample scored inside a chunk get the same masks.  ``batch_size`` only
    controls how many masked rows are sent to the model at once -- it never
    changes a score.

    ``paired_seed`` replaces the seed of the masks and drops the phase from
    their derivation, so the before/after phases of a paired sensitivity
    analysis share the same masks.
    """
    x_norm, indices = _prepare_scoring(
        x_raw,
        train_min,
        train_max,
        n_masks=n_masks,
        batch_size=batch_size,
        score_repeats=score_repeats,
        phase=phase,
        sample_indices=sample_indices,
    )
    errors = _reconstruction_errors_normalized(
        model,
        x_norm,
        indices,
        n_masks=n_masks,
        batch_size=batch_size,
        seed=seed,
        phase=phase,
        score_repeats=score_repeats,
        paired_seed=paired_seed,
    )
    return errors.mean(axis=0)


def score_samples(
    model: Any,
    x_raw: np.ndarray,
    train_min: np.ndarray,
    train_max: np.ndarray,
    *,
    mode: str,
    n_masks: int = DEFAULT_N_MASKS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 2019,
    phase: str = PHASE_ORIGINAL,
    sample_indices: Sequence[int] | None = None,
    score_repeats: int = 1,
    paired_seed: int | None = None,
) -> np.ndarray:
    """Return the RePO (``mode="repo"``) or RePO+ (``mode="repo_plus"``) scores.

    The scores have shape ``(n_samples,)``.  When ``score_repeats`` is
    greater than one, every repeat is scored with its own (independent) masks
    and the resulting scores are averaged -- the noise analysis of the
    methodology.
    """
    if mode not in ("repo", "repo_plus"):
        raise ValueError(f"unknown score mode: {mode!r}")

    x_norm, indices = _prepare_scoring(
        x_raw,
        train_min,
        train_max,
        n_masks=n_masks,
        batch_size=batch_size,
        score_repeats=score_repeats,
        phase=phase,
        sample_indices=sample_indices,
    )
    errors = _reconstruction_errors_normalized(
        model,
        x_norm,
        indices,
        n_masks=n_masks,
        batch_size=batch_size,
        seed=seed,
        phase=phase,
        score_repeats=score_repeats,
        paired_seed=paired_seed,
    )
    scores = np.stack(
        [score_from_errors(errors[repeat], mode) for repeat in range(errors.shape[0])]
    )
    return scores.mean(axis=0)


def validate_model(
    model: Any,
    train_min: np.ndarray,
    train_max: np.ndarray,
    benign_sample: np.ndarray,
    seed: int = 2019,
) -> dict[str, Any]:
    """Check that the model is usable by the evaluation and report its errors.

    The checks are the minimal validation of the methodology:

    * the model accepts a ``(None, 77)`` input (dynamic batch dimension);
    * the outputs are finite and hold one value per feature;
    * the reconstruction error of a benign sample is lower than the
      reconstruction error of uniform random noise.

    ``benign_sample`` holds the **raw** features of the benign rows used by
    the check.  A :class:`ValueError` is raised when one of the checks fails.
    """
    sample = np.asarray(benign_sample, dtype=np.float32)
    if sample.ndim == 1:
        sample = sample[None, :]
    if sample.ndim != 2 or sample.shape[1] != N_FEATURES:
        raise ValueError(
            f"The benign sample must have {N_FEATURES} features per row; "
            f"got shape {sample.shape}"
        )
    if sample.shape[0] == 0:
        raise ValueError("The benign sample must contain at least one row")

    declared_width = _declared_input_width(model)
    if declared_width is not None and declared_width != N_FEATURES:
        raise ValueError(
            f"The model must accept {N_FEATURES} features; the declared "
            f"input width is {declared_width}"
        )
    _check_dynamic_batch(model, _accepts_training(model))

    indices = np.arange(sample.shape[0], dtype=np.int64)
    x_norm = normalize_features(sample, train_min, train_max)
    benign = _reconstruction_errors_normalized(
        model,
        x_norm,
        indices,
        n_masks=DEFAULT_N_MASKS,
        batch_size=DEFAULT_BATCH_SIZE,
        seed=seed,
        phase=PHASE_ORIGINAL,
        score_repeats=1,
        paired_seed=None,
    )
    benign_error = float(np.mean(benign[0]))

    # The baseline is uniform random noise in the normalized space; it uses
    # the very same masks as the benign samples (same indices and seeds), so
    # the comparison is paired.
    generator = np.random.default_rng(seed)
    noise = generator.random((sample.shape[0], N_FEATURES)).astype(np.float32)
    noise_errors = _reconstruction_errors_normalized(
        model,
        noise,
        indices,
        n_masks=DEFAULT_N_MASKS,
        batch_size=DEFAULT_BATCH_SIZE,
        seed=seed,
        phase=PHASE_ORIGINAL,
        score_repeats=1,
        paired_seed=None,
    )
    noise_error = float(np.mean(noise_errors[0]))

    if not benign_error < noise_error:
        raise ValueError(
            "The reconstruction error of the benign sample is not lower than "
            f"the error of uniform random noise (benign={benign_error!r}, "
            f"noise={noise_error!r})"
        )

    return {
        "input_width": N_FEATURES,
        "declared_input_width": declared_width,
        "n_samples": int(sample.shape[0]),
        "n_masks": DEFAULT_N_MASKS,
        "benign_error": benign_error,
        "noise_error": noise_error,
        "seed": int(seed),
    }
