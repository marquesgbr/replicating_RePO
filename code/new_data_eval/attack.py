"""Adversarial flow attack used by the new dataset evaluation.

The class is a faithful port of ``FlowAttack`` from the Apuana reference
runner (``cluster/repo_runner.py``), which was transcribed from the reference
``Flow_based_RePO_Adversarial.ipynb`` notebook.  The attack optimizes a
modifier ``alpha`` added to the normalized sample; every optimization step
minimizes the sum of the per-mask mean squared reconstruction errors (the
RePO-style loss) under a fresh random mask.

Only three things differ from the reference implementation:

* the masks are **deterministic**: they are derived from
  ``(seed, sample_index, iteration)`` with a stateless NumPy generator, so the
  attack of a sample depends only on its global index and never on the order
  in which samples are processed;
* :meth:`FlowAttack.find_adv` returns an :class:`AttackOutcome` that
  classifies the result as ``evaded``, ``backup``, ``no_change`` or
  ``not_fooled`` (the four counts of the result schema);
* :meth:`FlowAttack.score` takes the global ``sample_index`` and the
  ``phase`` and delegates to :func:`new_data_eval.scoring.score_samples`, so
  the success/victim decisions use the repeat-averaged **RePO+** score (five
  groups of ``n_masks // 5`` masks) instead of a private test step.

Every feature index of the reference notebook is kept verbatim and validated
against ``num_input``: the flow model consumes exactly :data:`N_FEATURES`
(77) features and a mask index outside that range is a programming error, not
a silent truncation.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from new_data_eval.data import N_FEATURES
from new_data_eval.scoring import (
    MASK_PROBABILITY,
    PHASE_ATTACK,
    PHASES,
    score_samples,
)

__all__ = [
    "AttackOutcome",
    "FlowAttack",
    "ITERATIONS",
    "LEARNING_RATES",
]

LOG = logging.getLogger(__name__)

#: Adam learning rates of the four optimization attempts (reference order).
LEARNING_RATES: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)

#: Iteration budget of every learning rate of :data:`LEARNING_RATES`.
ITERATIONS: tuple[int, ...] = (30, 40, 50, 60)

#: Features zeroed by the reference fixed mask (they are *masked*, not kept).
_FIXED_ZEROED_INDICES: tuple[int, ...] = (
    0,
    3,
    4,
    5,
    10,
    11,
    12,
    13,
    25,
    26,
    27,
    28,
    29,
    31,
    33,
    35,
    37,
    53,
    54,
    66,
)

#: Duplicate features copied from their source column.
_DUPLICATE_FEATURES: dict[int, int] = {61: 2, 63: 3, 62: 4, 64: 5}

#: Features that must not decrease (clamped to their original value).
_MINS: tuple[int, ...] = (7, 19, 24, 38)

#: Features that must not increase (clamped to their original value).
_MAXS: tuple[int, ...] = (6, 18, 23, 39)

#: Binary flags that may only increase, and whose increase feeds feature 2.
_FLAGS: tuple[int, ...] = (30, 32, 43, 44, 45, 46, 47, 48, 49, 50)

#: Aggregated features that must stay non negative.
_AGGR_FEATURES: tuple[int, ...] = (
    9,
    17,
    22,
    41,
    42,
    55,
    56,
    57,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
)

#: Indices opened by ``mask_l2`` (on top of the fixed mask).
_MASK_L2_INDICES: tuple[int, ...] = (9, 17, 22, 41, 42)

#: Indices opened by ``mask_l3`` (on top of ``mask_l2``).
_MASK_L3_INDICES: tuple[int, ...] = (55, 56, 57) + tuple(range(68, 77))

#: Remaining zeroed indices of the reference fixed mask.
_FIXED_ZERO_INDICES: tuple[int, ...] = (
    8,
    14,
    15,
    16,
    20,
    21,
    34,
    36,
    38,
    39,
    40,
    51,
    52,
    67,
)

_TF = None


def _tensorflow():
    """Import TensorFlow lazily (the module stays importable without it)."""
    global _TF
    if _TF is None:
        import tensorflow as tensorflow

        _TF = tensorflow
    return _TF


def _derive_seed(*parts: object) -> int:
    """Return a stable 64 bit seed derived from ``parts``.

    BLAKE2b is used instead of :func:`hash` so the attack masks never depend
    on the interpreter hash randomization: a rerun with the same
    ``(seed, sample_index, iteration)`` must produce the same masks.
    """
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


@dataclass(frozen=True)
class AttackOutcome:
    """Result of :meth:`FlowAttack.find_adv` for a single sample.

    ``kind`` is the classification of the sample:

    ``evaded``
        A modified sample scored below the attack threshold.
    ``backup``
        No modified sample reached the attack threshold, but one reached the
        (looser) real threshold; it is returned as a best effort.
    ``no_change``
        The untouched sample already scored below the attack threshold, so no
        modification is needed (``x_adv`` is ``None``).
    ``not_fooled``
        The attack did not find any sample below the real threshold.

    ``score`` is the attack-phase RePO+ score of ``x_adv`` (or of the
    untouched sample for ``no_change``); it is ``None`` for ``not_fooled``.
    """

    kind: Literal["evaded", "backup", "no_change", "not_fooled"]
    x_adv: np.ndarray | None = None
    score: float | None = None


class FlowAttack:
    """Adversarial flow attack with deterministic, per-sample masks."""

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
        seed: int,
        n_masks: int = 100,
        score_repeats: int = 1,
        paired_seed: int | None = None,
    ) -> None:
        num_input = int(num_input)
        if num_input != N_FEATURES:
            raise ValueError(
                f"FlowAttack requires num_input == {N_FEATURES}, got {num_input}"
            )
        n_masks = int(n_masks)
        if n_masks <= 0:
            raise ValueError(f"n_masks must be positive, got {n_masks}")
        if n_masks % 5 != 0:
            raise ValueError(
                f"n_masks must be divisible by 5, got {n_masks}"
            )
        score_repeats = int(score_repeats)
        if score_repeats < 1:
            raise ValueError(
                f"score_repeats must be positive, got {score_repeats}"
            )

        self.model = model
        self.num_input = num_input
        self.n_masks = n_masks
        self.score_repeats = score_repeats
        self.paired_seed = None if paired_seed is None else int(paired_seed)
        self.seed = int(seed)
        self.real_threshold = self._finite_threshold(real_threshold, "real_threshold")
        self.attack_threshold = self._finite_threshold(
            attack_threshold, "attack_threshold"
        )

        self.train_min = self._as_bounds(train_min, "train_min")
        self.train_max = self._as_bounds(train_max, "train_max")

        self.decimal_features = [int(index) for index in decimal_features]
        for index in self.decimal_features:
            if not 0 <= index < self.num_input:
                raise ValueError(
                    f"decimal feature index {index} is outside [0, {self.num_input})"
                )

        self.fixed_mask, self.mask_l1, self.mask_l2, self.mask_l3 = self._build_masks()
        self._validate_masks()

        # Reference notebook attributes used by the constraint projection.
        self.dup_ht = dict(_DUPLICATE_FEATURES)
        self.mins = list(_MINS)
        self.maxs = list(_MAXS)
        self.flags = list(_FLAGS)
        self.aggr_features = list(_AGGR_FEATURES)

        t = _tensorflow()
        self.alpha = t.Variable(
            np.zeros((1, self.num_input), dtype=np.float32), name="modifier"
        )
        self.optimizers = [
            t.keras.optimizers.Adam(learning_rate=float(rate))
            for rate in LEARNING_RATES
        ]
        # Match the reference notebook: initialize the optimizer slots once.
        for optimizer in self.optimizers:
            optimizer.apply_gradients(zip([self.alpha], [self.alpha]))

        self._get_adv = self._make_get_adv(t)
        self._optim = self._make_optim(t)

    # -- validation helpers -------------------------------------------------

    @staticmethod
    def _finite_threshold(value: float, name: str) -> float:
        threshold = float(value)
        if not np.isfinite(threshold):
            raise ValueError(f"{name} must be finite, got {value!r}")
        return threshold

    def _as_bounds(self, values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32).ravel()
        if array.shape != (self.num_input,):
            raise ValueError(
                f"{name} must have shape ({self.num_input},), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        return array

    def _build_masks(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build the reference fixed mask and its three variants."""
        fixed = np.ones(self.num_input, dtype=np.float32)
        for index in _FIXED_ZEROED_INDICES:
            fixed[index] = 0
        for index in range(58, 65):
            fixed[index] = 0
        for index in _AGGR_FEATURES[:8]:
            fixed[index] = 0
        for index in range(68, 77):
            fixed[index] = 0
        for index in _FIXED_ZERO_INDICES:
            fixed[index] = 0

        mask_l1 = np.copy(fixed)
        mask_l2 = np.copy(fixed)
        for index in _MASK_L2_INDICES:
            mask_l2[index] = 1
        mask_l3 = np.copy(mask_l2)
        for index in _MASK_L3_INDICES:
            mask_l3[index] = 1
        return fixed, mask_l1, mask_l2, mask_l3

    def _validate_masks(self) -> None:
        """Reject a source feature index outside ``[0, num_input)``.

        The masks themselves are always ``num_input`` wide, so a
        ``np.flatnonzero(mask)`` bound check is dead code.  The source index
        tuples (zeroed features, duplicates, clamps, flags, aggregates and the
        extra mask layers) are the only place where an out-of-range index
        could silently index the wrong feature, so they are validated here.
        """
        sources: tuple[tuple[str, Sequence[int]], ...] = (
            ("_FIXED_ZEROED_INDICES", _FIXED_ZEROED_INDICES),
            ("_DUPLICATE_FEATURES keys", tuple(_DUPLICATE_FEATURES)),
            ("_DUPLICATE_FEATURES values", tuple(_DUPLICATE_FEATURES.values())),
            ("_MINS", _MINS),
            ("_MAXS", _MAXS),
            ("_FLAGS", _FLAGS),
            ("_AGGR_FEATURES", _AGGR_FEATURES),
            ("_MASK_L2_INDICES", _MASK_L2_INDICES),
            ("_MASK_L3_INDICES", _MASK_L3_INDICES),
            ("_FIXED_ZERO_INDICES", _FIXED_ZERO_INDICES),
        )
        for name, indices in sources:
            for index in indices:
                if not 0 <= int(index) < self.num_input:
                    raise ValueError(
                        f"{name} references feature index {index} outside "
                        f"[0, {self.num_input})"
                    )
        for name, mask in (
            ("fixed_mask", self.fixed_mask),
            ("mask_l1", self.mask_l1),
            ("mask_l2", self.mask_l2),
            ("mask_l3", self.mask_l3),
        ):
            if mask.shape != (self.num_input,):
                raise ValueError(
                    f"{name} must have shape ({self.num_input},), got {mask.shape}"
                )

    # -- deterministic masks ------------------------------------------------

    def _iteration_masks(self, sample_index: int, iteration: int) -> np.ndarray:
        """Return the mask of one optimization step.

        The mask depends only on ``(seed, sample_index, iteration)`` and keeps
        a feature with probability ``1 - MASK_PROBABILITY`` (the reference
        keeps a feature when ``uniform > 0.75``).
        """
        generator = np.random.default_rng(
            _derive_seed("attack-mask", self.seed, int(sample_index), int(iteration))
        )
        uniforms = generator.random((self.n_masks, self.num_input))
        return (uniforms > MASK_PROBABILITY).astype(np.float32)

    # -- TensorFlow graph ---------------------------------------------------

    def _make_get_adv(self, t):
        """Return the compiled ``(normalized + alpha) -> raw`` projection."""

        @t.function
        def get_adv(x, adv_mask):
            span = self.train_max - self.train_min + 1e-6
            x_normalized = (x - self.train_min) / span
            alpha_masked = self.alpha * adv_mask
            adv_ex_normalized = x_normalized + alpha_masked
            return adv_ex_normalized * span + self.train_min

        return get_adv

    def _make_optim(self, t):
        """Return the compiled single optimization step.

        The mask is passed as an argument (instead of being drawn with
        ``tf.random``) so it can be derived deterministically from
        ``(seed, sample_index, iteration)``.  The loss is the RePO-style sum
        of the per-mask mean squared errors.
        """

        @t.function
        def optim(x, adv_mask, def_mask, optimizer):
            with t.GradientTape() as tape:
                span = self.train_max - self.train_min + 1e-6
                x_normalized = (x - self.train_min) / span
                adv_ex_normalized = x_normalized + self.alpha * adv_mask
                partial_x = def_mask * adv_ex_normalized
                rec_x = self.model(partial_x, training=False)
                score = t.reduce_mean(
                    t.square(rec_x - adv_ex_normalized), axis=1
                )
                loss = t.reduce_sum(score)
            gradients = tape.gradient(loss, [self.alpha])
            optimizer.apply_gradients(zip(gradients, [self.alpha]))

        return optim

    # -- public API ---------------------------------------------------------

    def score(self, sample: np.ndarray, sample_index: int, phase: str) -> float:
        """Return the RePO+ score of ``sample`` in ``phase``.

        ``sample`` holds the **raw** features of one sample.  The masks are
        derived by :func:`score_samples` from
        ``(seed, sample_index, phase, repeat)``, so the score is deterministic
        and independent of any batching.  The attack forwards its configured
        ``score_repeats`` and ``paired_seed`` so the success/victim decisions
        use the same repeat-averaged scores as the evaluation.
        """
        if phase not in PHASES:
            raise ValueError(
                f"unknown phase: {phase!r}; expected one of {list(PHASES)}"
            )
        batch = self._as_sample(sample)[None, :]
        scores = score_samples(
            self.model,
            batch,
            self.train_min,
            self.train_max,
            mode="repo_plus",
            n_masks=self.n_masks,
            seed=self.seed,
            phase=phase,
            sample_indices=[int(sample_index)],
            score_repeats=self.score_repeats,
            paired_seed=self.paired_seed,
        )
        return float(scores[0])

    def find_adv(self, sample: np.ndarray, sample_index: int) -> AttackOutcome:
        """Run the attack on one raw sample and classify the outcome."""
        sample_index = int(sample_index)
        if sample_index < 0:
            raise ValueError(f"sample_index must not be negative, got {sample_index}")
        orig_sample = self._as_sample(sample)[None, :]

        self.alpha.assign(np.zeros((1, self.num_input), dtype=np.float32))

        initial_score = self.score(orig_sample, sample_index, PHASE_ATTACK)
        if initial_score < self.attack_threshold:
            return AttackOutcome(kind="no_change", x_adv=None, score=initial_score)

        backup: list[np.ndarray] = []
        backup_score: list[float] = []
        iteration = 0

        def optimize(
            optimizer, total_iter: int, mask_v: np.ndarray
        ) -> AttackOutcome | None:
            nonlocal iteration
            self.alpha.assign(np.zeros((1, self.num_input), dtype=np.float32))
            for _ in range(total_iter):
                def_mask = self._iteration_masks(sample_index, iteration)
                iteration += 1
                self._optim(orig_sample, mask_v, def_mask, optimizer)
                adv_ex = self._get_adv(orig_sample, mask_v)
                adv_ex_np = np.array(adv_ex.numpy(), dtype=np.float32, copy=True)

                for key, source in self.dup_ht.items():
                    adv_ex_np[0, key] = adv_ex_np[0, source]
                adv_ex_np[0, self.mins] = np.maximum(0, adv_ex_np[0, self.mins])
                adv_ex_np[0, self.mins] = np.minimum(
                    orig_sample[0, self.mins], adv_ex_np[0, self.mins]
                )
                adv_ex_np[0, self.maxs] = np.maximum(
                    orig_sample[0, self.maxs], adv_ex_np[0, self.maxs]
                )
                adv_ex_np[0, self.flags] = np.maximum(
                    orig_sample[0, self.flags], adv_ex_np[0, self.flags]
                )
                flags_max_changed = np.max(
                    adv_ex_np[0, self.flags] - orig_sample[0, self.flags]
                )
                adv_ex_np[0, self.aggr_features] = np.maximum(
                    0, adv_ex_np[0, self.aggr_features]
                )
                adv_ex_np[0, 2] = np.maximum(
                    orig_sample[0, 2] + flags_max_changed, adv_ex_np[0, 2]
                )
                adv_ex_np[0, 65] = np.maximum(0, adv_ex_np[0, 65])

                if self.decimal_features:
                    adv_ex_np[0, self.decimal_features] = np.round(
                        adv_ex_np[0, self.decimal_features]
                    )

                adv_ex_np[0, 4] = (
                    adv_ex_np[0, 4]
                    + (adv_ex_np[0, 6] != orig_sample[0, 6]) * adv_ex_np[0, 6]
                    + (adv_ex_np[0, 7] != orig_sample[0, 7]) * adv_ex_np[0, 7]
                )
                adv_ex_np[0, 8] = adv_ex_np[0, 4] / adv_ex_np[0, 2]
                adv_ex_np[0, 14] = (
                    (adv_ex_np[0, 4] + adv_ex_np[0, 5]) / adv_ex_np[0, 1] * 1e6
                )
                adv_ex_np[0, 15] = (
                    (adv_ex_np[0, 2] + adv_ex_np[0, 3]) / adv_ex_np[0, 1] * 1e6
                )
                adv_ex_np[0, 16] = adv_ex_np[0, 1] / (
                    adv_ex_np[0, 2] + adv_ex_np[0, 3] - 1
                )

                adv_ex_np[0, 21] = adv_ex_np[0, 20] / (adv_ex_np[0, 2] - 1)
                adv_ex_np[0, 34] = adv_ex_np[0, 34] + 20 * (
                    adv_ex_np[0, 2] - orig_sample[0, 2]
                )
                adv_ex_np[0, 36] = adv_ex_np[0, 2] / adv_ex_np[0, 1] * 1e6

                adv_ex_np[0, 38] = np.minimum(adv_ex_np[0, 38], adv_ex_np[0, 7])
                adv_ex_np[0, 39] = np.maximum(adv_ex_np[0, 39], adv_ex_np[0, 6])

                adv_ex_np[0, 40] = (adv_ex_np[0, 4] + adv_ex_np[0, 5]) / (
                    adv_ex_np[0, 3] + adv_ex_np[0, 2] + 1
                )
                adv_ex_np[0, 51] = adv_ex_np[0, 3] / adv_ex_np[0, 2]
                adv_ex_np[0, 52] = (adv_ex_np[0, 4] + adv_ex_np[0, 5]) / (
                    adv_ex_np[0, 3] + adv_ex_np[0, 2]
                )
                adv_ex_np[0, 53] = adv_ex_np[0, 8]
                adv_ex_np[0, 67] = (
                    adv_ex_np[0, 67]
                    + (adv_ex_np[0, 6] != orig_sample[0, 6]) * 1
                    + (adv_ex_np[0, 7] != orig_sample[0, 7]) * 1
                )

                adv_ex_np[np.isinf(adv_ex_np)] = 0
                adv_ex_np[np.isnan(adv_ex_np)] = 0
                score = self.score(adv_ex_np, sample_index, PHASE_ATTACK)
                if score < self.attack_threshold:
                    return AttackOutcome(
                        kind="evaded",
                        x_adv=np.array(adv_ex_np[0], dtype=np.float32, copy=True),
                        score=score,
                    )
                if score < self.real_threshold and not backup:
                    backup.append(
                        np.array(adv_ex_np[0], dtype=np.float32, copy=True)
                    )
                    backup_score.append(score)
            return None

        for mask in (self.mask_l1, self.mask_l2, self.mask_l3):
            for optimizer, total_iter in zip(self.optimizers, ITERATIONS):
                outcome = optimize(optimizer, total_iter, mask)
                if outcome is not None:
                    return outcome

        if backup:
            return AttackOutcome(kind="backup", x_adv=backup[0], score=backup_score[0])
        return AttackOutcome(kind="not_fooled", x_adv=None, score=None)

    # -- checkpoint helpers -------------------------------------------------

    def optimizer_state(self) -> dict[str, np.ndarray]:
        """Return the optimizer slots as a namespaced NumPy mapping."""
        state: dict[str, np.ndarray] = {}
        for index, optimizer in enumerate(self.optimizers):
            for weight_index, variable in enumerate(optimizer.variables):
                state[f"flow_opt_{index}_{weight_index}"] = np.asarray(
                    variable.numpy()
                )
        return state

    def load_optimizer_state(self, state: Mapping[str, np.ndarray]) -> None:
        """Restore the optimizer slots saved by :meth:`optimizer_state`."""
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
                    "optimizer %d state has %d tensors, expected %d; ignoring",
                    index,
                    len(weights),
                    len(variables),
                )
                continue
            for variable, value in zip(variables, weights):
                variable.assign(value)

    # -- input helpers ------------------------------------------------------

    def _as_sample(self, sample: np.ndarray) -> np.ndarray:
        """Return ``sample`` as one finite ``(num_input,)`` float32 row."""
        array = np.asarray(sample, dtype=np.float32)
        if array.shape == (1, self.num_input):
            array = array[0]
        if array.shape != (self.num_input,):
            raise ValueError(
                f"sample must have {self.num_input} features, got shape {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("sample must be finite")
        return array
