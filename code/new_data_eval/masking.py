"""Random mask policies used by the RePO/RePO+ protocol.

The reference implementation draws, for every sample and every mask, an
independent Bernoulli(0.25) keep decision per feature: a feature is kept when
``uniform > 0.75`` and zeroed otherwise.  The masks are *stateless*: the seed of
one ``(sample, phase, repeat)`` triple is derived from the global seed with
BLAKE2b, so neither the batch size nor the CSV chunking can change a score.

This module centralises that logic behind :class:`MaskPolicy` so the research
variants (balanced masks, covering designs, fixed deployments) can be swapped
without touching the scoring code.  :class:`IIDMaskPolicy` reproduces the
reference behaviour bit for bit.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "BalancedMaskPolicy",
    "CoveringMaskPolicy",
    "DEFAULT_N_MASKS",
    "IIDMaskPolicy",
    "MASK_PROBABILITY",
    "MaskPolicy",
    "PHASES",
    "PHASE_ATTACK",
    "PHASE_ORIGINAL",
    "PHASE_VICTIM",
    "POLICIES",
    "REPO_PLUS_GROUPS",
    "derive_seed",
    "mask_seed",
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


def derive_seed(*parts: object) -> int:
    """Return a stable 64 bit seed derived from ``parts``.

    The derivation uses BLAKE2b instead of :func:`hash` so the seeds do not
    depend on the interpreter hash randomization -- the same seed must always
    produce the same masks.
    """
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def mask_seed(
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
        return derive_seed("mask", seed, sample_index, phase, repeat)
    return derive_seed("mask-paired", paired_seed, sample_index, repeat)


@runtime_checkable
class MaskPolicy(Protocol):
    """Strategy that generates the masks of one sample.

    Implementations must be deterministic given the arguments and must return
    a ``(n_masks, n_features)`` float32 array of zeros and ones.
    """

    name: str

    def masks(
        self,
        *,
        seed: int,
        sample_index: int,
        phase: str,
        repeat: int,
        n_masks: int,
        n_features: int,
        paired_seed: int | None = None,
    ) -> np.ndarray:
        """Return the masks of one ``(sample, phase, repeat)`` triple."""
        ...


class IIDMaskPolicy:
    """Reference policy: every feature is zeroed independently with p=0.75."""

    name = "iid"

    def masks(
        self,
        *,
        seed: int,
        sample_index: int,
        phase: str,
        repeat: int,
        n_masks: int,
        n_features: int,
        paired_seed: int | None = None,
    ) -> np.ndarray:
        generator = np.random.default_rng(
            mask_seed(seed, sample_index, phase, repeat, paired_seed)
        )
        uniforms = generator.random((n_masks, n_features))
        return (uniforms > MASK_PROBABILITY).astype(np.float32)


def _balanced_rows(
    generator: np.random.Generator, n_masks: int, n_features: int, keep: int
) -> np.ndarray:
    """Return binary masks that keep exactly ``keep`` features per row."""
    if keep <= 0:
        return np.zeros((n_masks, n_features), dtype=np.float32)
    if keep >= n_features:
        return np.ones((n_masks, n_features), dtype=np.float32)
    uniforms = generator.random((n_masks, n_features))
    selected = np.argpartition(uniforms, n_features - keep, axis=1)[
        :, n_features - keep :
    ]
    masks = np.zeros((n_masks, n_features), dtype=np.float32)
    np.put_along_axis(masks, selected, 1.0, axis=1)
    return masks


class BalancedMaskPolicy:
    """Keep exactly 25 % of the features per mask (same marginal as the IID).

    The reference masks keep a Binomial(77, 0.25) number of features, so the
    amount of information available to the autoencoder varies by roughly 20 %
    from mask to mask.  Fixing the count removes that source of variance at no
    extra cost.  The feature subset is still uniformly random and the masks are
    still stateless.
    """

    name = "balanced"

    def masks(
        self,
        *,
        seed: int,
        sample_index: int,
        phase: str,
        repeat: int,
        n_masks: int,
        n_features: int,
        paired_seed: int | None = None,
    ) -> np.ndarray:
        generator = np.random.default_rng(
            mask_seed(seed, sample_index, phase, repeat, paired_seed)
        )
        keep = max(1, int(round((1.0 - MASK_PROBABILITY) * n_features)))
        return _balanced_rows(generator, n_masks, n_features, keep)


class CoveringMaskPolicy:
    """Balanced masks that cover every feature in every RePO+ group.

    With 20 IID masks, a feature is absent from the whole group with
    probability ``0.75**20 ~ 0.3 %``; with 5 masks it rises to 24 %.  This
    policy repairs the group after drawing balanced masks: every feature that
    no mask of the group keeps replaces a random kept feature of a random mask
    of the group.  Each mask still keeps exactly 25 % of the features.
    """

    name = "covering"

    def masks(
        self,
        *,
        seed: int,
        sample_index: int,
        phase: str,
        repeat: int,
        n_masks: int,
        n_features: int,
        paired_seed: int | None = None,
    ) -> np.ndarray:
        generator = np.random.default_rng(
            mask_seed(seed, sample_index, phase, repeat, paired_seed)
        )
        keep = max(1, int(round((1.0 - MASK_PROBABILITY) * n_features)))
        masks = _balanced_rows(generator, n_masks, n_features, keep)
        group_size = max(1, n_masks // REPO_PLUS_GROUPS)
        if group_size * keep < n_features:
            # Covering every feature is impossible when the group does not have
            # at least one slot per feature (e.g. 5 or 10 masks).  Degrade to
            # the balanced policy instead of looping forever trying to repair
            # an impossible cover.
            return masks
        for start in range(0, n_masks, group_size):
            stop = min(start + group_size, n_masks)
            group = masks[start:stop]
            for _ in range(10):
                counts = group.sum(axis=0)
                missing = np.flatnonzero(counts == 0)
                if missing.size == 0:
                    break
                for feature in missing:
                    rows = np.flatnonzero(group[:, feature] == 0)
                    if rows.size == 0:
                        continue
                    row = int(rows[int(generator.integers(rows.size))])
                    # Only drop a feature that another mask of the group keeps,
                    # so the repair never opens a new hole.
                    kept = np.flatnonzero((group[row] > 0) & (counts > 1))
                    if kept.size == 0:
                        continue
                    drop = int(kept[int(generator.integers(kept.size))])
                    group[row, drop] = 0.0
                    group[row, feature] = 1.0
                    counts[drop] -= 1
                    counts[feature] += 1
        return masks


#: Registry used by the experiment harness (name -> policy instance).
POLICIES: dict[str, MaskPolicy] = {
    "iid": IIDMaskPolicy(),
    "balanced": BalancedMaskPolicy(),
    "covering": CoveringMaskPolicy(),
}

