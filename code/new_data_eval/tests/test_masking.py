"""Tests for the mask policies of the RePO protocol.

The reference ``IIDMaskPolicy`` must reproduce the masks of the previous
implementation bit for bit.  The golden values below were captured from that
implementation before the policy abstraction was introduced, so a regression
in the seed derivation or in the keep rule is caught even if the code is
refactored again.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from new_data_eval.masking import (
    DEFAULT_N_MASKS,
    IIDMaskPolicy,
    MASK_PROBABILITY,
    PHASES,
    POLICIES,
    BalancedMaskPolicy,
    CoveringMaskPolicy,
    derive_seed,
    mask_seed,
)
from new_data_eval.scoring import _sample_masks


#: ``(seed, sample_index, phase, repeat, paired_seed)`` -> legacy fingerprint.
GOLDEN = {
    (2019, 0, "original", 0, None): (
        1959,
        "98536005d756b7e3096a668df54e6d1084e8c7cb407dd9326acf386fde4d0741",
    ),
    (2019, 7, "original", 0, None): (
        1946,
        "7c2ff52d5dec8ab7b1d51c08ebb536fd43e90fa4315e9299299e0e9c151cc4b3",
    ),
    (2019, 7, "attack", 3, None): (
        1939,
        "ccc9bcf9ff881813e6e1d26b74f70db32ddb0f518d10835296a965e051281a34",
    ),
    (42, 123, "victim", 1, None): (
        1971,
        "e72fec27e91180126540bf03482b6e71b3f4d37301cfa601a303a7277d9b423b",
    ),
}


def _fingerprint(masks: np.ndarray) -> tuple[int, str]:
    digest = hashlib.sha256(
        np.ascontiguousarray(masks, dtype=np.float32).tobytes()
    ).hexdigest()
    return int(masks.sum()), digest


@pytest.mark.parametrize("case", sorted(GOLDEN))
def test_iid_policy_reproduces_the_legacy_masks(case):
    seed, sample_index, phase, repeat, paired_seed = case
    expected = GOLDEN[case]

    policy = IIDMaskPolicy().masks(
        seed=seed,
        sample_index=sample_index,
        phase=phase,
        repeat=repeat,
        n_masks=DEFAULT_N_MASKS,
        n_features=77,
        paired_seed=paired_seed,
    )
    assert policy.shape == (DEFAULT_N_MASKS, 77)
    assert policy.dtype == np.float32
    assert _fingerprint(policy) == expected

    # The scoring helper (used by the evaluation code) delegates to the same
    # policy and must stay bit-identical.
    helper = _sample_masks(seed, sample_index, phase, repeat, DEFAULT_N_MASKS, paired_seed)
    np.testing.assert_array_equal(helper, policy)


def test_masks_are_deterministic_and_depend_on_every_key():
    policy = IIDMaskPolicy()
    kwargs = dict(
        seed=2019,
        sample_index=3,
        phase="original",
        repeat=0,
        n_masks=10,
        n_features=77,
    )
    first = policy.masks(**kwargs)
    second = policy.masks(**kwargs)
    np.testing.assert_array_equal(first, second)

    for key, value in (
        ("seed", 2020),
        ("sample_index", 4),
        ("phase", "victim"),
        ("repeat", 1),
    ):
        changed = policy.masks(**{**kwargs, key: value})
        assert not np.array_equal(first, changed), key


def test_masks_are_binary_and_keep_roughly_one_quarter():
    masks = IIDMaskPolicy().masks(
        seed=2019,
        sample_index=0,
        phase="original",
        repeat=0,
        n_masks=1000,
        n_features=77,
    )
    assert set(np.unique(masks)) == {0.0, 1.0}
    kept = masks.mean()
    assert abs(kept - (1.0 - MASK_PROBABILITY)) < 0.02


def test_paired_seed_drops_the_phase_and_is_reproducible():
    policy = IIDMaskPolicy()
    kwargs = dict(
        seed=2019,
        sample_index=11,
        repeat=2,
        n_masks=25,
        n_features=77,
        paired_seed=7,
    )
    original = policy.masks(phase="original", **kwargs)
    victim = policy.masks(phase="victim", **kwargs)
    np.testing.assert_array_equal(original, victim)

    # The paired derivation is stable and independent of the plain one.
    assert mask_seed(2019, 11, "original", 2, 7) == mask_seed(
        2019, 11, "victim", 2, 7
    )
    plain = policy.masks(phase="original", **{**kwargs, "paired_seed": None})
    assert not np.array_equal(original, plain)


def test_phases_and_derivation_are_exported():
    assert PHASES == ("original", "attack", "victim")
    assert derive_seed("a", 1) == derive_seed("a", 1)
    assert derive_seed("a", 1) != derive_seed("a", 2)


def test_balanced_policy_keeps_exactly_one_quarter_per_row():
    masks = BalancedMaskPolicy().masks(
        seed=2019,
        sample_index=4,
        phase="original",
        repeat=0,
        n_masks=50,
        n_features=77,
    )
    assert masks.shape == (50, 77)
    assert set(np.unique(masks)) == {0.0, 1.0}
    kept = masks.sum(axis=1)
    assert np.all(kept == round((1.0 - MASK_PROBABILITY) * 77))
    # Still stateless.
    again = BalancedMaskPolicy().masks(
        seed=2019, sample_index=4, phase="original", repeat=0,
        n_masks=50, n_features=77,
    )
    np.testing.assert_array_equal(masks, again)


def test_covering_policy_covers_every_feature_in_every_group():
    n_masks = 25
    masks = CoveringMaskPolicy().masks(
        seed=2020,
        sample_index=9,
        phase="original",
        repeat=1,
        n_masks=n_masks,
        n_features=77,
    )
    assert np.all(masks.sum(axis=1) == round((1.0 - MASK_PROBABILITY) * 77))
    for start in range(0, n_masks, n_masks // 5):
        group = masks[start : start + n_masks // 5]
        assert np.all(group.sum(axis=0) > 0)


def test_covering_policy_falls_back_when_coverage_is_impossible():
    kwargs = dict(
        seed=2020, sample_index=5, phase="original", repeat=0,
        n_features=77,
    )
    covering = CoveringMaskPolicy().masks(n_masks=10, **kwargs)
    balanced = BalancedMaskPolicy().masks(n_masks=10, **kwargs)
    # 10 masks -> groups of 2 -> 38 slots < 77 features: coverage is
    # impossible, so the policy degrades to the balanced masks.
    np.testing.assert_array_equal(covering, balanced)


def test_policy_registry_exposes_the_three_policies():
    assert set(POLICIES) == {"iid", "balanced", "covering"}
    assert POLICIES["iid"].name == "iid"
    assert POLICIES["balanced"].name == "balanced"
    assert POLICIES["covering"].name == "covering"

