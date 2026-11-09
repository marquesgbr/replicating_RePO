"""Tests for the adversarial flow attack port.

The attack is the one transcribed from the reference
``Flow_based_RePO_Adversarial.ipynb`` notebook (and the Apuana runner): a
modifier variable is optimized against the reconstruction error of the model
under the same 100 random masks used to score a sample.  The port changes only
the mask generation (deterministic, per sample), the return type
(:class:`~new_data_eval.attack.AttackOutcome`) and the score used for the
success/victim decisions (the RePO+ score of ``score_samples``).
"""

from __future__ import annotations

import numpy as np
import pytest

from new_data_eval.attack import AttackOutcome, FlowAttack
from new_data_eval.scoring import PHASE_ATTACK, PHASE_VICTIM


def _attack(tiny_model, meta, **overrides):
    """Build an attack with the defaults of the brief."""
    options = {
        "decimal_features": [],
        "num_input": 77,
        "real_threshold": 0.1,
        "attack_threshold": 0.1,
        "seed": 2019,
    }
    options.update(overrides)
    return FlowAttack(tiny_model, *meta, **options)


def test_masks_have_expected_size_and_zeroed_features(tiny_model, meta):
    attack = _attack(tiny_model, meta)

    assert attack.mask_l1.shape == (77,)
    assert attack.mask_l1[0] == 0
    assert attack.mask_l3[76] == 1


def test_masks_follow_the_reference_mapping(tiny_model, meta):
    attack = _attack(tiny_model, meta)

    # mask_l1 only exposes the "fixed" features of the reference notebook.
    assert attack.mask_l1[1] == 1
    assert attack.mask_l1[7] == 1
    assert attack.mask_l1[9] == 0
    assert attack.mask_l1[55] == 0
    assert attack.mask_l1[76] == 0

    # mask_l2 opens the per-packet aggregates 9, 17, 22, 41 and 42.
    for index in (9, 17, 22, 41, 42):
        assert attack.mask_l2[index] == 1
        assert attack.mask_l1[index] == 0
    assert attack.mask_l2[55] == 0

    # mask_l3 additionally opens 55-57 and 68-76.
    for index in (55, 56, 57):
        assert attack.mask_l3[index] == 1
    for index in range(68, 77):
        assert attack.mask_l3[index] == 1


def test_find_adv_returns_a_known_outcome(tiny_model, meta):
    attack = _attack(tiny_model, meta)
    outcome = attack.find_adv(np.zeros(77, dtype=np.float32), sample_index=0)

    assert isinstance(outcome, AttackOutcome)
    assert outcome.kind in {"evaded", "backup", "no_change", "not_fooled"}
    if outcome.x_adv is not None:
        assert outcome.x_adv.shape == (77,)
        assert np.isfinite(outcome.x_adv).all()


def test_find_adv_returns_no_change_below_the_attack_threshold(tiny_model, meta):
    # The identity model reconstructs a zero sample perfectly, so no modifier
    # is needed and the reference returns before any optimization step.
    attack = _attack(tiny_model, meta, attack_threshold=0.1)
    outcome = attack.find_adv(np.zeros(77, dtype=np.float32), sample_index=3)

    assert outcome.kind == "no_change"
    assert outcome.x_adv is None
    assert outcome.score is not None
    assert outcome.score < 0.1


def test_find_adv_returns_backup_when_only_the_real_threshold_is_met(
    tiny_model, meta
):
    # A zero sample scores exactly zero: it cannot strictly beat an attack
    # threshold of zero, but it does beat the (looser) real threshold.  This
    # exercises the whole optimization loop and the backup classification.
    attack = _attack(tiny_model, meta, real_threshold=0.1, attack_threshold=0.0)
    outcome = attack.find_adv(np.zeros(77, dtype=np.float32), sample_index=2)

    assert outcome.kind == "backup"
    assert outcome.x_adv is not None
    assert outcome.x_adv.shape == (77,)
    assert np.isfinite(outcome.x_adv).all()
    assert outcome.score == 0.0


def test_find_adv_returns_not_fooled_when_no_threshold_is_met(tiny_model, meta):
    # A zero sample scores exactly zero, which is not strictly below a zero
    # threshold; the attack therefore reports that the model was not fooled.
    attack = _attack(tiny_model, meta, real_threshold=0.0, attack_threshold=0.0)
    outcome = attack.find_adv(np.zeros(77, dtype=np.float32), sample_index=4)

    assert outcome.kind == "not_fooled"
    assert outcome.x_adv is None
    assert outcome.score is None


def test_score_repeats_must_be_positive(tiny_model, meta):
    with pytest.raises(ValueError, match="score_repeats"):
        _attack(tiny_model, meta, score_repeats=0)


def test_score_forwards_the_scoring_options(monkeypatch, tiny_model, meta):
    import new_data_eval.attack as attack_module

    calls: list[dict[str, object]] = []
    real_score_samples = attack_module.score_samples

    def spy(*args, **kwargs):
        calls.append(dict(kwargs))
        return real_score_samples(*args, **kwargs)

    monkeypatch.setattr(attack_module, "score_samples", spy)
    attack = _attack(
        tiny_model, meta, score_repeats=2, paired_seed=7
    )
    attack.score(np.zeros(77, dtype=np.float32), sample_index=3, phase=PHASE_ATTACK)

    assert calls
    assert all(call["score_repeats"] == 2 for call in calls)
    assert all(call["paired_seed"] == 7 for call in calls)


def test_mask_source_indices_are_validated(monkeypatch, tiny_model, meta):
    import new_data_eval.attack as attack_module

    monkeypatch.setattr(attack_module, "_MINS", (999,))
    with pytest.raises(ValueError, match="999"):
        _attack(tiny_model, meta)


def test_optimizer_state_round_trip(tiny_model, meta):
    attack = _attack(tiny_model, meta)
    state = attack.optimizer_state()

    assert state
    assert all(name.startswith("flow_opt_") for name in state)

    attack.load_optimizer_state(state)


def test_score_is_deterministic_per_sample_index_and_phase(tiny_model, meta):
    attack = _attack(tiny_model, meta)
    sample = np.linspace(0.0, 1.0, 77, dtype=np.float32)

    first = attack.score(sample, sample_index=5, phase=PHASE_ATTACK)
    second = attack.score(sample, sample_index=5, phase=PHASE_ATTACK)
    victim = attack.score(sample, sample_index=5, phase=PHASE_VICTIM)

    assert first == second
    assert np.isfinite(first)
    assert np.isfinite(victim)


def test_score_rejects_an_unknown_phase(tiny_model, meta):
    attack = _attack(tiny_model, meta)
    with pytest.raises(ValueError, match="phase"):
        attack.score(np.zeros(77, dtype=np.float32), 0, "not-a-phase")


def test_n_masks_must_be_divisible_by_five(tiny_model, meta):
    with pytest.raises(ValueError, match="divisible"):
        _attack(tiny_model, meta, n_masks=7)


def test_num_input_must_be_seventy_seven(tiny_model, meta):
    with pytest.raises(ValueError, match="77"):
        _attack(tiny_model, meta, num_input=76)


def test_decimal_features_out_of_range_are_rejected(tiny_model, meta):
    with pytest.raises(ValueError, match="decimal"):
        _attack(tiny_model, meta, decimal_features=[77])
