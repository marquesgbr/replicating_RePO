"""Tests for the Markdown aggregator (no TensorFlow needed)."""

from __future__ import annotations

import json
from pathlib import Path

from mask_experiments.aggregate import build_markdown


def test_missing_results_are_reported_as_placeholders(tmp_path):
    markdown = build_markdown(tmp_path)
    assert "Eficiência do sistema de máscaras" in markdown
    assert "Baseline ainda não executado" in markdown
    assert "Estudo de variância ainda não executado" in markdown


def test_tables_are_rendered_from_the_json_documents(tmp_path):
    variance = {
        "results": {
            "iid": {
                "25": {
                    "repo_plus": {
                        "thresholds": {
                            "fpr_0.01": {
                                "tpr": {
                                    "macro": 0.2,
                                    "macro_std": 0.01,
                                    "weighted": 0.1,
                                }
                            }
                        },
                        "score_std_across_repeats": 0.01,
                        "spearman_vs_reference": 0.9,
                    },
                    "cost": {"masked_rows": 1234},
                }
            }
        }
    }
    (tmp_path / "variance_flow.json").write_text(json.dumps(variance))
    markdown = build_markdown(tmp_path)
    assert "| iid | 25 | 0.2000 | 0.0100 | 0.1000 |" in markdown
    assert "1234" in markdown


def test_adversarial_table_uses_the_victim_tpr(tmp_path):
    payload = {
        "variants": {
            "iid-25": {
                "real_threshold": 0.05,
                "attack_threshold": 0.04,
                "attacks": {
                    "DDoS": {
                        "n": 10,
                        "normal_tpr": 0.5,
                        "victim_tpr": 0.2,
                        "fooled": 3,
                        "kinds": {
                            "evaded": 3,
                            "backup": 0,
                            "no_change": 3,
                            "not_fooled": 4,
                        },
                        "outcomes": [
                            {"masks_evaluated": 1000, "model_calls": 10}
                        ] * 10,
                    }
                }
            }
        }
    }
    (tmp_path / "adversarial_pilot.json").write_text(json.dumps(payload))
    markdown = build_markdown(tmp_path)
    assert "| iid-25 | DDoS | 10 | 0.5000 | 0.2000 | 3/10 | 3 | 1000 |" in markdown
