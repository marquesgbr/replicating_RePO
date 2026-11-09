from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

CLUSTER_DIR = Path(__file__).resolve().parents[1]
if str(CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_DIR))

import aggregate_new_data  # noqa: E402


def write_json(root: Path, relative: str, payload: dict) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def normal_payload(mode: str) -> dict:
    return {
        "schema_version": "new_data_v1",
        "model_type": "new_data_flow",
        "experiment": "normal",
        "score_mode": mode,
        "dataset": {"n_samples": 1000},
        "thresholds": {"fpr_0.01": 0.1, "fpr_0.10": 0.2},
        "fpr": {
            "at_fpr_0.01_threshold": 0.01,
            "at_fpr_0.10_threshold": 0.10,
        },
        "average_tpr": {
            "at_fpr_0.01": {"macro": 0.3, "weighted": 0.4},
            "at_fpr_0.10": {"macro": 0.5, "weighted": 0.6},
        },
        "runtime_seconds": 12.0,
        "config_hash": "a" * 64,
        "git_commit": "b" * 40,
    }


def adversarial_payload(variant: str, attack: str = "ddos") -> dict:
    return {
        "schema_version": "new_data_v1",
        "model_type": "new_data_flow",
        "experiment": "adversarial",
        "variant": variant,
        "attack": attack,
        "score_mode": "repo_plus",
        "thresholds": {"policy": "calibrated", "attack": 0.2, "fpr_0.10": 0.2},
        "fpr_at_attack_threshold": 0.10,
        "normal_tpr_at_attack_threshold": 0.8,
        "victim_tpr": 0.5,
        "attack_success_rate": 0.5,
        "conditional_drop": 0.375,
        "fooled_rate": 0.4,
        "counts": {
            "n": 10,
            "fooled": 4,
            "evaded": 3,
            "backup": 1,
            "cons_as_mal": 5,
            "cons_as_ben": 1,
        },
        "runtime_seconds": 30.0,
        "config_hash": "c" * 64,
        "git_commit": "d" * 40,
    }


def build_tree(root: Path) -> None:
    write_json(root, "normal/repo.json", normal_payload("repo"))
    write_json(root, "normal/repo_plus.json", normal_payload("repo_plus"))
    write_json(root, "adversarial/primary/ddos.json", adversarial_payload("primary"))
    write_json(
        root,
        "adversarial/sensitivity_first_n/ddos.json",
        adversarial_payload("sensitivity_first_n"),
    )
    # This is the isolated smoke layout and must be ignored by the aggregator.
    write_json(root, "smoke/adversarial/ddos.json", adversarial_payload("smoke"))


def test_summary_rows_reads_normal_and_adversarial(tmp_path):
    build_tree(tmp_path)

    rows, missing = aggregate_new_data.summary_rows(tmp_path)

    assert [row["experiment"] for row in rows] == [
        "normal",
        "normal",
        "adversarial",
        "adversarial",
    ]
    assert {row["variant"] for row in rows if row["experiment"] == "adversarial"} == {
        "primary",
        "sensitivity_first_n",
    }
    assert all("smoke" not in row["path"] for row in rows)
    assert "adversarial/primary/xss.json" in missing
    assert "adversarial/primary/ddos.json" not in missing


def test_write_summary_creates_csv_markdown_and_json(tmp_path):
    results = tmp_path / "results"
    out = tmp_path / "out"
    build_tree(results)

    aggregate_new_data.write_summary(results, out)

    with (out / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["experiment"] == "normal"
    assert rows[2]["attack"] == "ddos"
    assert rows[2]["victim_tpr"] == "0.5"
    assert rows[0]["macro_tpr"] == "0.5"

    markdown = (out / "summary.md").read_text(encoding="utf-8")
    assert "## Normal" in markdown
    assert "## Adversarial — primary" in markdown
    assert "## Adversarial — sensitivity_first_n" in markdown
    assert "xss" in markdown

    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert len(payload["results"]) == 4
    assert "adversarial/primary/xss.json" in payload["missing"]


def test_write_summary_handles_an_empty_results_tree(tmp_path):
    out = tmp_path / "out"
    aggregate_new_data.write_summary(tmp_path / "missing", out)

    assert (out / "summary.csv").is_file()
    markdown = (out / "summary.md").read_text(encoding="utf-8")
    assert "## Normal" in markdown
    assert "Nenhum resultado ausente" not in markdown
