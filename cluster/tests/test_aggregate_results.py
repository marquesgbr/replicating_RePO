from __future__ import annotations

import json
import sys
from pathlib import Path

CLUSTER_DIR = Path(__file__).resolve().parents[1]
if str(CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_DIR))

import aggregate_results  # noqa: E402


def write_result(root: Path, relative: str, payload: dict) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def base_result(**overrides):
    payload = {
        "schema_version": 2,
        "status": "completed",
        "run_id": "run-1",
        "experiment": "normal",
        "model_type": "flow",
        "score_mode": "repo_plus",
        "attack_type": None,
        "threshold": 0.1,
        "threshold_fpr": 0.01,
        "fpr": 0.01,
        "class_rates": {
            "BENIGN": {"rate": 0.01, "count": 100},
            "DoS Hulk": {"rate": 0.5, "count": 10},
        },
        "average_tpr": 0.5,
        "n_samples": 110,
        "runtime_seconds": 10.0,
        "finished_at": "2026-10-29T00:00:00+00:00",
        "config": {"git": {"commit": "abc"}},
    }
    payload.update(overrides)
    return payload


def test_load_deduplicate_and_flatten(tmp_path):
    write_result(tmp_path, "flow/normal/repo_plus.json", base_result())
    write_result(
        tmp_path,
        "flow/normal/repo_plus-old.json",
        base_result(finished_at="2026-10-28T00:00:00+00:00", threshold=0.2),
    )
    write_result(
        tmp_path,
        "manifests/ignore.json",
        base_result(experiment="normal"),
    )

    loaded = aggregate_results.load_results(tmp_path)
    assert len(loaded) == 2
    deduplicated = aggregate_results.deduplicate(loaded)
    assert len(deduplicated) == 1
    assert deduplicated[0]["threshold"] == 0.1

    row = aggregate_results.flatten(deduplicated[0])
    assert row["git_commit"] == "abc"
    assert row["class_rates"]["DoS Hulk"]["rate"] == 0.5
    assert row["paper_average_tpr"] == 0.2549


def test_missing_results_lists_normal_and_attacks():
    missing = aggregate_results.missing_results([])
    assert any(item.startswith("normal flow/repo") for item in missing)
    assert any(item.startswith("adversarial packet/FTP-Patator") for item in missing)
    assert not any("Web-Attack" in item for item in missing)


def test_write_csv_and_markdown(tmp_path):
    rows = [aggregate_results.flatten(base_result())]
    missing = aggregate_results.missing_results(rows)
    aggregate_results.write_csv(tmp_path / "summary.csv", rows)
    aggregate_results.write_markdown(tmp_path / "summary.md", rows, missing)

    csv_text = (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "model_type" in csv_text
    assert "flow" in csv_text
    markdown = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Avaliação normal" in markdown
    assert "DoS Hulk" in markdown
    assert "Resultados ausentes" in markdown
