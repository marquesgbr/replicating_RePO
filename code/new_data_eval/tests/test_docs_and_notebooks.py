"""Documentation and notebook wrapper checks for the new dataset."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _notebook_code(path: Path) -> str:
    """Return the concatenated source of every code cell of ``path``."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_new_dataset_doc_has_required_sections():
    text = (REPO_ROOT / "docs" / "NEW_DATASET.md").read_text(encoding="utf-8")
    for section in ("Fonte e citação", "Split", "Protocolo", "Comandos"):
        assert section in text


def test_new_dataset_doc_documents_source_split_and_features():
    text = (REPO_ROOT / "docs" / "NEW_DATASET.md").read_text(encoding="utf-8")
    assert "pendente" in text.lower()
    assert "New-dataset-split-clean.zip" in text
    assert "new_train.csv" in text
    assert "new_test.csv" in text
    assert "77" in text
    assert "Dst Port" in text
    assert "31" in text and "32" in text and "33" in text
    assert "constante" in text.lower()


def test_score_files_are_gitignored():
    output = subprocess.run(
        ["git", "check-ignore", "results/new_data/scores/example.npz"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert output.returncode == 0, output.stderr


def test_normal_notebook_uses_package_entrypoint():
    source = _notebook_code(REPO_ROOT / "code" / "new_data_RePO_Normal_Test.ipynb")
    assert "new_data_eval" in source


def test_normal_notebook_has_no_duplicated_scoring_logic():
    source = _notebook_code(REPO_ROOT / "code" / "new_data_RePO_Normal_Test.ipynb")
    for token in ("def get_test_set", "def get_scores", "def test_step", "fpr = 0.01"):
        assert token not in source


def test_adversarial_notebook_uses_package_entrypoint():
    source = _notebook_code(REPO_ROOT / "code" / "new_data_RePO_Adversarial.ipynb")
    assert "new_data_eval" in source


def test_adversarial_notebook_has_no_hardcoded_threshold_attack_or_result_path():
    source = _notebook_code(REPO_ROOT / "code" / "new_data_RePO_Adversarial.ipynb")
    for token in (
        "real_thr",
        "attack_type",
        "flow_based_adversarial_results",
        "../results",
        "DDoS",
    ):
        assert token not in source


def test_notebooks_use_isolated_smoke_results_directory():
    for name in (
        "new_data_RePO_Normal_Test.ipynb",
        "new_data_RePO_Adversarial.ipynb",
    ):
        source = _notebook_code(REPO_ROOT / "code" / name)
        assert "notebook_smoke" in source
        assignments = [
            line
            for line in source.splitlines()
            if line.strip().startswith("RESULTS_DIR =")
        ]
        assert len(assignments) == 1, assignments
        assert "DEFAULT_RESULTS_DIR" in assignments[0]
        assert '"notebook_smoke"' in assignments[0]
        # The canonical results root must never be the notebook output root.
        for token in (
            '"results" / "new_data"',
            "results/new_data/normal",
            "results/new_data/adversarial",
            "adversarial/primary",
        ):
            assert token not in source
