"""Tests for the chained mask-study submission script."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "cluster" / "run_mask_study.sh"
STEPS = (
    "baseline",
    "variance",
    "adversarial",
    "cascade",
    "cascade_attack",
    "distill",
    "distill_attack",
    "packet",
    "aggregate",
)


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _env(tmp_path: Path) -> dict[str, str]:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text("", encoding="utf-8")
    _write_executable(
        venv / "bin" / "python",
        f'#!/bin/sh\nexec "{sys.executable}" "$@"\n',
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sbatch",
        """#!/bin/sh
printf 'NOTE=%s ARGS=%s\\n' "${MASK_STEP_NOTE:-}" "$*" >> "$SBATCH_LOG"
echo 12345
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "SCRATCH": str(scratch),
            "RESULTS_ROOT": str(results),
            "VENV": str(venv),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "SBATCH_LOG": str(tmp_path / "sbatch.log"),
            "MODEL_SOURCE": "pretrained",
            "SEED": "2019",
        }
    )
    env.pop("MODEL_DIR", None)
    return env


def _run(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def test_dry_run_prints_every_step_without_submitting(tmp_path):
    env = _env(tmp_path)
    result = _run(["--smoke", "--dry-run"], env)

    assert result.returncode == 0, result.stderr
    for step in STEPS:
        assert f"[chain] {step}" in result.stdout
    manifest = tmp_path / "results" / "mask_efficiency" / "chain_manifest.tsv"
    rows = manifest.read_text(encoding="utf-8").splitlines()
    assert len(rows) == len(STEPS) + 1
    assert not (tmp_path / "sbatch.log").exists()


def test_smoke_chain_submits_all_steps_and_chains_dependencies(tmp_path):
    env = _env(tmp_path)
    result = _run(["--smoke"], env)

    assert result.returncode == 0, result.stderr + result.stdout
    log = (tmp_path / "sbatch.log").read_text(encoding="utf-8")
    assert log.count("ARGS=") == len(STEPS)
    # The student is trained by an earlier job, so the submission guard must be
    # bypassed when a dependency is present.
    assert "TASK=mask_distill_attack" in log
    assert "NOTE=" in log and "Ataque white-box no aluno" in log

    manifest = (
        tmp_path / "results" / "mask_efficiency" / "chain_manifest.tsv"
    ).read_text(encoding="utf-8")
    for step in STEPS:
        assert f"{step}\t12345\t" in manifest
    # aggregate depends on all eight prior steps.
    assert "afterok:12345:12345:12345:12345:12345:12345:12345:12345" in manifest


def test_no_aggregate_skips_the_last_step(tmp_path):
    env = _env(tmp_path)
    result = _run(["--no-aggregate", "--dry-run"], env)

    assert result.returncode == 0, result.stderr
    assert "[chain] aggregate" not in result.stdout
    manifest = (
        tmp_path / "results" / "mask_efficiency" / "chain_manifest.tsv"
    ).read_text(encoding="utf-8")
    assert "aggregate\t" not in manifest
