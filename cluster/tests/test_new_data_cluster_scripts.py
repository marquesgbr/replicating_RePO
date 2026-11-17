from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLUSTER_DIR = REPO_ROOT / "cluster"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def cluster_env(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    results = tmp_path / "results"
    results.mkdir()

    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text("", encoding="utf-8")
    _write_executable(
        venv / "bin" / "python",
        """#!/bin/sh
if [ "${PY_MODE:-run}" = "log" ]; then
  printf '%s\\n' "$*" >> "$PY_LOG"
  exit 0
fi
exec "$REAL_PYTHON" "$@"
""",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sbatch",
        """#!/bin/sh
printf '%s\\n' "$*" >> "$SBATCH_LOG"
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
            "PY_LOG": str(tmp_path / "python.log"),
            "REAL_PYTHON": sys.executable,
            "PY_MODE": "run",
            "MODEL_SOURCE": "pretrained",
            "SEED": "2019",
        }
    )
    env.pop("MODEL_DIR", None)
    return env, tmp_path


def _run(script: Path, args: list[str], env: dict[str, str], cwd: Path = REPO_ROOT):
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
    )


def test_new_data_model_dir_honours_model_source(cluster_env):
    env, _tmp = cluster_env
    env["MODEL_SOURCE"] = "retrained"

    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source cluster/common.sh; new_data_model_dir_for "$MODEL_SOURCE"',
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(
        REPO_ROOT / "models" / "retrained" / "new_data_model"
    )


def test_submit_normal_forwards_modes_and_scoring_env(cluster_env):
    env, tmp_path = cluster_env
    env.update({"NEW_DATA_MODES": "repo", "N_MASKS": "5", "PAIRED_SCORING": "1"})

    completed = _run(
        CLUSTER_DIR / "submit.sh", ["new-data", "normal"], env
    )

    assert completed.returncode == 0, completed.stderr
    call = (tmp_path / "sbatch.log").read_text(encoding="utf-8")
    assert "NEW_DATA_MODES=repo," in call or "NEW_DATA_MODES=repo " in call
    assert "N_MASKS=5" in call
    assert "PAIRED_SCORING=1" in call


def test_submit_adversarial_all_builds_nine_attack_array(cluster_env):
    env, tmp_path = cluster_env
    env["CONCURRENCY"] = "2"
    env["DEPENDENCY"] = "afterok:29540"

    completed = _run(
        CLUSTER_DIR / "submit.sh",
        ["new-data", "adversarial", "all", "primary"],
        env,
    )

    assert completed.returncode == 0, completed.stderr
    call = (tmp_path / "sbatch.log").read_text(encoding="utf-8")
    assert "--array=0-8%2" in call
    assert "--dependency afterok:29540" in call
    assert "VARIANT=primary" in call
    assert "SAMPLE_MODE=random" in call
    assert "THRESHOLD_POLICY=calibrated" in call

    match = re.search(r"ATTACK_LIST_FILE=([^,\s]+)", call)
    assert match is not None
    attacks = Path(match.group(1)).read_text(encoding="utf-8").split()
    assert len(attacks) == 9
    assert "ddos" in attacks
    assert "xss" in attacks


def test_submit_export_job_is_cpu_only_and_dependency_aware(cluster_env):
    env, tmp_path = cluster_env
    env["DEPENDENCY"] = "afterok:999"
    export_dir = tmp_path / "export"
    env["NEW_DATA_EXPORT_DIR"] = str(export_dir)

    completed = _run(
        CLUSTER_DIR / "submit.sh", ["new-data", "export-job"], env
    )

    assert completed.returncode == 0, completed.stderr
    call = (tmp_path / "sbatch.log").read_text(encoding="utf-8")
    assert "--dependency afterok:999" in call
    assert "--gres" not in call
    assert "TASK=new_data_export" in call
    assert f"NEW_DATA_EXPORT_DIR={export_dir}" in call


def test_job_export_runs_the_aggregator(cluster_env):
    env, tmp_path = cluster_env
    env["PY_MODE"] = "log"
    export_dir = tmp_path / "export"

    completed = subprocess.run(
        ["bash", "cluster/slurm/job.sh"],
        cwd=REPO_ROOT,
        env={
            **env,
            "TASK": "new_data_export",
            "MODEL": "new_data",
            "NEW_DATA_EXPORT_DIR": str(export_dir),
        },
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    argv = (tmp_path / "python.log").read_text(encoding="utf-8")
    assert "cluster/aggregate_new_data.py" in argv
    assert f"--results-dir {tmp_path / 'results' / 'new_data'}" in argv
    assert f"--out {export_dir}" in argv


def test_job_uses_the_retrained_model_and_the_dataset_results_root(cluster_env):
    env, tmp_path = cluster_env
    env.update({"PY_MODE": "log", "MODEL_SOURCE": "retrained"})

    completed = subprocess.run(
        ["bash", "cluster/slurm/job.sh"],
        cwd=REPO_ROOT,
        env={
            **env,
            "TASK": "new_data_normal",
            "MODEL": "new_data",
            "NEW_DATA_MODES": "repo+repo_plus",
            "BATCH_SIZE": "1024",
        },
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    argv = (tmp_path / "python.log").read_text(encoding="utf-8")
    assert str(REPO_ROOT / "models" / "retrained" / "new_data_model") in argv
    assert " --modes repo,repo_plus" in argv


def test_job_adversarial_fails_before_scoring_when_normal_is_missing(cluster_env):
    env, tmp_path = cluster_env
    env["PY_MODE"] = "log"

    completed = subprocess.run(
        ["bash", "cluster/slurm/job.sh"],
        cwd=REPO_ROOT,
        env={
            **env,
            "TASK": "new_data_adversarial",
            "MODEL": "new_data",
            "ATTACK": "ddos",
        },
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "normal ausente" in completed.stderr
    assert not (tmp_path / "python.log").exists()


def test_job_smoke_passes_the_dataset_root(cluster_env):
    env, tmp_path = cluster_env
    env["PY_MODE"] = "log"

    completed = subprocess.run(
        ["bash", "cluster/slurm/job.sh"],
        cwd=REPO_ROOT,
        env={**env, "TASK": "new_data_smoke", "MODEL": "new_data", "ATTACK": "ddos"},
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    argv = (tmp_path / "python.log").read_text(encoding="utf-8")
    results_root = tmp_path / "results" / "new_data"
    assert f"--results-dir {results_root} " in argv
    assert f"--results-dir {results_root}/smoke" not in argv


def test_train_guard_rejects_protected_paths(cluster_env):
    env, tmp_path = cluster_env
    env["MODEL_DIR"] = "models/new_data_model"

    completed = _run(CLUSTER_DIR / "submit.sh", ["new-data", "train"], env)

    assert completed.returncode == 2
    assert "modelo dos autores" in completed.stderr
    assert not (tmp_path / "sbatch.log").exists()

    job = subprocess.run(
        ["bash", "cluster/slurm/job.sh"],
        cwd=REPO_ROOT,
        env={
            **env,
            "TASK": "new_data_train",
            "MODEL": "new_data",
            "PY_MODE": "log",
        },
        text=True,
        capture_output=True,
    )
    assert job.returncode == 2
    assert "modelo dos autores" in job.stderr
