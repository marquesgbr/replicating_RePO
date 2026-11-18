"""Structural tests of the sequential-attack SLURM wiring.

The scripts are not executed here (they need the cluster); the tests pin the
entry points the study depends on and the bash syntax.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
JOB_SH = REPO_ROOT / "cluster" / "slurm" / "job.sh"
SUBMIT_SH = REPO_ROOT / "cluster" / "submit.sh"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_job_sh_is_valid_bash_and_defines_the_task():
    subprocess.run(["bash", "-n", str(JOB_SH)], check=True)
    text = _text(JOB_SH)
    assert "mask_sequential_attack)" in text
    assert "sequential_attack/${ATTACK}" in text
    assert "MASK_SEQ_BLOCKS" in text
    assert "mask_experiments.sequential_attack" in text


def test_job_sh_decodes_the_plus_separated_block_list():
    text = _text(JOB_SH)
    assert 'MASK_SEQ_BLOCKS="${MASK_SEQ_BLOCKS//+/,}"' in text


def test_submit_sh_is_valid_bash_and_exposes_the_command():
    subprocess.run(["bash", "-n", str(SUBMIT_SH)], check=True)
    text = _text(SUBMIT_SH)
    assert 'sequential-attack) task="mask_sequential_attack"' in text
    assert 'task_mask_attack sequential-attack' in text
    assert "MASK_SEQ_BLOCKS" in text
    assert "mask-exp sequential-attack <ataque|all>" in text
