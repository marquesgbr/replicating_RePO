"""Persistence helpers for adversarial experiment metrics.

The JSONL format stores one experiment result per line, which makes it
possible to append results while running several attacks.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ExperimentResult:
    """Metrics collected from one adversarial experiment."""

    best_threshold: float
    victim_tpr: float
    attacker_tpr: float
    total_runtime_seconds: float
    attack_type: str | None = None
    model_type: str | None = None
    normal_tpr: float | None = None

    def __post_init__(self) -> None:
        if self.best_threshold < 0 or not math.isfinite(self.best_threshold):
            raise ValueError("best_threshold must be a finite non-negative number")
        for name, value in (
            ("victim_tpr", self.victim_tpr),
            ("attacker_tpr", self.attacker_tpr),
            ("normal_tpr", self.normal_tpr),
        ):
            if value is not None and (
                not math.isfinite(value) or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be a finite number between 0 and 1")
        if (
            not math.isfinite(self.total_runtime_seconds)
            or self.total_runtime_seconds < 0
        ):
            raise ValueError(
                "total_runtime_seconds must be a finite non-negative number"
            )
        for name, value in (
            ("attack_type", self.attack_type),
            ("model_type", self.model_type),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"{name} cannot be empty")

def save_result(result: ExperimentResult, output_path: str | Path) -> None:
    """Append one experiment result to a JSONL file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        json.dump(asdict(result), file, ensure_ascii=False, allow_nan=False)
        file.write("\n")


def save_results(
    results: Iterable[ExperimentResult], output_path: str | Path
) -> None:
    """Append multiple experiment results to a JSONL file."""

    for result in results:
        save_result(result, output_path)


def save_result_json(result: ExperimentResult, output_path: str | Path) -> None:
    """Append one experiment result to a JSON array file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}")
        results = data
    results.append(asdict(result))
    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
        file.write("\n")


def load_results(input_path: str | Path) -> list[ExperimentResult]:
    """Load all experiment results from a JSON or JSONL file.

    Raises:
        FileNotFoundError: if ``input_path`` does not exist.
        ValueError: if a line does not contain a valid experiment result.
    """

    path = Path(input_path)
    results: list[ExperimentResult] = []
    with path.open("r", encoding="utf-8") as file:
        content = file.read()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, list):
        try:
            return [ExperimentResult(**item) for item in data]
        except (TypeError, KeyError, ValueError) as error:
            raise ValueError(f"Invalid experiment result array in {path}") from error

    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            results.append(ExperimentResult(**data))
        except (TypeError, KeyError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid experiment result on line {line_number} of {path}"
            ) from error
    return results
