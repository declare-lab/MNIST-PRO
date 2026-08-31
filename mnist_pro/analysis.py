"""Result loading and table generation."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass

from .matrix import Cell, discover_runs


@dataclass
class RunResult:
    cell: Cell
    path: str
    metrics: dict
    episodes: list

    @property
    def accuracy(self):
        return self.metrics.get("accuracy")

    @property
    def average_steps(self):
        return self.metrics.get("average_steps")


def load_results(results_dir: str) -> list[RunResult]:
    out = []
    for run in discover_runs(results_dir):
        summary = os.path.join(run.path, "results_summary.json")
        if not os.path.exists(summary):
            continue
        try:
            with open(summary) as f:
                doc = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        out.append(
            RunResult(
                cell=run.cell,
                path=run.path,
                metrics=doc.get("metrics", {}),
                episodes=doc.get("episodes", []),
            )
        )
    return out


def latest_per_cell(results: list[RunResult]) -> dict:
    """Most recently modified run for each cell."""
    best: dict = {}
    for r in results:
        key = r.cell.key()
        if key not in best or os.path.getmtime(r.path) > os.path.getmtime(
            best[key].path
        ):
            best[key] = r
    return best


def to_csv(results: list[RunResult], path: str) -> str:
    fields = [
        "model",
        "digits",
        "memory",
        "horizon",
        "harness",
        "arm",
        "box_size",
        "accuracy",
        "average_steps",
        "average_stroke_coverage",
        "average_stroke_coverage_readable",
        "control_accuracy",
        "total_episodes",
        "failed_episodes",
        "path",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {**r.cell.to_dict()}
            row.update(
                {
                    k: r.metrics.get(k)
                    for k in (
                        "accuracy",
                        "average_steps",
                        "average_stroke_coverage",
                        "average_stroke_coverage_readable",
                        "control_accuracy",
                        "total_episodes",
                        "failed_episodes",
                    )
                }
            )
            row["path"] = r.path
            w.writerow(row)
    return path


def main_table(results: list[RunResult], digits: int = 1) -> list[dict]:
    """Accuracy and steps per (model, memory, horizon), for one digit count."""
    rows = []
    for r in latest_per_cell(results).values():
        if r.cell.digits != digits:
            continue
        rows.append(
            {
                "model": r.cell.model,
                "memory": r.cell.memory,
                "horizon": r.cell.horizon,
                "harness": r.cell.harness,
                "arm": r.cell.arm,
                "accuracy": r.accuracy,
                "average_steps": r.average_steps,
                "control_accuracy": r.metrics.get("control_accuracy"),
            }
        )
    rows.sort(key=lambda x: (x["model"], x["memory"], x["horizon"]))
    return rows
