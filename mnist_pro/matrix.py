"""The evaluation matrix, declared in a config and checked against what exists.

`mnist-pro matrix status` prints a table of present and missing cells against
whichever results directory is passed. Nothing is hardcoded to `results/`.
"""

from __future__ import annotations

import itertools
import json
import os
import re
from dataclasses import asdict, dataclass, field

import yaml

from .agents.specs import MEMORY_SPECS

RUN_DIR_RE = re.compile(
    r"^(?:multidigit_(?P<digits>\d+)_)?(?P<model>.+?)"
    r"_img(?P<image_size>\d+)_box(?P<box>\d+)_step(?P<step>\d+)"
    r"_maxsteps(?P<max_steps>\d+)_seed(?P<seed>\d+)"
    r"_(?P<agent>[A-Za-z]+)_hist(?P<hist>-?\d+)"
    r"_evalsets(?P<evalsets>\d+)_(?P<timestamp>\d{8}_\d{6})$"
)


# How the agent is driven, sourced from the harness registry.
from .harness.registry import HARNESSES as _HARNESS_SPECS

HARNESSES = tuple(_HARNESS_SPECS)

# Cross-episode learning condition, from the Antigravity suite. Orthogonal to the
# within-episode memory taxonomy: A0 carries nothing between episodes, A1 carries a
# persistent NOTES.md, A2 additionally receives correctness feedback on submission.
ARMS = ("A0", "A1", "A2")


@dataclass(frozen=True)
class Cell:
    """One intended evaluation condition."""

    model: str
    digits: int = 1
    memory: str = "textual_state"
    horizon: int = -1
    turn_mode: str = "natural"
    harness: str = "natural"
    arm: str = "A0"
    box_size: int = 64
    step_size: int = 32
    image_size: int = 224
    seed: int = 42

    def __post_init__(self):
        if self.harness not in HARNESSES:
            raise ValueError(f"unknown harness {self.harness!r}; expected {HARNESSES}")
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; expected {ARMS}")

    @property
    def level(self) -> int:
        return self.digits

    def key(self) -> tuple:
        return (
            self.model,
            self.digits,
            self.memory,
            self.horizon,
            self.harness,
            self.arm,
            self.box_size,
        )

    def label(self) -> str:
        return (
            f"L{self.digits} {self.model} {self.memory} H={self.horizon} "
            f"{self.harness}/{self.arm} box{self.box_size}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DiscoveredRun:
    path: str
    cell: Cell
    n_episodes: int = 0
    metrics: dict = field(default_factory=dict)
    source: str = "run_config.json"


def load_expectations(path: str) -> list[tuple]:
    """Axis pairs the study intends to cross fully, from `expect_crossed:`.

    This restricts checks to the pairs that matter, avoiding noise from partial ablations.
    """
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    return [tuple(pair) for pair in doc.get("expect_crossed", []) or []]


def load_matrix(path: str) -> list[Cell]:
    """Expand a YAML declaration into concrete cells.

    Each block is a cross-product over list-valued fields.
    """
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    defaults = doc.get("defaults", {}) or {}
    cells: list[Cell] = []
    for block in doc.get("matrix", []) or []:
        merged = {**defaults, **block}
        keys = list(merged)
        values = [v if isinstance(v, list) else [v] for v in merged.values()]
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            memory = params.get("memory")
            if memory not in MEMORY_SPECS:
                raise ValueError(f"unknown memory config {memory!r} in {path}")
            cells.append(Cell(**params))
    # de-duplicate, preserving declaration order
    seen, unique = set(), []
    for c in cells:
        if c.key() not in seen:
            seen.add(c.key())
            unique.append(c)
    return unique


def discover_runs(results_dir: str) -> list[DiscoveredRun]:
    """Find completed runs using their run_config.json files."""
    found: list[DiscoveredRun] = []
    if not os.path.isdir(results_dir):
        return found
    for name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, name)
        if not os.path.isdir(path):
            continue
        summary_path = os.path.join(path, "results_summary.json")
        config_path = os.path.join(path, "run_config.json")
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            known = {k: v for k, v in cfg.items() if k in Cell.__dataclass_fields__}
            cell = Cell(**known)
        except Exception:
            continue
        n, metrics = 0, {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
                n = len(summary.get("episodes", []))
                metrics = summary.get("metrics", {})
            except (json.JSONDecodeError, OSError):
                pass
        found.append(
            DiscoveredRun(
                path=path,
                cell=cell,
                n_episodes=n,
                metrics=metrics,
                source="run_config.json",
            )
        )
    return found


def status(matrix: list[Cell], results_dir: str) -> dict:
    """Which declared cells exist, and which are missing."""
    runs = discover_runs(results_dir)
    by_key: dict[tuple, list[DiscoveredRun]] = {}
    for r in runs:
        by_key.setdefault(r.cell.key(), []).append(r)
    present, missing = [], []
    for cell in matrix:
        hits = by_key.get(cell.key(), [])
        (present if hits else missing).append({"cell": cell, "runs": hits})
    undeclared = [r for r in runs if r.cell.key() not in {c.key() for c in matrix}]
    return {
        "present": present,
        "missing": missing,
        "undeclared": undeclared,
        "n_declared": len(matrix),
        "n_runs": len(runs),
    }


def confounds(matrix: list[Cell], expect: list[tuple] | None = None) -> list[str]:
    """Flag axes that cannot be varied independently in the declared matrix.

    A pair of axes is confounded when no two declared cells differ in exactly one of them.

    `expect` restricts the check to the pairs the study claims to cross. Passing None
    checks every pair, which is thorough but flags deliberately partial ablations too.
    """
    warnings = []
    axes = expect or list(
        itertools.combinations(("memory", "horizon", "harness", "arm", "box_size"), 2)
    )
    for a, b in axes:
        pairs = {(getattr(c, a), getattr(c, b)) for c in matrix}
        values_a = {p[0] for p in pairs}
        values_b = {p[1] for p in pairs}
        if (
            len(values_a) > 1
            and len(values_b) > 1
            and len(pairs) < len(values_a) * len(values_b)
        ):
            got = ", ".join(
                f"({x}, {y})"
                for x, y in sorted(tuple(str(v) for v in p) for p in pairs)
            )
            warnings.append(
                f"{a} and {b} are not fully crossed: {len(pairs)} of "
                f"{len(values_a) * len(values_b)} combinations declared [{got}]"
            )
    return warnings
