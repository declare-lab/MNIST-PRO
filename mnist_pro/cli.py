"""Command line entry point: `mnist-pro <command>`."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .analysis import load_results, main_table, to_csv
from .matrix import Cell, confounds, load_expectations, load_matrix, status


def cmd_run(args):
    from .runner import run_cell
    cell = Cell(model=args.model, digits=args.digits, memory=args.memory,
                horizon=args.horizon, turn_mode=args.turn_mode,
                harness=args.harness, arm=args.arm, box_size=args.box_size,
                step_size=args.step_size, image_size=args.image_size, seed=args.seed)
    path = run_cell(cell, results_dir=args.results_dir, evalsets=args.evalsets,
                    workers=args.workers, data_dir=args.data_dir, limit=args.limit,
                    draw_border=not args.no_border, with_control=not args.no_control)
    print(f"wrote {path}")


def cmd_matrix(args):
    matrix = load_matrix(args.config)
    st = status(matrix, args.results_dir)
    print(f"declared cells: {st['n_declared']}    runs found: {st['n_runs']}\n")
    print(f"PRESENT ({len(st['present'])})")
    for entry in st["present"]:
        acc = entry["runs"][0].metrics.get("accuracy")
        acc_txt = "--" if acc is None else f"{acc:.2f}"
        print(f"  ok   {entry['cell'].label():<70} acc {acc_txt}")
    print(f"\nMISSING ({len(st['missing'])})")
    for entry in st["missing"]:
        print(f"  --   {entry['cell'].label()}")
    if st["undeclared"]:
        print(f"\nUNDECLARED RUNS ({len(st['undeclared'])})")
        for run in st["undeclared"]:
            print(f"  ?    {run.cell.label():<70} {os.path.basename(run.path)}")
    warns = confounds(matrix, load_expectations(args.config) or None)
    if warns:
        print("\nCONFOUNDS")
        for w in warns:
            print(f"  !    {w}")
    return 1 if st["missing"] else 0


def cmd_analyse(args):
    results = load_results(args.results_dir)
    if not results:
        print(f"no runs found under {args.results_dir}")
        return 1
    if args.csv:
        print(f"wrote {to_csv(results, args.csv)}")
    for digits in sorted({r.cell.digits for r in results}):
        print(f"\n=== Level {digits} ===")
        for row in main_table(results, digits=digits):
            acc = "--" if row["accuracy"] is None else f"{row['accuracy']:.2f}"
            steps = "--" if row["average_steps"] is None else f"{row['average_steps']:.2f}"
            print(f"  {row['model']:<24} {row['memory']:<22} H={row['horizon']:<3} "
                  f"{row['harness']}/{row['arm']:<4} acc {acc:>5}  steps {steps:>6}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="mnist-pro",
                                description="Active-glimpse evaluation framework")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="evaluate one cell")
    r.add_argument("--model", required=True)
    r.add_argument("--digits", type=int, default=1)
    r.add_argument("--memory", default="textual_belief_state")
    r.add_argument("--horizon", type=int, default=1)
    r.add_argument("--turn-mode", default="turn_based", dest="turn_mode")
    r.add_argument("--harness", default="turn_based")
    r.add_argument("--arm", default="A0")
    r.add_argument("--box-size", type=int, default=64, dest="box_size")
    r.add_argument("--step-size", type=int, default=32, dest="step_size")
    r.add_argument("--image-size", type=int, default=224, dest="image_size")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--evalsets", type=int, default=10)
    r.add_argument("--workers", type=int, default=10)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--data-dir", default="data", dest="data_dir")
    r.add_argument("--results-dir", default="results", dest="results_dir")
    r.add_argument("--no-border", action="store_true",
                   help="omit the cyan outline that occludes the glimpse edge")
    r.add_argument("--no-control", action="store_true")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("matrix", help="report present, missing and confounded cells")
    m.add_argument("--config", default="configs/main_table.yaml")
    m.add_argument("--results-dir", default="results", dest="results_dir")
    m.set_defaults(func=cmd_matrix)

    a = sub.add_parser("analyse", help="summarise any results directory")
    a.add_argument("--results-dir", default="results", dest="results_dir")
    a.add_argument("--csv", default=None)
    a.set_defaults(func=cmd_analyse)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
