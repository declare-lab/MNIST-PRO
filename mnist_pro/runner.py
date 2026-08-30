"""The single evaluation driver.

Replaces `evaluate.py` and `evaluate_multidigit.py`, which were 495 and 526 lines and
83% textually identical -- every change had to be made twice, and they had drifted.
Digit count is a parameter here, not a fork.

What each run writes:

    run_config.json        the exact Cell, so nothing is ever recovered by parsing a
                           directory name again
    results_summary.json   metrics plus one entry per episode
    episode_<i>/
        original.png       the unmasked canvas
        step_<n>.png       every observation shown to the agent
        trajectory.json    from TrajectoryRecorder: windows, actions, rewards,
                           per-step latency and token usage, termination reason
        response_<n>.json  the raw model response, saved before anything parses it

A failed episode is recorded with its error and the sweep continues; previously a
backend that exhausted its retries raised and killed the whole run.
"""

from __future__ import annotations

import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from PIL import Image

from .agents import AgentConfig, GlimpseAgent
from .backends import get_backend
from .dataset import EpisodeSpec, images_for, load_mnist, sample_balanced
from .env import TerminationReason
from .matrix import Cell
from .metrics import exploration_stats
from .rendering import CanvasSpec
from .wrappers import make_env


def run_dir_name(cell: Cell, evalsets: int, timestamp: str) -> str:
    """Kept compatible with the released naming so old and new runs coexist."""
    prefix = f"multidigit_{cell.digits}_" if cell.digits > 1 else ""
    legacy = {"visual_buffer": "SensoryVisionAgent", "event_logging": "DefaultVisionAgent",
              "textual_belief_state": "MemoryVisionAgent",
              "metric_grid_map": "SpatialMemoryVisionAgent"}[cell.memory]
    if cell.digits > 1:
        legacy = "MultiDigit" + legacy
    max_steps = default_max_steps(cell)
    return (f"{prefix}{cell.model}_img{cell.image_size}_box{cell.box_size}"
            f"_step{cell.step_size}_maxsteps{max_steps}_seed{cell.seed}"
            f"_{legacy}_hist{cell.horizon}_evalsets{evalsets}_{timestamp}")


def default_max_steps(cell: Cell) -> int:
    """Steps needed to sweep the canvas, scaled by its area. Matches the drivers."""
    span_x = (cell.image_size * cell.digits - cell.box_size) // cell.step_size + 1
    span_y = (cell.image_size - cell.box_size) // cell.step_size + 1
    return span_x * span_y + 2


def run_episode(spec: EpisodeSpec, cell: Cell, dataset, out_root: str,
                max_steps: int, draw_border: bool = True) -> dict:
    ep_dir = os.path.join(out_root, f"episode_{spec.episode_id}")
    os.makedirs(ep_dir, exist_ok=True)
    canvas_spec = CanvasSpec(digits=cell.digits, image_size=cell.image_size,
                             box_size=cell.box_size, step_size=cell.step_size)
    images = images_for(dataset, spec)

    env = make_env(images, spec.label, spec=canvas_spec, max_steps=max_steps,
                   draw_border=draw_border)
    Image.fromarray(env.canvas).save(os.path.join(ep_dir, "original.png"))

    agent = GlimpseAgent(
        backend=get_backend(cell.model),
        config=AgentConfig(memory=cell.memory, digits=cell.digits,
                           horizon=cell.horizon, turn_mode=cell.turn_mode,
                           max_steps=max_steps))

    obs, _ = env.reset()
    obs.save(os.path.join(ep_dir, "step_0.png"))
    error = None
    try:
        terminated = truncated = False
        while not (terminated or truncated):
            action, raw, usage = agent.act(obs)
            obs, _, terminated, truncated, _ = env.step(action, raw_response=raw,
                                                        usage=usage)
            obs.save(os.path.join(ep_dir, f"step_{env.steps}.png"))
            with open(os.path.join(ep_dir, f"response_{env.steps}.json"), "w") as f:
                json.dump({"step": env.steps, "raw_response": raw, "usage": usage},
                          f, indent=2)
    except Exception as exc:  # recorded, never fatal to the sweep
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    record = env.record
    payload = record.to_dict()
    payload.update({"episode_id": spec.episode_id, "indices": list(spec.indices),
                    "error": error})
    payload.update(exploration_stats(env.canvas, record.windows, canvas_spec))
    with open(os.path.join(ep_dir, "trajectory.json"), "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def run_control(spec: EpisodeSpec, cell: Cell, dataset) -> dict:
    """The unmasked-canvas control, through the same agent object as the main path."""
    canvas_spec = CanvasSpec(digits=cell.digits, image_size=cell.image_size,
                             box_size=cell.box_size, step_size=cell.step_size)
    from .rendering import build_canvas
    canvas = build_canvas(images_for(dataset, spec), canvas_spec)
    agent = GlimpseAgent(backend=get_backend(cell.model),
                         config=AgentConfig(memory=cell.memory, digits=cell.digits,
                                            horizon=cell.horizon))
    raw, value, _ = agent.predict_full_image(Image.fromarray(canvas))
    return {"episode_id": spec.episode_id, "indices": list(spec.indices),
            "label": spec.label, "control_prediction": value,
            "control_success": value == spec.label, "raw_response": raw}


def run_cell(cell: Cell, results_dir: str, evalsets: int = 10, workers: int = 10,
             data_dir: str = "data", limit: int | None = None,
             draw_border: bool = True, with_control: bool = True) -> str:
    dataset = load_mnist(data_dir)
    specs = sample_balanced(dataset, evalsets, digits=cell.digits, seed=cell.seed)
    if limit:
        specs = specs[:limit]
    max_steps = default_max_steps(cell)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(results_dir, run_dir_name(cell, evalsets, timestamp))
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "run_config.json"), "w") as f:
        json.dump({**cell.to_dict(), "max_steps": max_steps, "evalsets": evalsets,
                   "draw_border": draw_border, "timestamp": timestamp}, f, indent=2)

    episodes = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_episode, s, cell, dataset, out_root, max_steps,
                               draw_border): s for s in specs}
        for fut in as_completed(futures):
            episodes.append(fut.result())
    episodes.sort(key=lambda e: e["episode_id"])

    controls = {}
    if with_control:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_control, s, cell, dataset) for s in specs]
            for fut in as_completed(futures):
                c = fut.result()
                controls[c["episode_id"]] = c
        for e in episodes:
            c = controls.get(e["episode_id"], {})
            e["control_prediction"] = c.get("control_prediction")
            e["control_success"] = c.get("control_success")

    scored = [e for e in episodes if e.get("error") is None]
    reasons = {}
    for e in episodes:
        reasons[e.get("termination_reason")] = reasons.get(e.get("termination_reason"), 0) + 1
    metrics = {
        "total_episodes": len(episodes),
        "scored_episodes": len(scored),
        "failed_episodes": len(episodes) - len(scored),
        "accuracy": (sum(e["success"] for e in scored) / len(scored)) if scored else None,
        "average_steps": (sum(e["n_steps"] for e in scored) / len(scored)) if scored else None,
        "average_stroke_coverage": (sum(e["stroke_coverage"] for e in scored) / len(scored))
                                   if scored else None,
        "average_stroke_coverage_readable":
            (sum(e["stroke_coverage_readable"] for e in scored) / len(scored))
            if scored else None,
        "termination_reasons": reasons,
    }
    if with_control:
        ok = [e for e in episodes if e.get("control_success") is not None]
        metrics["control_accuracy"] = (sum(e["control_success"] for e in ok) / len(ok)) if ok else None

    with open(os.path.join(out_root, "results_summary.json"), "w") as f:
        json.dump({"run_config": {**cell.to_dict(), "max_steps": max_steps},
                   "metrics": metrics, "episodes": episodes}, f, indent=2)
    return out_root
