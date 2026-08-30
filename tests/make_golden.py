#!/usr/bin/env python3
"""Regenerate the golden fixtures from released runs.

Copies a few `original.png` canvases and the observations rendered from them by the
original implementation, plus the window position recovered from each observation's
cyan border. `test_renderer_golden.py` then re-renders each observation from the
canvas and requires a byte-for-byte match.

Run only when deliberately re-baselining:
    python tests/make_golden.py --logs /path/to/main_table_logs
"""

import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image

BORDER = (0, 229, 255)

SOURCES = [
    ("L1", "gemini-3.7-flash_img224_box64_step32_maxsteps36_seed42"
           "_MemoryVisionAgent_hist1_evalsets10_20260824_110242", 1, [0, 7]),
    ("L2", "multidigit_2_gemini-3.7-flash_img224_box64_step32_maxsteps78_seed42"
           "_MultiDigitMemoryVisionAgent_hist1_evalsets10_20260824_114120", 2, [0, 4]),
]


def window_of(path):
    a = np.array(Image.open(path).convert("RGB"))
    m = ((a[:, :, 0] == BORDER[0]) & (a[:, :, 1] == BORDER[1]) & (a[:, :, 2] == BORDER[2]))
    ys, xs = np.where(m)
    return (int(xs.min()), int(ys.min())) if len(xs) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="main_table_logs")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "golden"))
    ap.add_argument("--steps-per-episode", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest = []
    for tag, run, digits, episodes in SOURCES:
        for ep in episodes:
            src = os.path.join(args.logs, run, f"episode_{ep}")
            if not os.path.isdir(src):
                raise SystemExit(f"missing {src}")
            stem = f"{tag}_ep{ep}"
            shutil.copy(os.path.join(src, "original.png"),
                        os.path.join(args.out, f"{stem}_canvas.png"))
            steps = sorted(
                (int(f[5:-4]) for f in os.listdir(src)
                 if f.startswith("step_") and f.endswith(".png")))
            chosen = steps[:args.steps_per_episode]
            obs = []
            for n in chosen:
                name = f"{stem}_step{n}.png"
                shutil.copy(os.path.join(src, f"step_{n}.png"),
                            os.path.join(args.out, name))
                obs.append({"step": n, "file": name,
                            "window": window_of(os.path.join(src, f"step_{n}.png"))})
            manifest.append({"tag": tag, "episode": ep, "digits": digits,
                             "canvas": f"{stem}_canvas.png", "box_size": 64,
                             "step_size": 32, "image_size": 224,
                             "source_run": run, "observations": obs})
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(manifest)} episodes to {args.out}")


if __name__ == "__main__":
    main()
