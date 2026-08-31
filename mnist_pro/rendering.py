"""Canvas construction and observation rendering.

Pipeline, in order:
    1. horizontally concatenate `digits` MNIST images (28x28 each)
    2. invert, so dark strokes sit on a light ground
    3. bilinear resize to (image_size * digits, image_size)
    4. threshold at 200: anything brighter becomes 255, everything else 0

The result is a two-valued canvas: strokes at 0, background at 255.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageOps

# --- rendering constants ------------------------------------------------------
MASK_RGB = (128, 128, 128)  # unrevealed region
BORDER_HEX = "#00E5FF"  # glimpse-window outline
BORDER_RGB = (0, 229, 255)
BORDER_WIDTH = 2
STROKE_VALUE = 0  # dark stroke
BACKGROUND_VALUE = 255  # light ground
BINARISE_THRESHOLD = 200


@dataclass(frozen=True)
class CanvasSpec:
    """Geometry of one episode's canvas."""

    digits: int = 1
    image_size: int = 224
    box_size: int = 64
    step_size: int = 32

    @property
    def width(self) -> int:
        return self.image_size * self.digits

    @property
    def height(self) -> int:
        return self.image_size

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


def build_canvas(images, spec: CanvasSpec) -> np.ndarray:
    """Concatenate, invert, resize and binarise. Returns a uint8 (H, W) array."""
    if not images:
        raise ValueError("images must not be empty")
    if len(images) != spec.digits:
        raise ValueError(f"spec.digits={spec.digits} but {len(images)} images given")

    total_width = sum(im.width for im in images)
    max_height = max(im.height for im in images)
    combined = Image.new(images[0].mode, (total_width, max_height))
    x = 0
    for im in images:
        combined.paste(im, (x, 0))
        x += im.width

    flipped = ImageOps.invert(combined.convert("RGB")).convert("L")
    resized = flipped.resize((spec.width, spec.height), Image.Resampling.BILINEAR)
    binarised = resized.point(lambda p: 255 if p > BINARISE_THRESHOLD else 0)
    return np.array(binarised)


def deterministic_start(canvas: np.ndarray, spec: CanvasSpec) -> tuple[int, int]:
    """Calculate a deterministic start position centered on a randomly chosen stroke pixel.

    Seeds the RNG with the MD5 hash of the canvas to ensure reproducibility.
    """
    ys, xs = np.where(canvas == STROKE_VALUE)
    if len(ys) == 0:
        return ((spec.width - spec.box_size) // 2, (spec.height - spec.box_size) // 2)
    seed = int(hashlib.md5(canvas.tobytes()).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed)
    return _centre_on(canvas, spec, rng)


def seeded_start(canvas: np.ndarray, spec: CanvasSpec, seed: int) -> tuple[int, int]:
    """A start position drawn from an explicit seed rather than the canvas hash."""
    ys, xs = np.where(canvas == STROKE_VALUE)
    if len(ys) == 0:
        return ((spec.width - spec.box_size) // 2, (spec.height - spec.box_size) // 2)
    return _centre_on(canvas, spec, np.random.RandomState(seed % (2**32)))


def _centre_on(canvas, spec, rng):
    ys, xs = np.where(canvas == STROKE_VALUE)
    i = rng.randint(len(ys))
    cy, cx = ys[i], xs[i]
    return (
        int(max(0, min(spec.width - spec.box_size, cx - spec.box_size // 2))),
        int(max(0, min(spec.height - spec.box_size, cy - spec.box_size // 2))),
    )


def render_observation(
    canvas: np.ndarray, x: int, y: int, spec: CanvasSpec, draw_border: bool = True
) -> Image.Image:
    """One glimpse: grey everywhere, the window pasted sharp, cyan outline on top.

    When `draw_border` is True, the outline is drawn over the window, occluding the outer
    two pixels of the glimpse.
    """
    img = Image.fromarray(canvas).convert("RGB")
    out = Image.new("RGB", (spec.width, spec.height), MASK_RGB)
    out.paste(img.crop((x, y, x + spec.box_size, y + spec.box_size)), (x, y))
    if draw_border:
        ImageDraw.Draw(out).rectangle(
            [x, y, x + spec.box_size, y + spec.box_size],
            outline=BORDER_HEX,
            width=BORDER_WIDTH,
        )
    return out


def render_composite(
    canvas: np.ndarray, windows, spec: CanvasSpec, draw_border: bool = False
) -> Image.Image:
    """The same renderer with more than one window revealed, for offline replay."""
    img = Image.fromarray(canvas).convert("RGB")
    out = Image.new("RGB", (spec.width, spec.height), MASK_RGB)
    for x, y in windows:
        out.paste(img.crop((x, y, x + spec.box_size, y + spec.box_size)), (x, y))
    if draw_border:
        draw = ImageDraw.Draw(out)
        for x, y in windows:
            draw.rectangle(
                [x, y, x + spec.box_size, y + spec.box_size],
                outline=BORDER_HEX,
                width=BORDER_WIDTH,
            )
    return out
