"""Procedural training scenes for NC-DCE.

The evaluation corpus is ``sensorsim.data.synthetic`` at seed 0. Training
scenes come from a separate generator with seeds starting at 1000, so no test
image (or a re-seeded copy of one) is ever seen in training.

The generator aims for variety rather than realism: flat and graded regions,
saturated objects, thin lines, text, high-frequency texture, and a smooth
random illumination field. The last one matters most: it puts deep shadows
next to well-lit areas inside one frame, which is the situation a curve
estimator has to handle and where noise amplification shows.
"""

from __future__ import annotations

import string

import cv2
import numpy as np

from sensorsim.data.synthetic import srgb_to_linear


def _colour(rng: np.random.Generator) -> tuple[float, float, float]:
    return tuple(float(v) for v in rng.uniform(0.05, 0.95, 3))


def training_scene(seed: int, size: int = 256) -> np.ndarray:
    """One linear-RGB scene in [0, 1], shape (size, size, 3)."""
    rng = np.random.default_rng(seed)
    h = w = size

    # Background: a linear blend between two colours along a random direction.
    c0, c1 = np.array(_colour(rng)), np.array(_colour(rng))
    theta = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[:h, :w].astype(np.float32) / size
    t = np.clip(np.cos(theta) * xx + np.sin(theta) * yy, 0, 1)[..., None]
    img = (c0 * (1 - t) + c1 * t).astype(np.float32)

    for _ in range(int(rng.integers(6, 16))):
        kind = rng.choice(["circle", "rect", "ellipse", "line", "poly"])
        col = _colour(rng)
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        if kind == "circle":
            cv2.circle(img, (cx, cy), int(rng.integers(4, size // 4)), col, -1, cv2.LINE_AA)
        elif kind == "rect":
            cv2.rectangle(img, (cx, cy), (cx + int(rng.integers(8, size // 3)), cy + int(rng.integers(8, size // 3))), col, -1)
        elif kind == "ellipse":
            axes = (int(rng.integers(4, size // 4)), int(rng.integers(4, size // 5)))
            cv2.ellipse(img, (cx, cy), axes, float(rng.uniform(0, 180)), 0, 360, col, -1, cv2.LINE_AA)
        elif kind == "line":
            end = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.line(img, (cx, cy), end, col, int(rng.integers(1, 4)), cv2.LINE_AA)
        else:
            pts = rng.integers(0, size, size=(int(rng.integers(3, 7)), 2)).astype(np.int32)
            cv2.fillPoly(img, [pts], col, cv2.LINE_AA)

    if rng.random() < 0.6:
        tex = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), float(rng.uniform(0.5, 2.0)))
        mask = np.zeros((h, w), np.float32)
        y0, x0 = int(rng.integers(0, h // 2)), int(rng.integers(0, w // 2))
        mask[y0:y0 + int(rng.integers(h // 6, h // 2)), x0:x0 + int(rng.integers(w // 6, w // 2))] = 1.0
        img += ((tex - tex.mean()) * float(rng.uniform(0.1, 0.4)) * mask)[..., None]

    if rng.random() < 0.4:
        canvas = (np.clip(img, 0, 1) * 255).astype(np.uint8).copy()
        for _ in range(int(rng.integers(1, 4))):
            text = "".join(rng.choice(list(string.ascii_letters + string.digits + "  "), int(rng.integers(5, 16))))
            ink = tuple(int(v) for v in rng.integers(0, 256, 3))
            cv2.putText(canvas, text, (int(rng.integers(0, w // 3)), int(rng.integers(16, h))),
                        cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.3, 0.9)), ink, 1, cv2.LINE_AA)
        img = canvas.astype(np.float32) / 255.0

    linear = srgb_to_linear(np.clip(img, 0.0, 1.0))

    # Smooth illumination: a 4x4 random grid, upsampled - shadows and highlights.
    grid = rng.uniform(0.06, 1.0, size=(4, 4)).astype(np.float32)
    light = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(linear * np.clip(light, 0.04, 1.0)[..., None], 0.0, 1.0).astype(np.float32)
