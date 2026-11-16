"""Synthetic micrographs of fungal hyphae.

Wild-type hyphae are drawn as long, sparsely branched filaments; colonial
C-variants as short, densely branched clusters. An image of a culture with
C-variant fraction f mixes the two morphologies in proportion. The images are
crude and synthetic - they exist to give the models a realistic *imaging
modality* to learn from, not to imitate real microscopy.
"""
from __future__ import annotations

import numpy as np


def _draw_filament(img, rng, x, y, angle, length, branch_p, depth=0):
    size = img.shape[0]
    for _ in range(length):
        xi, yi = int(x) % size, int(y) % size
        img[yi, xi] = 1.0
        angle += rng.normal(0, 0.25)
        x += np.cos(angle)
        y += np.sin(angle)
        if depth < 3 and rng.random() < branch_p:
            _draw_filament(img, rng, x, y, angle + rng.choice([-1, 1]) * rng.uniform(0.6, 1.2),
                           max(2, length // 2), branch_p, depth + 1)
            length = max(1, length - 1)


def hyphae_image(c_fraction: float, rng: np.random.Generator, size: int = 32) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.float32)
    n_fil = max(3, int(size / 5))
    n_colonial = rng.binomial(n_fil, float(np.clip(c_fraction, 0, 1)))
    scale = size / 32
    for k in range(n_fil):
        x, y = rng.uniform(0, size, 2)
        angle = rng.uniform(0, 2 * np.pi)
        if k < n_colonial:   # colonial variant: short and highly branched
            _draw_filament(img, rng, x, y, angle, int(5 * scale), branch_p=0.45)
        else:                # wild type: long, sparsely branched
            _draw_filament(img, rng, x, y, angle, int(22 * scale), branch_p=0.04)
    img += rng.normal(0, 0.15, img.shape).astype(np.float32)   # sensor noise
    return np.clip(img, 0, 1)


def image_series(daily_fraction: np.ndarray, seed: int = 0, size: int = 32) -> np.ndarray:
    """daily_fraction (N, days) -> images (N, days, size, size)."""
    rng = np.random.default_rng(seed)
    n, days = daily_fraction.shape
    out = np.zeros((n, days, size, size), dtype=np.float32)
    for i in range(n):
        for d in range(days):
            out[i, d] = hyphae_image(daily_fraction[i, d], rng, size)
    return out
