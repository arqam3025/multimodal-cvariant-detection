"""Synthetic continuous-fermentation campaign simulator.

A deliberately simple, *mechanistic* model of a continuous (chemostat) fungal
fermentation, used to generate realistic-looking multimodal data for the
projects in this repository. It is NOT a model of any real plant: all
parameters are illustrative and the data are synthetic.

State per campaign (vectorised over N campaigns):
    W  wild-type biomass            [g/L]
    C  colonial "C-variant" biomass [g/L]
    S  limiting substrate (glucose) [g/L]

Dynamics (Monod kinetics in a chemostat, dilution rate D):
    mu_W = mu_max_W * f_T(T)   * S / (Ks + S)
    mu_C = mu_max_C * f_T_C(T) * S / (Ks + S)
    dW/dt = (mu_W - D) W - m mu_W W          (m = mutation rate W -> C)
    dC/dt = (mu_C - D) C + m mu_W W
    dS/dt = D (S_in - S) - (mu_W W + mu_C C) / Y

The C-variant has a small growth advantage that depends on operating
conditions (dilution rate, temperature excursions) and on slow plant drift
across the campaign index, so whether it takes over within 30 days varies
from campaign to campaign - the "C-variant problem".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOURS = 30 * 24          # 30-day campaign
DT = 0.1                 # integration step [h]
SENSOR_NAMES = ["biomass", "dissolved_O2", "temperature", "CO2_offgas", "pH", "dilution_rate"]
N_PROTEINS = 16


@dataclass
class Campaigns:
    sensors: np.ndarray        # (N, HOURS, 6) hourly noisy sensor readings
    c_fraction: np.ndarray     # (N, HOURS) true C-variant fraction (hidden state)
    daily_c_fraction: np.ndarray  # (N, 30) fraction at the end of each day
    proteomics: np.ndarray     # (N, 10, N_PROTEINS) every 3 days, NaN where missing
    proteomics_day: np.ndarray # (10,) sampling day of each proteomics sample
    quality: np.ndarray        # (N,) final paste quality score (0-100)
    takeover: np.ndarray       # (N,) 1 if C-fraction > 0.5 at day 30
    params: dict               # per-campaign hidden parameters


def _temperature_profile(rng, n, hours):
    """Set point 30 C + campaign offset + a few random excursion events."""
    t = np.arange(hours)
    T = 30.0 + rng.normal(0, 0.4, size=(n, 1)) + np.zeros((n, hours))
    n_events = rng.poisson(2.0, size=n)
    for i in range(n):
        for _ in range(n_events[i]):
            start = rng.integers(0, hours - 48)
            dur = rng.integers(6, 48)
            amp = rng.normal(0, 1.5)
            T[i, start:start + dur] += amp * np.hanning(dur)
    T += 0.05 * np.sin(2 * np.pi * t / 24.0)  # small diurnal effect
    return T


def simulate_campaigns(n: int = 800, seed: int = 0) -> Campaigns:
    rng = np.random.default_rng(seed)
    idx = np.arange(n) / max(n - 1, 1)        # chronological position 0..1

    # --- hidden per-campaign parameters -----------------------------------
    mu_max_W = rng.normal(0.25, 0.01, n)      # 1/h
    Ks = rng.normal(0.10, 0.01, n)            # g/L
    Y = rng.normal(0.50, 0.02, n)             # g biomass / g substrate
    S_in = rng.normal(20.0, 1.0, n)           # g/L
    D_base = rng.uniform(0.10, 0.17, n)       # 1/h
    strain_adv = rng.normal(0.03, 0.10, n)    # intrinsic advantage of variant
    drift = 0.08 * idx                        # slow plant drift (ageing seed train)
    mutation = 10 ** rng.uniform(-6, -4.5, n)
    kLa = rng.normal(120.0, 15.0, n)          # 1/h

    T = _temperature_profile(rng, n, HOURS)
    # Dilution-rate schedule: base level plus occasional step changes
    D = np.repeat(D_base[:, None], HOURS, axis=1)
    for i in range(n):
        for _ in range(rng.poisson(1.0)):
            start = rng.integers(48, HOURS - 48)
            D[i, start:] += rng.normal(0, 0.01)
    D = np.clip(D, 0.08, 0.19)

    # --- integrate ---------------------------------------------------------
    W = np.full(n, 8.0)
    C = W * 1e-5
    S = np.full(n, 0.2)
    steps_per_hour = int(round(1 / DT))
    W_h = np.zeros((n, HOURS)); C_h = np.zeros((n, HOURS)); S_h = np.zeros((n, HOURS))
    growth_h = np.zeros((n, HOURS))
    for h in range(HOURS):
        Th, Dh = T[:, h], D[:, h]
        fT_W = np.exp(-((Th - 30.0) / 4.0) ** 2)
        fT_C = np.exp(-((Th - 30.0) / 7.0) ** 2)      # variant is more heat tolerant
        # variant advantage grows with dilution rate and plant drift
        adv = 1.0 + strain_adv + drift + 2.0 * (Dh - 0.135)
        mu_max_C = mu_max_W * adv
        for _ in range(steps_per_hour):
            monod = S / (Ks + S)
            mu_W = mu_max_W * fT_W * monod
            mu_C = mu_max_C * fT_C * monod
            dW = (mu_W - Dh) * W - mutation * mu_W * W
            dC = (mu_C - Dh) * C + mutation * mu_W * W
            dS = Dh * (S_in - S) - (mu_W * W + mu_C * C) / Y
            W = np.maximum(W + DT * dW, 1e-6)
            C = np.maximum(C + DT * dC, 1e-9)
            S = np.maximum(S + DT * dS, 1e-6)
        W_h[:, h], C_h[:, h], S_h[:, h] = W, C, S
        growth_h[:, h] = mu_W * W + mu_C * C

    X = W_h + C_h
    frac = C_h / X

    # --- sensors (hourly, noisy) ------------------------------------------
    OUR = 0.6 * growth_h                               # oxygen uptake (arb. units)
    DO = np.clip(7.0 - 60.0 * OUR / kLa[:, None], 0.0, None)
    CO2 = 0.9 * growth_h
    pH = 6.0 - 0.15 * (growth_h - growth_h.mean()) + 0.1 * frac
    sensors = np.stack([
        X + rng.normal(0, 0.15, X.shape),
        DO + rng.normal(0, 0.10, X.shape),
        T + rng.normal(0, 0.05, X.shape),
        CO2 + rng.normal(0, 0.03, X.shape),
        pH + rng.normal(0, 0.01, X.shape),
        D + rng.normal(0, 0.001, X.shape),
    ], axis=-1).astype(np.float32)

    daily_frac = frac[:, 23::24]                       # (n, 30)

    # --- sparse proteomics: every 3 days, ~30 % of samples missing ---------
    prot_days = np.arange(3, 31, 3)                    # days 3,6,...,30
    W_prot = rng.normal(0, 1, size=(5, N_PROTEINS))    # fixed loading matrix
    prot = np.zeros((n, len(prot_days), N_PROTEINS), dtype=np.float32)
    for j, d in enumerate(prot_days):
        h = d * 24 - 1
        latent = np.stack([
            (growth_h[:, h] - growth_h.mean()) / growth_h.std(),
            frac[:, h] * 5.0,
            T[:, max(0, h - 48):h + 1].std(axis=1),          # recent heat stress
            (DO[:, h] < 1.0).astype(float),
            np.ones(n),
        ], axis=1)
        prot[:, j] = latent @ W_prot + rng.normal(0, 0.5, (n, N_PROTEINS))
    missing = rng.random((n, len(prot_days))) < 0.3
    prot[missing] = np.nan

    # --- final quality ------------------------------------------------------
    late_frac = frac[:, -5 * 24:].mean(axis=1)
    protein_content = (D[:, -5 * 24:].mean(axis=1) - 0.08) / 0.11   # 0..1
    o2_limited = (DO[:, -5 * 24:] < 0.5).mean(axis=1)
    quality = 100 * (0.70 * (1 - late_frac) + 0.25 * protein_content) - 15 * o2_limited
    quality = quality + rng.normal(0, 2.0, n)

    return Campaigns(
        sensors=sensors,
        c_fraction=frac.astype(np.float32),
        daily_c_fraction=daily_frac.astype(np.float32),
        proteomics=prot,
        proteomics_day=prot_days,
        quality=quality.astype(np.float32),
        takeover=(daily_frac[:, -1] > 0.5).astype(np.int64),
        params=dict(mu_max_W=mu_max_W, Ks=Ks, Y=Y, S_in=S_in, D_base=D_base,
                    strain_adv=strain_adv, drift=drift, mutation=mutation, kLa=kLa),
    )


def chronological_split(n: int, train: float = 0.6, cal: float = 0.2):
    """Split campaign indices in time order: train | calibration/val | test."""
    a, b = int(n * train), int(n * (train + cal))
    idx = np.arange(n)
    return idx[:a], idx[a:b], idx[b:]
