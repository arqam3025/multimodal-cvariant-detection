"""Small shared training utilities."""
from __future__ import annotations

import copy

import numpy as np
import torch


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit(model, train_loader, val_loader, loss_fn, epochs=80, lr=1e-3, patience=15, wd=1e-4):
    """Adam + early stopping on validation loss. Returns the best model."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = np.mean([loss_fn(model, b).item() for b in val_loader])
        if v < best - 1e-5:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def split_conformal(cal_residuals: np.ndarray, alpha: float = 0.1) -> float:
    """Half-width q such that [y_hat - q, y_hat + q] has >= 1-alpha coverage
    (finite-sample corrected quantile of absolute calibration residuals)."""
    n = len(cal_residuals)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(np.abs(cal_residuals), level, method="higher"))
