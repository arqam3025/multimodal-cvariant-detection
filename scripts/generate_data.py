"""Generate the shared synthetic dataset used by projects 2 and 3.

    python scripts/generate_data.py            # 800 campaigns, ~1-2 min on CPU
"""
import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fermsim import simulate_campaigns, image_series  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/campaigns.npz")
    a = ap.parse_args()

    t0 = time.time()
    c = simulate_campaigns(a.n, a.seed)
    imgs = image_series(c.daily_c_fraction, seed=a.seed + 1, size=32)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, sensors=c.sensors, c_fraction=c.c_fraction,
                        daily_c_fraction=c.daily_c_fraction, proteomics=c.proteomics,
                        proteomics_day=c.proteomics_day, quality=c.quality,
                        takeover=c.takeover, images=imgs)
    print(f"saved {out} in {time.time() - t0:.0f}s | takeover rate {c.takeover.mean():.2f} | "
          f"quality {c.quality.mean():.1f} +/- {c.quality.std():.1f}")
