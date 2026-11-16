"""Project 3 - Multimodal cross-attention model for early C-variant detection.

After k days of a 30-day campaign, predict (a) whether the colonial C-variant
will take over the culture by day 30 and (b) final paste quality, by fusing
three modalities sampled at very different rates:

    process sensors  - 6 channels, 2-hourly (dense)
    micrographs      - one 32x32 hyphal image per day
    proteomics       - 16 proteins every 3 days, ~30 % of samples missing

Architecture
    sensor patches (12 h) -> linear  -> tokens  \
    daily images          -> CNN     -> tokens   +- modality + time embeddings
    proteomics samples    -> MLP     -> tokens  /
    1) cross-attention: sensor tokens query image/proteomics tokens
    2) transformer fusion over [CLS | sensors | images | proteomics]
       with key-padding masks for missing samples
    3) heads: takeover logit, quality regression

Modality dropout during training (whole image or proteomics stream removed at
random) teaches the model to degrade gracefully when a modality is missing.

Experiments
    * ablation: sensors only vs. sensors+images vs. all three (3 seeds)
    * robustness: full model tested with modalities removed
    * early detection: AUROC vs. day of prediction (k = 6, 10, 14)
    * cross-attention weights: which days' images the model attends to

    python scripts/generate_data.py
    python p3_multimodal_cvariant/train.py           # ~20-40 min on CPU
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from fermsim import chronological_split  # noqa: E402
from fermsim.training import fit, set_seed  # noqa: E402

OUT = pathlib.Path(__file__).parent / "results"
STEP_H, PATCH = 2, 6
D_MODEL = 64
torch.set_num_threads(8)


# ----------------------------------------------------------------------------- model
class MultimodalNet(nn.Module):
    def __init__(self, n_sensor=6, n_prot=16, d=D_MODEL, modalities=("sensor", "image", "prot"),
                 modality_dropout=0.3):
        super().__init__()
        self.modalities, self.p_drop = set(modalities), modality_dropout
        self.sensor_enc = nn.Linear(n_sensor * PATCH, d)
        self.image_enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, d))
        self.prot_enc = nn.Sequential(nn.Linear(n_prot, d), nn.GELU(), nn.Linear(d, d))
        self.time_enc = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.mod_emb = nn.Parameter(torch.randn(3, d) * 0.02)
        self.null = nn.Parameter(torch.zeros(1, 1, d))      # always-present memory token
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.cross = nn.MultiheadAttention(d, 4, batch_first=True, dropout=0.1)
        self.cross_norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head_takeover = nn.Linear(d, 1)
        self.head_quality = nn.Linear(d, 1)

    def forward(self, sens, imgs, prot, prot_mask, drop=None):
        """sens (B,T,6); imgs (B,Dy,32,32); prot (B,P,16) zero-filled; prot_mask (B,P) True = missing.
        drop: optional set of modality names to remove (used for robustness tests)."""
        B, T, C = sens.shape
        days, P = imgs.shape[1], prot.shape[1]
        drop = set(drop or ())
        drop |= {"image", "prot"} - self.modalities
        img_missing = torch.zeros(B, dtype=torch.bool)
        prot_missing_all = torch.zeros(B, dtype=torch.bool)
        if self.training and self.p_drop > 0:              # modality dropout
            img_missing |= torch.rand(B) < self.p_drop
            prot_missing_all |= torch.rand(B) < self.p_drop
        if "image" in drop:
            img_missing[:] = True
        if "prot" in drop:
            prot_missing_all[:] = True

        hours_per_patch = PATCH * STEP_H
        t_s = (torch.arange(T // PATCH) + 1.0) * hours_per_patch
        t_i = (torch.arange(days) + 1.0) * 24
        t_p = (torch.arange(P) + 1.0) * 72
        te = lambda t: self.time_enc((t / 720.0)[:, None])   # noqa: E731  (time in campaign fraction)

        s_tok = self.sensor_enc(sens.reshape(B, T // PATCH, PATCH * C)) + te(t_s) + self.mod_emb[0]
        i_tok = self.image_enc(imgs.reshape(B * days, 1, *imgs.shape[2:])).reshape(B, days, -1) + te(t_i) + self.mod_emb[1]
        p_tok = self.prot_enc(prot) + te(t_p) + self.mod_emb[2]

        i_mask = img_missing[:, None].expand(B, days)
        p_mask = prot_mask | prot_missing_all[:, None]
        mem = torch.cat([self.null.expand(B, -1, -1), i_tok, p_tok], 1)
        mem_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool), i_mask, p_mask], 1)

        attn_out, attn_w = self.cross(s_tok, mem, mem, key_padding_mask=mem_mask, need_weights=True)
        s_tok = self.cross_norm(s_tok + attn_out)
        self.last_cross_attn = attn_w.detach()               # (B, n_sensor_tokens, 1+days+P)

        z = torch.cat([self.cls.expand(B, -1, -1), s_tok, i_tok, p_tok], 1)
        z_mask = torch.cat([torch.zeros(B, 1 + s_tok.shape[1], dtype=torch.bool), i_mask, p_mask], 1)
        h = self.norm(self.fusion(z, src_key_padding_mask=z_mask)[:, 0])
        return self.head_takeover(h).squeeze(-1), self.head_quality(h).squeeze(-1)


# ----------------------------------------------------------------------------- data
def load(day):
    d = np.load(ROOT / "data" / "campaigns.npz")
    s = d["sensors"]
    n, H, c = s.shape
    s = s.reshape(n, H // STEP_H, STEP_H, c).mean(2)[:, : day * 24 // STEP_H]
    imgs = d["images"][:, :day]
    n_p = int((d["proteomics_day"] <= day).sum())
    prot = d["proteomics"][:, :n_p]
    return dict(sens=s.astype(np.float32), imgs=imgs.astype(np.float32), prot=prot.astype(np.float32),
                takeover=d["takeover"].astype(np.float32), quality=d["quality"].astype(np.float32))


def tensors(data, idx, stats):
    mu, sd, pm, ps, qm, qs = stats
    prot = data["prot"][idx]
    mask = np.isnan(prot).any(-1)
    prot = np.nan_to_num((prot - pm) / ps)
    return (torch.tensor((data["sens"][idx] - mu) / sd), torch.tensor(data["imgs"][idx]),
            torch.tensor(prot, dtype=torch.float32), torch.tensor(mask),
            torch.tensor(data["takeover"][idx]), torch.tensor((data["quality"][idx] - qm) / qs))


def loss_fn(m, b):
    logit, q = m(*b[:4])
    return F.binary_cross_entropy_with_logits(logit, b[4]) + 0.5 * F.mse_loss(q, b[5])


def evaluate(m, tens, stats, drop=None):
    m.eval()
    with torch.no_grad():
        logit, q = m(*tens[:4], drop=drop)
    y = tens[4].numpy()
    qm, qs = stats[4], stats[5]
    mae = float(np.mean(np.abs(q.numpy() * qs + qm - (tens[5].numpy() * qs + qm))))
    return dict(auroc=float(roc_auc_score(y, logit.numpy())), quality_mae=mae)


def train_config(day, modalities, seed):
    data = load(day)
    n = len(data["quality"])
    tr, val, te = chronological_split(n, 0.6, 0.2)
    mu, sd = data["sens"][tr].mean((0, 1)), data["sens"][tr].std((0, 1)) + 1e-6
    pm, ps = np.nanmean(data["prot"][tr], (0, 1)), np.nanstd(data["prot"][tr], (0, 1)) + 1e-6
    qm, qs = data["quality"][tr].mean(), data["quality"][tr].std()
    stats = (mu, sd, pm, ps, qm, qs)
    T = {k: tensors(data, idx, stats) for k, idx in [("tr", tr), ("val", val), ("te", te)]}
    set_seed(seed)
    model = MultimodalNet(modalities=modalities, modality_dropout=0.3 if len(modalities) > 1 else 0.0)
    model = fit(model, DataLoader(TensorDataset(*T["tr"]), batch_size=32, shuffle=True),
                DataLoader(TensorDataset(*T["val"]), batch_size=64), loss_fn, epochs=100, lr=1e-3, patience=15)
    return model, T, stats


VARIANTS = {"Sensors only": ("sensor",), "Sensors + images": ("sensor", "image"),
            "Sensors + images + proteomics": ("sensor", "image", "prot")}


def main():
    OUT.mkdir(exist_ok=True)
    seeds = (0, 1, 2)
    results = {"ablation_day10": {}, "robustness_day10": {}, "early_detection": {}}

    # 1) ablation at day 10 (+ robustness of the full model)
    full_models = []
    for name, mods in VARIANTS.items():
        runs = []
        for s in seeds:
            m, T, stats = train_config(10, mods, s)
            runs.append(evaluate(m, T["te"], stats))
            if len(mods) == 3:
                full_models.append((m, T, stats))
        results["ablation_day10"][name] = {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))]
                                           for k in runs[0]}
        print("day 10 |", name, results["ablation_day10"][name], flush=True)

    for label, drop in [("all modalities", None), ("images missing", {"image"}),
                        ("proteomics missing", {"prot"}), ("images + proteomics missing", {"image", "prot"})]:
        rs = [evaluate(m, T["te"], st, drop=drop) for m, T, st in full_models]
        results["robustness_day10"][label] = {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))]
                                              for k in rs[0]}
        print("robustness |", label, results["robustness_day10"][label], flush=True)

    # 2) early detection curve
    for name in ["Sensors only", "Sensors + images + proteomics"]:
        results["early_detection"][name] = {}
        for day in (6, 10, 14):
            if day == 10:
                results["early_detection"][name][day] = results["ablation_day10"][name]["auroc"]
                continue
            aucs = []
            for s in seeds:
                m, T, st = train_config(day, VARIANTS[name], s)
                aucs.append(evaluate(m, T["te"], st)["auroc"])
            results["early_detection"][name][day] = [float(np.mean(aucs)), float(np.std(aucs))]
            print("early |", name, day, results["early_detection"][name][day], flush=True)

    # 3) cross-attention: attention from sensor tokens to each day's image
    m, T, st = full_models[0]
    m.eval()
    with torch.no_grad():
        m(*T["te"][:4])
    w = m.last_cross_attn.mean(1).numpy()[:, 1:11]            # (B, 10 image days)
    y = T["te"][4].numpy().astype(bool)

    # --- figures -----------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    names = list(results["ablation_day10"])
    auc = [results["ablation_day10"][k]["auroc"] for k in names]
    ax[0].bar(range(3), [a[0] for a in auc], yerr=[a[1] for a in auc], capsize=4,
              color=["tab:gray", "tab:orange", "tab:blue"])
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["sensors", "+ images", "+ proteomics"])
    ax[0].set_ylim(0.5, 1.0); ax[0].set_ylabel("test AUROC (takeover by day 30)")
    ax[0].set_title("Day-10 prediction: modality ablation")

    for name, c in [("Sensors only", "tab:gray"), ("Sensors + images + proteomics", "tab:blue")]:
        ds = [6, 10, 14]
        v = [results["early_detection"][name][d] for d in ds]
        ax[1].errorbar(ds, [a[0] for a in v], yerr=[a[1] for a in v], marker="o", capsize=3, color=c, label=name)
    ax[1].set_xlabel("day of prediction"); ax[1].set_ylabel("test AUROC"); ax[1].legend(fontsize=8)
    ax[1].set_title("Early detection of C-variant takeover")

    days = np.arange(1, 11)
    ax[2].plot(days, w[y].mean(0), marker="o", color="tab:red", label="campaigns that take over")
    ax[2].plot(days, w[~y].mean(0), marker="o", color="tab:green", label="campaigns that do not")
    ax[2].set_xlabel("image day"); ax[2].set_ylabel("mean cross-attention weight")
    ax[2].set_title("Where sensor tokens look in the image stream"); ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "multimodal_results.png", dpi=150)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
