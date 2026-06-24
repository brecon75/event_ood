"""
Qualitative visualization: how an event representation looks BEFORE vs AFTER each
corruption.

Renders a Gen1 event-histogram frame (20 channels = ON bins 0-9, OFF bins 10-19)
as a 2-colour event image (ON = red, OFF = blue) and lays out a gallery:
one row per corruption, columns = [clean, severity 1, 3, 5]. This is the input-side
qualitative figure for the paper (the cluster produces only quantitative numbers).

Data source, in priority order:
  1. --sample PATH : a saved histogram (.npy/.pt) of shape (20,H,W) or (N,20,H,W)
  2. a cached sample auto-found under outputs/ (first (.,20,H,W) array)
  3. a synthetic structured event frame (moving-edge contours) for layout dev when
     no Gen1 sample is available locally.

Usage:
    python analysis/viz_corruptions.py                 # auto / synthetic
    python analysis/viz_corruptions.py --sample path/to/frame.npy
    python analysis/viz_corruptions.py --severities 1 3 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "event_corruption"))
from vmem_benchmark import benchmark_config as cfg          # noqa: E402
from corrupt.registry import apply_corruption, CORRUPTIONS    # noqa: E402

H, W = 240, 304


def _synthetic_frame(seed=0) -> np.ndarray:
    """A plausible event histogram (20,H,W): events concentrate on moving contours.
    Only for local layout dev; replace with a real Gen1 sample via --sample."""
    rng = np.random.default_rng(seed)
    hist = np.zeros((20, H, W), dtype=np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(4):  # a few "objects" whose edges fire
        cy, cx = rng.integers(40, H - 40), rng.integers(40, W - 40)
        r = rng.integers(20, 50)
        ring = np.abs(np.hypot(yy - cy, xx - cx) - r) < 1.5
        on = ring & (xx > cx)   # leading edge -> ON
        off = ring & (xx <= cx)  # trailing edge -> OFF
        for b in range(10):
            hist[b][on] = rng.integers(3, 25, size=on.sum()).astype(np.uint8)
            hist[10 + b][off] = rng.integers(3, 25, size=off.sum()).astype(np.uint8)
    return hist


def _load_sample(path: str | None) -> np.ndarray:
    if path:
        p = Path(path)
        if p.suffix == ".npy":
            arr = np.load(p)
        else:
            import torch
            obj = torch.load(p, map_location="cpu", weights_only=False)
            arr = obj.numpy() if hasattr(obj, "numpy") else np.asarray(obj)
        arr = np.asarray(arr)
        if arr.ndim == 4:  # (N,20,H,W) -> pick the busiest frame
            arr = arr[np.argmax(arr.reshape(arr.shape[0], -1).sum(1))]
        assert arr.ndim == 3 and arr.shape[0] == 20, f"expected (20,H,W), got {arr.shape}"
        return np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    print("[viz] no --sample given; using a SYNTHETIC frame (layout only).")
    return _synthetic_frame()


def _to_rgb(hist: np.ndarray) -> np.ndarray:
    """(20,H,W) -> (H,W,3): ON counts -> red, OFF counts -> blue, log-scaled."""
    on = hist[:10].astype(np.float32).sum(0)
    off = hist[10:].astype(np.float32).sum(0)
    def norm(x):
        x = np.log1p(x)
        m = x.max()
        return x / m if m > 0 else x
    rgb = np.zeros((*on.shape, 3), dtype=np.float32)
    rgb[..., 0] = norm(on)    # red = ON
    rgb[..., 2] = norm(off)   # blue = OFF
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=None, help="path to a (20,H,W)/(N,20,H,W) histogram")
    ap.add_argument("--severities", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = _load_sample(args.sample)
    corruptions = list(CORRUPTIONS)
    cols = ["clean"] + [f"sev {s}" for s in args.severities]

    cfg.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(corruptions), len(cols),
                             figsize=(2.6 * len(cols), 2.4 * len(corruptions)))
    axes = np.atleast_2d(axes)
    for ri, c in enumerate(corruptions):
        for ci, col in enumerate(cols):
            ax = axes[ri, ci]
            if col == "clean":
                img = _to_rgb(base)
            else:
                sev = args.severities[ci - 1]
                rng = np.random.default_rng([args.seed, ri, sev])
                corr = apply_corruption(base[None], None, c, sev, rng)[0]
                img = _to_rgb(corr)
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(c, fontsize=9)
            if ri == 0:
                ax.set_title(col, fontsize=9)
    fig.suptitle("Event representation before/after corruption  (ON=red, OFF=blue)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    for ext in ("pdf", "png"):
        out = cfg.PLOT_DIR / f"corruption_gallery.{ext}"
        plt.savefig(out, dpi=140)
        print(f"  Saved {out}")
    plt.close()


if __name__ == "__main__":
    main()
