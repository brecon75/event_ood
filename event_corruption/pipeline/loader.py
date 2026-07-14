"""
Dataset loader for the Gen1 preprocessed HDF5 format.

Expected on-disk structure per sequence directory:
    <seq_dir>/
        event_representations_v2/
            stacked_histogram_dt=50_nbins=10/
                event_representations.h5          key='data'  (N,20,240,304) uint8
                timestamps_us.npy                 (N, 2) int64  [t_start, t_end] µs
                objframe_idx_2_repr_idx.npy        (M,) int64
        labels_v2/
            labels.npz                             keys: 'labels', 'objframe_idx_2_label_idx'
            timestamps_us.npy                      (M,) int64

NOTE: We use PyTables (`tables`) instead of h5py for HDF5 reading.
      pip-distributed h5py wheels now bundle HDF5 2.0.0 which has a regression
      that breaks reading the old-style chunked datasets used in Gen1.
      PyTables bundles HDF5 1.14.x and reads these files correctly.
"""
import tables
import numpy as np
from pathlib import Path

H5_SUBPATH  = "event_representations_v2/stacked_histogram_dt=50_nbins=10"
EXPECTED_CHANNELS = 20   # 2 * nbins — fixed regardless of resolution
# Gen1: (20, 240, 304). Gen4 (downsample_by_factor_2, 720x1280 -> half res): (20, 360, 640).
H5_FILENAME_CANDIDATES = ("event_representations_ds2_nearest.h5", "event_representations.h5")


def load_histogram(seq_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the stacked-histogram tensor and its per-frame timestamps.

    Returns
    -------
    histogram  : (N, 20, H, W) uint8   — H,W depend on dataset (Gen1 240x304, Gen4 360x640)
    timestamps : (N, 2)         int64  — [t_start_us, t_end_us] per frame
    """
    seq_dir   = Path(seq_dir)
    repr_dir  = seq_dir / H5_SUBPATH
    ts_path   = repr_dir / "timestamps_us.npy"

    h5_path = None
    for candidate in H5_FILENAME_CANDIDATES:
        p = repr_dir / candidate
        if p.exists():
            h5_path = p
            break
    if h5_path is None:
        raise FileNotFoundError(
            f"HDF5 not found: tried {[str(repr_dir / c) for c in H5_FILENAME_CANDIDATES]}"
        )
    if not ts_path.exists():
        raise FileNotFoundError(f"Timestamps not found: {ts_path}")

    with tables.open_file(str(h5_path), mode="r") as f:
        if not hasattr(f.root, "data"):
            raise KeyError(f"Expected node 'data' in {h5_path}")
        ds        = f.root.data
        histogram = ds[:]           # 0.35s for full 1.62 GB sequence — fastest strategy

    if histogram.shape[1] != EXPECTED_CHANNELS:
        raise ValueError(
            f"Unexpected histogram shape {histogram.shape}; "
            f"expected (N, {EXPECTED_CHANNELS}, H, W)"
        )

    timestamps = np.load(ts_path)
    # Gen1 stores (N, 2) [t_start, t_end] windows. The RVT gen4 preprocessor
    # writes only the 1D (N,) end-timestamps, so synthesize the start column as
    # the previous frame's end (contiguous windows, matching Gen1 semantics).
    # The values are carried through as metadata only — none of the corruptions
    # read them — so the exact reconstruction of frame 0's start is immaterial.
    if timestamps.ndim == 1:
        ends   = timestamps.astype(np.int64)
        starts = np.empty_like(ends)
        starts[1:] = ends[:-1]
        starts[0]  = ends[0] - (ends[1] - ends[0] if len(ends) > 1 else 0)
        timestamps = np.stack([starts, ends], axis=1)
    assert timestamps.ndim == 2 and timestamps.shape[1] == 2, \
        f"timestamps_us must be (N, 2) or (N,), got {timestamps.shape}"
    assert len(timestamps) == len(histogram), \
        f"Frame count mismatch: histogram={len(histogram)}, timestamps={len(timestamps)}"

    return histogram, timestamps
