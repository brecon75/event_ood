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
LABELS_DIR  = "labels_v2"
EXPECTED_SHAPE = (20, 240, 304)   # (C, H, W) — frame-count N varies


def load_histogram(seq_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the stacked-histogram tensor and its per-frame timestamps.

    Returns
    -------
    histogram  : (N, 20, 240, 304) uint8
    timestamps : (N, 2)            int64  — [t_start_us, t_end_us] per frame
    """
    seq_dir   = Path(seq_dir)
    h5_path   = seq_dir / H5_SUBPATH / "event_representations.h5"
    ts_path   = seq_dir / H5_SUBPATH / "timestamps_us.npy"

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {h5_path}")
    if not ts_path.exists():
        raise FileNotFoundError(f"Timestamps not found: {ts_path}")

    with tables.open_file(str(h5_path), mode="r") as f:
        if not hasattr(f.root, "data"):
            raise KeyError(f"Expected node 'data' in {h5_path}")
        ds        = f.root.data
        histogram = ds[:]           # 0.35s for full 1.62 GB sequence — fastest strategy

    if histogram.shape[1:] != EXPECTED_SHAPE:
        raise ValueError(
            f"Unexpected histogram shape {histogram.shape}; "
            f"expected (N, {EXPECTED_SHAPE[0]}, {EXPECTED_SHAPE[1]}, {EXPECTED_SHAPE[2]})"
        )

    timestamps = np.load(ts_path)
    assert timestamps.ndim == 2 and timestamps.shape[1] == 2, \
        f"timestamps_us must be (N, 2), got {timestamps.shape}"
    assert len(timestamps) == len(histogram), \
        f"Frame count mismatch: histogram={len(histogram)}, timestamps={len(timestamps)}"

    return histogram, timestamps


def load_labels(seq_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load bounding box labels and the objframe→label index.

    Returns
    -------
    labels    : structured array, dtype [('t','u8'),('x','f4'),('y','f4'),
                ('w','f4'),('h','f4'),('class_id','u1'),
                ('class_confidence','f4'),('track_id','u4')]
    label_idx : (M,) int64 — objframe_idx_2_label_idx
    """
    seq_dir  = Path(seq_dir)
    npz_path = seq_dir / LABELS_DIR / "labels.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"Labels not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    required = {"labels", "objframe_idx_2_label_idx"}
    missing  = required - set(data.keys())
    if missing:
        raise KeyError(f"labels.npz missing keys: {missing}")

    return data["labels"], data["objframe_idx_2_label_idx"]


def load_repr_index(seq_dir: Path) -> np.ndarray:
    """
    Load the objframe_idx → repr_idx mapping.

    Returns
    -------
    repr_idx : (M,) int64
    """
    path = Path(seq_dir) / H5_SUBPATH / "objframe_idx_2_repr_idx.npy"
    if not path.exists():
        raise FileNotFoundError(f"repr index not found: {path}")
    return np.load(path)


def load_label_timestamps(seq_dir: Path) -> np.ndarray:
    """
    Load per-objframe label timestamps (µs).

    Returns
    -------
    ts : (M,) int64
    """
    path = Path(seq_dir) / LABELS_DIR / "timestamps_us.npy"
    if not path.exists():
        raise FileNotFoundError(f"Label timestamps not found: {path}")
    return np.load(path)
