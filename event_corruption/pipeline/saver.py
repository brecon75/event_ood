"""
Save corrupted data to disk, mirroring the Gen1 on-disk structure.

Output layout:
    <out_dir>/
        event_representations_v2/
            stacked_histogram_dt=50_nbins=10/
                event_representations.h5          (corrupted)
                timestamps_us.npy                 (unchanged copy)
                objframe_idx_2_repr_idx.npy        (unchanged copy)
        labels_v2/
            labels.npz                             (unchanged copy)
            timestamps_us.npy                      (unchanged copy)

Labels and index arrays are NEVER modified.
"""
import tables
import numpy as np
from pathlib import Path

H5_SUBPATH = "event_representations_v2/stacked_histogram_dt=50_nbins=10"
LABELS_DIR = "labels_v2"


def save_corrupted(
    histogram      : np.ndarray,   # (N, 20, 240, 304) uint8  — corrupted
    timestamps     : np.ndarray,   # (N, 2)            int64  — unchanged
    labels         : np.ndarray,   # structured array          — unchanged
    label_idx      : np.ndarray,   # (M,) int64                — unchanged
    repr_idx       : np.ndarray,   # (M,) int64                — unchanged
    label_timestamps: np.ndarray,  # (M,) int64                — unchanged
    out_dir        : Path,
) -> None:
    """
    Write all data to `out_dir`, preserving the Gen1 directory layout.

    The HDF5 file is written with per-frame chunking and gzip-1 compression
    to keep output manageable (~30 × 470 sequences ≈ 52 GB uncompressed).
    """
    out_dir = Path(out_dir)

    # --- HDF5 + timestamps ---
    h5_dir = out_dir / H5_SUBPATH
    h5_dir.mkdir(parents=True, exist_ok=True)

    h5_out = h5_dir / "event_representations.h5"
    atom      = tables.UInt8Atom()
    filters   = tables.Filters(complevel=1, complib="blosc")
    with tables.open_file(str(h5_out), mode="w") as f:
        ea = f.create_earray(
            f.root, "data",
            atom=atom,
            shape=(0, *histogram.shape[1:]),
            expectedrows=histogram.shape[0],
            filters=filters,
        )
        ea.append(histogram)

    np.save(h5_dir / "timestamps_us.npy",           timestamps)
    np.save(h5_dir / "objframe_idx_2_repr_idx.npy", repr_idx)

    # --- Labels (copied unchanged) ---
    lbl_dir = out_dir / LABELS_DIR
    lbl_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        lbl_dir / "labels.npz",
        labels=labels,
        objframe_idx_2_label_idx=label_idx,
    )
    np.save(lbl_dir / "timestamps_us.npy", label_timestamps)
