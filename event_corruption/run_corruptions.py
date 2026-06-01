"""
run_corruptions.py — entry point for the Gen1 corruption benchmark.

Usage
-----
    python run_corruptions.py [--split test|train|val] [--output <dir>]
                              [--seed 42] [--corruptions hot_pixel,event_flood]
                              [--severities 1,3,5] [--no-skip]

Examples
--------
    # Run all 6 corruptions x 5 severities on the test split
    python run_corruptions.py --split test

    # Run only two corruptions at severities 1 and 5
    python run_corruptions.py --split test --corruptions hot_pixel,spatial_dropout --severities 1,5
"""
import argparse
import logging
import sys
from pathlib import Path

# Allow running from the event_corruption/ root without installing as a package
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.runner import run_all_corruptions
from corrupt.registry import CORRUPTIONS, SEVERITIES

# ---------------------------------------------------------------------------
# Defaults — adjust to your actual paths
# ---------------------------------------------------------------------------
DEFAULT_GEN1_ROOT = Path("d:/Perdue/gen1")
DEFAULT_OUTPUT    = Path("d:/Perdue/gen1_corrupted")
DEFAULT_SEED      = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply event-stream corruptions to the Gen1 HDF5 dataset."
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["test", "train", "val"],
        help="Which Gen1 split to corrupt (default: test)",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Override input directory (default: <gen1_root>/<split>). "
            "Must contain timestamped sequence subdirs."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Root output directory (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--gen1-root",
        type=Path,
        default=DEFAULT_GEN1_ROOT,
        help=f"Root of the Gen1 dataset (default: {DEFAULT_GEN1_ROOT})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base RNG seed (default: {DEFAULT_SEED})",
    )
    p.add_argument(
        "--corruptions",
        type=str,
        default=None,
        help=(
            "Comma-separated list of corruptions to run "
            f"(default: all). Options: {','.join(CORRUPTIONS)}"
        ),
    )
    p.add_argument(
        "--severities",
        type=str,
        default=None,
        help=(
            "Comma-separated severity levels to run "
            f"(default: all). Options: {','.join(map(str, SEVERITIES))}"
        ),
    )
    p.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-run even if output already exists (default: skip existing)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    # Resolve input directory
    input_dir = args.input or (args.gen1_root / args.split)
    if not input_dir.exists():
        logging.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    # Parse optional corruption / severity subsets
    selected_corruptions = (
        [c.strip() for c in args.corruptions.split(",")]
        if args.corruptions else None
    )
    selected_severities = (
        [int(s.strip()) for s in args.severities.split(",")]
        if args.severities else None
    )

    # Validate
    if selected_corruptions:
        unknown = set(selected_corruptions) - set(CORRUPTIONS)
        if unknown:
            logging.error("Unknown corruption(s): %s", unknown)
            sys.exit(1)

    if selected_severities:
        unknown = set(selected_severities) - set(SEVERITIES)
        if unknown:
            logging.error("Unknown severity/severities: %s", unknown)
            sys.exit(1)

    output_dir = args.output / args.split

    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    print(f"Seed   : {args.seed}")

    run_all_corruptions(
        input_dir        = input_dir,
        output_dir       = output_dir,
        seed             = args.seed,
        split            = args.split,
        corruptions      = selected_corruptions,
        severities       = selected_severities,
        skip_existing    = not args.no_skip,
    )


if __name__ == "__main__":
    main()
