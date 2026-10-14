"""Inspect one EuRoC stereo sequence without changing its files."""

import argparse
import sys
from pathlib import Path

from aeronav_vio.data.euroc import EurocSequence


def main() -> int:
    """Print stereo metadata and the first pair's image properties."""
    parser = argparse.ArgumentParser(description="Inspect a EuRoC stereo sequence")
    parser.add_argument(
        "--sequence", required=True, type=Path, help="Sequence root containing mav0"
    )
    args = parser.parse_args()

    try:
        sequence_root = args.sequence.expanduser().resolve(strict=True)
        sequence = EurocSequence(sequence_root)
        left, right = sequence.load_stereo_pair(0)
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        if error.__cause__ is not None:
            cause = error.__cause__
            detail += f" (caused by {type(cause).__name__}: {cause})"
        print(f"Error: {detail}", file=sys.stderr)
        return 1

    print(f"Sequence path: {sequence_root}")
    print(f"cam0 record count: {sequence.cam0_record_count}")
    print(f"cam1 record count: {sequence.cam1_record_count}")
    print(f"Synchronized stereo-pair count: {sequence.synchronized_pair_count}")
    print(f"Unmatched cam0 timestamp count: {len(sequence.unmatched_cam0_timestamps)}")
    print(f"Unmatched cam1 timestamp count: {len(sequence.unmatched_cam1_timestamps)}")
    print(f"First synchronized timestamp (ns): {sequence.frames[0].timestamp_ns}")
    print(f"Last synchronized timestamp (ns): {sequence.frames[-1].timestamp_ns}")
    print(f"First left-image resolution: {left.shape[1]}x{left.shape[0]}")
    print(f"First right-image resolution: {right.shape[1]}x{right.shape[0]}")
    print(f"First left-image NumPy data type: {left.dtype}")
    print(f"First right-image NumPy data type: {right.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())