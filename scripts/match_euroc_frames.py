"""Match ORB features across two consecutive EuRoC cam0 frames."""

import argparse
import sys
from pathlib import Path

import cv2

from aeronav_vio.data.euroc import EurocSequence
from aeronav_vio.features.orb import match_temporal_pair, render_inlier_matches


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect ORB features and verify temporal matches between consecutive "
            "EuRoC cam0 frames."
        )
    )
    parser.add_argument(
        "--sequence",
        required=True,
        type=Path,
        help="EuRoC sequence root containing mav0",
    )
    parser.add_argument(
        "--frame-index",
        required=True,
        type=int,
        help="Zero-based reference frame index (uses frame-index + 1 as current)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for the inlier match visualization image",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=1000,
        help="Maximum ORB features to retain per image (default: 1000)",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=0.75,
        help="Lowe KNN ratio-test threshold in (0, 1) (default: 0.75)",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=1.0,
        help="RANSAC epipolar reprojection threshold in pixels (default: 1.0)",
    )
    parser.add_argument(
        "--ransac-confidence",
        type=float,
        default=0.99,
        help="RANSAC confidence in (0, 1) (default: 0.99)",
    )
    return parser


def main() -> int:
    """Run temporal ORB matching on a consecutive EuRoC cam0 pair."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        sequence_root = args.sequence.expanduser().resolve(strict=True)
        sequence = EurocSequence(sequence_root)

        ref_index = args.frame_index
        cur_index = ref_index + 1
        pair_count = sequence.synchronized_pair_count
        if ref_index < 0 or cur_index >= pair_count:
            raise IndexError(
                f"Reference frame index {ref_index} requires consecutive pair "
                f"({ref_index}, {cur_index}), but sequence has {pair_count} "
                "synchronized pairs"
            )

        ref_frame = sequence.frames[ref_index]
        cur_frame = sequence.frames[cur_index]
        ref_image = sequence.load_left_image(ref_index)
        cur_image = sequence.load_left_image(cur_index)

        match_result = match_temporal_pair(
            ref_image,
            cur_image,
            max_features=args.max_features,
            ratio_threshold=args.ratio_threshold,
            ransac_reprojection_threshold=args.ransac_threshold,
            ransac_confidence=args.ransac_confidence,
        )
        visualization = render_inlier_matches(
            ref_image,
            cur_image,
            match_result,
            reference_timestamp_ns=ref_frame.timestamp_ns,
            current_timestamp_ns=cur_frame.timestamp_ns,
        )

        output_path = args.output.expanduser().resolve(strict=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), visualization):
            raise OSError(f"Failed to write output visualization: {output_path}")
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        if error.__cause__ is not None:
            cause = error.__cause__
            detail += f" (caused by {type(cause).__name__}: {cause})"
        print(f"Error: {detail}", file=sys.stderr)
        return 1

    stats = match_result.statistics
    dt_ns = cur_frame.timestamp_ns - ref_frame.timestamp_ns
    vis_height, vis_width = visualization.shape[0], visualization.shape[1]

    print(f"Sequence path: {sequence_root}")
    print(f"Reference frame index: {ref_index}")
    print(f"Current frame index: {cur_index}")
    print(f"Reference timestamp (ns): {ref_frame.timestamp_ns}")
    print(f"Current timestamp (ns): {cur_frame.timestamp_ns}")
    print(f"Timestamp difference (ns): {dt_ns}")
    print(f"Reference keypoint count: {stats.reference_keypoint_count}")
    print(f"Current keypoint count: {stats.current_keypoint_count}")
    print(f"Raw KNN match count: {stats.raw_knn_match_count}")
    print(f"Ratio-filtered match count: {stats.ratio_filtered_match_count}")
    print(f"Geometric inlier count: {stats.geometric_inlier_count}")
    print(f"Geometric outlier count: {stats.geometric_outlier_count}")
    print(f"Geometric inlier ratio: {stats.inlier_ratio:.4f}")
    print(f"Output image path: {output_path}")
    print(f"Output image resolution: {vis_width}x{vis_height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
