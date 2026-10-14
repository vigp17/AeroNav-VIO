"""Estimate calibrated relative camera pose between consecutive EuRoC cam0 frames."""

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from aeronav_vio.calibration.euroc import load_cam0_calibration
from aeronav_vio.data.euroc import EurocSequence
from aeronav_vio.geometry.relative_pose import (
    DEFAULT_MIN_CHEIRALITY_COUNT,
    DEFAULT_MIN_CHEIRALITY_RATIO,
    MIN_CALIBRATED_CORRESPONDENCES,
    TRANSFORM_CONVENTION,
    TRANSLATION_SCALE_STATUS,
    estimate_calibrated_relative_pose,
    validate_rotation_matrix,
)


def to_json_compatible(value: Any) -> Any:
    """Recursively convert values to finite, standards-compliant JSON types."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        float_val = float(value)
        if not math.isfinite(float_val):
            raise ValueError(f"Nonfinite float cannot be serialized to JSON: {value!r}")
        return float_val
    if isinstance(value, np.ndarray):
        return to_json_compatible(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [to_json_compatible(item) for item in value]
    raise TypeError(f"Unsupported type for JSON serialization: {type(value)!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate calibrated two-view relative camera pose between consecutive "
            "EuRoC cam0 frames and write a JSON report."
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
        help=(
            "Zero-based reference frame index "
            "(uses frame-index + frame-gap as current)"
        ),
    )
    parser.add_argument(
        "--frame-gap",
        type=int,
        default=1,
        help="Positive integer step between reference and current frames (default: 1)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for the JSON relative-pose report",
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
        help="Milestone 2 fundamental-matrix RANSAC threshold in pixels (default: 1.0)",
    )
    parser.add_argument(
        "--ransac-confidence",
        type=float,
        default=0.99,
        help=(
            "Milestone 2 fundamental-matrix RANSAC confidence in (0, 1) "
            "(default: 0.99)"
        ),
    )
    parser.add_argument(
        "--essential-probability",
        type=float,
        default=0.999,
        help="Essential-matrix RANSAC probability in (0, 1) (default: 0.999)",
    )
    parser.add_argument(
        "--essential-threshold",
        type=float,
        default=1e-3,
        help=(
            "Essential-matrix RANSAC threshold in normalized image coordinates "
            "(default: 0.001)"
        ),
    )
    parser.add_argument(
        "--min-correspondences",
        type=int,
        default=MIN_CALIBRATED_CORRESPONDENCES,
        help="Minimum calibrated correspondences required (default: 8)",
    )
    parser.add_argument(
        "--min-cheirality-count",
        type=int,
        default=DEFAULT_MIN_CHEIRALITY_COUNT,
        help=(
            "Minimum cheirality-inlier count required for a valid pose quality gate "
            f"(default: {DEFAULT_MIN_CHEIRALITY_COUNT})"
        ),
    )
    parser.add_argument(
        "--min-cheirality-ratio",
        type=float,
        default=DEFAULT_MIN_CHEIRALITY_RATIO,
        help=(
            "Minimum cheirality-inlier ratio in (0, 1] required for a valid pose "
            f"quality gate (default: {DEFAULT_MIN_CHEIRALITY_RATIO})"
        ),
    )
    return parser


def main() -> int:
    """Run calibrated relative pose estimation and save a JSON report."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        sequence_root = args.sequence.expanduser().resolve(strict=True)
        output_path = args.output.expanduser().resolve(strict=False)
        if output_path == sequence_root or sequence_root in output_path.parents:
            raise ValueError(
                f"Refusing to write output inside dataset directory: {output_path}"
            )

        frame_gap = args.frame_gap
        if frame_gap < 1:
            raise ValueError(
                f"frame_gap must be a positive integer (>= 1), got {frame_gap}"
            )

        sequence = EurocSequence(sequence_root)
        calibration = load_cam0_calibration(sequence_root)

        ref_index = args.frame_index
        cur_index = ref_index + frame_gap
        pair_count = sequence.synchronized_pair_count
        if ref_index < 0 or ref_index >= pair_count:
            raise IndexError(
                f"Reference frame index {ref_index} is out of range for sequence "
                f"with {pair_count} synchronized pairs"
            )
        if cur_index >= pair_count:
            raise IndexError(
                f"Reference frame index {ref_index} with frame_gap {frame_gap} "
                f"requires current frame index {cur_index}, but sequence has "
                f"{pair_count} synchronized pairs"
            )

        ref_frame = sequence.frames[ref_index]
        cur_frame = sequence.frames[cur_index]
        ref_image = sequence.load_left_image(ref_index)
        cur_image = sequence.load_left_image(cur_index)

        match_result, pose_result = estimate_calibrated_relative_pose(
            ref_image,
            cur_image,
            calibration,
            max_features=args.max_features,
            ratio_threshold=args.ratio_threshold,
            fundamental_ransac_threshold=args.ransac_threshold,
            fundamental_ransac_confidence=args.ransac_confidence,
            essential_ransac_probability=args.essential_probability,
            essential_ransac_threshold=args.essential_threshold,
            min_correspondences=args.min_correspondences,
            minimum_cheirality_count=args.min_cheirality_count,
            minimum_cheirality_ratio=args.min_cheirality_ratio,
        )

        if (
            not pose_result.success
            or pose_result.essential_matrix is None
            or pose_result.rotation_matrix is None
            or pose_result.translation_direction is None
            or pose_result.statistics.rotation_angle_degrees is None
            or pose_result.statistics.translation_norm is None
        ):
            reason = pose_result.failure_reason or "Unknown pose recovery failure"
            raise RuntimeError(f"Relative pose estimation failed: {reason}")

        rotation_det = validate_rotation_matrix(pose_result.rotation_matrix)
        dt_ns = cur_frame.timestamp_ns - ref_frame.timestamp_ns
        match_stats = match_result.statistics
        pose_stats = pose_result.statistics

        raw_report = {
            "sequence_path": sequence_root,
            "calibration_file": calibration.calibration_file.resolve(),
            "reference_frame_index": ref_index,
            "current_frame_index": cur_index,
            "frame_gap": frame_gap,
            "reference_timestamp_ns": ref_frame.timestamp_ns,
            "current_timestamp_ns": cur_frame.timestamp_ns,
            "timestamp_difference_ns": dt_ns,
            "camera_model": calibration.camera_model,
            "distortion_model": calibration.distortion_model,
            "resolution": list(calibration.resolution),
            "rate_hz": calibration.rate_hz,
            "fx": calibration.fx,
            "fy": calibration.fy,
            "cx": calibration.cx,
            "cy": calibration.cy,
            "camera_matrix": calibration.camera_matrix,
            "distortion_coefficients": list(calibration.distortion_coefficients),
            "body_to_sensor_transform": calibration.body_to_sensor_transform,
            "reference_keypoint_count": match_stats.reference_keypoint_count,
            "current_keypoint_count": match_stats.current_keypoint_count,
            "ratio_filtered_match_count": match_stats.ratio_filtered_match_count,
            "milestone2_geometric_inlier_count": match_stats.geometric_inlier_count,
            "calibrated_input_correspondence_count": (
                pose_stats.input_correspondence_count
            ),
            "essential_inlier_count": pose_stats.essential_inlier_count,
            "cheirality_inlier_count": pose_stats.cheirality_inlier_count,
            "essential_inlier_ratio": pose_stats.essential_inlier_ratio,
            "cheirality_inlier_ratio": pose_stats.cheirality_inlier_ratio,
            "minimum_cheirality_count": pose_stats.minimum_cheirality_count,
            "minimum_cheirality_ratio": pose_stats.minimum_cheirality_ratio,
            "pose_quality_valid": pose_stats.pose_quality_valid,
            "pose_quality_reason": pose_stats.pose_quality_reason,
            "essential_matrix": pose_result.essential_matrix,
            "relative_rotation_matrix": pose_result.rotation_matrix,
            "rotation_determinant": rotation_det,
            "unit_translation_direction": pose_result.translation_direction,
            "translation_norm": pose_stats.translation_norm,
            "rotation_angle_degrees": pose_stats.rotation_angle_degrees,
            "transform_convention": TRANSFORM_CONVENTION,
            "translation_scale_status": TRANSLATION_SCALE_STATUS,
        }
        serializable_report = to_json_compatible(raw_report)
        json_text = json.dumps(serializable_report, indent=2, allow_nan=False)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_text + "\n", encoding="utf-8")
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        if error.__cause__ is not None:
            cause = error.__cause__
            detail += f" (caused by {type(cause).__name__}: {cause})"
        print(f"Error: {detail}", file=sys.stderr)
        return 1

    t_dir = pose_result.translation_direction.tolist()
    coeffs = list(calibration.distortion_coefficients)
    res_w, res_h = calibration.resolution

    print(f"Sequence path: {sequence_root}")
    print(f"Calibration file: {calibration.calibration_file.resolve()}")
    print(f"Reference frame index: {ref_index}")
    print(f"Current frame index: {cur_index}")
    print(f"Frame gap: {frame_gap}")
    print(f"Reference timestamp (ns): {ref_frame.timestamp_ns}")
    print(f"Current timestamp (ns): {cur_frame.timestamp_ns}")
    print(f"Timestamp difference (ns): {dt_ns}")
    print(f"Camera model: {calibration.camera_model}")
    print(f"Distortion model: {calibration.distortion_model}")
    print(f"Calibration resolution: {res_w}x{res_h}")
    print(f"fx: {calibration.fx}")
    print(f"fy: {calibration.fy}")
    print(f"cx: {calibration.cx}")
    print(f"cy: {calibration.cy}")
    print(f"Distortion coefficients: {coeffs}")
    print(f"Reference keypoint count: {match_stats.reference_keypoint_count}")
    print(f"Current keypoint count: {match_stats.current_keypoint_count}")
    print(f"Ratio-filtered match count: {match_stats.ratio_filtered_match_count}")
    print(
        f"Milestone 2 geometric-inlier count: {match_stats.geometric_inlier_count}"
    )
    print(
        "Calibrated input-correspondence count: "
        f"{pose_stats.input_correspondence_count}"
    )
    print(f"Essential-matrix inlier count: {pose_stats.essential_inlier_count}")
    print(f"Cheirality-inlier count: {pose_stats.cheirality_inlier_count}")
    print(f"Essential inlier ratio: {pose_stats.essential_inlier_ratio:.4f}")
    print(f"Cheirality inlier ratio: {pose_stats.cheirality_inlier_ratio:.4f}")
    print(f"Minimum cheirality count: {pose_stats.minimum_cheirality_count}")
    print(f"Minimum cheirality ratio: {pose_stats.minimum_cheirality_ratio:.4f}")
    print(f"Pose quality valid: {pose_stats.pose_quality_valid}")
    print(f"Pose quality reason: {pose_stats.pose_quality_reason}")
    print(f"Rotation angle (deg): {pose_stats.rotation_angle_degrees:.6f}")
    print(f"Rotation determinant: {rotation_det:.6f}")
    print(f"Unit translation direction: {t_dir}")
    print(f"Translation norm: {pose_stats.translation_norm:.6f}")
    print(f"Translation scale status: {TRANSLATION_SCALE_STATUS}")
    print(f"Output JSON path: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
