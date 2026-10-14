"""Evaluate Milestone 3 relative camera pose against EuRoC ground truth."""

import argparse
import json
import sys
from pathlib import Path

from aeronav_vio.calibration.euroc import load_cam0_calibration
from aeronav_vio.data.euroc import EurocSequence
from aeronav_vio.evaluation.relative_pose import (
    DEFAULT_TRANSLATION_EPSILON_M,
    evaluate_frame_pair_with_loaded_context,
    to_json_compatible,
)
from aeronav_vio.geometry.relative_pose import (
    DEFAULT_MIN_CHEIRALITY_COUNT,
    DEFAULT_MIN_CHEIRALITY_RATIO,
    MIN_CALIBRATED_CORRESPONDENCES,
    TRANSFORM_CONVENTION,
    TRANSLATION_SCALE_STATUS,
)
from aeronav_vio.ground_truth.euroc import EuRoCGroundTruthSequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Milestone 3 calibrated relative camera pose against "
            "interpolated EuRoC cam0 ground truth."
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
        help="Output path for the JSON evaluation report",
    )
    parser.add_argument(
        "--translation-epsilon",
        type=float,
        default=DEFAULT_TRANSLATION_EPSILON_M,
        help=(
            "Minimum ground-truth translation magnitude (m) required to define a "
            f"unit translation direction (default: {DEFAULT_TRANSLATION_EPSILON_M})"
        ),
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
    """Run selected-pair ground-truth relative-pose evaluation and write JSON."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        sequence_root = args.sequence.expanduser().resolve(strict=True)
        output_path = args.output.expanduser().resolve(strict=False)
        if output_path == sequence_root or sequence_root in output_path.parents:
            raise ValueError(
                f"Refusing to write output inside dataset directory: {output_path}"
            )

        sequence = EurocSequence(sequence_root)
        calibration = load_cam0_calibration(sequence_root)
        ground_truth = EuRoCGroundTruthSequence(sequence_root)

        evaluation = evaluate_frame_pair_with_loaded_context(
            sequence,
            calibration,
            ground_truth,
            args.frame_index,
            frame_gap=args.frame_gap,
            translation_epsilon_m=args.translation_epsilon,
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

        est = evaluation.estimated_result
        if (
            not est.success
            or est.rotation_matrix is None
            or est.translation_direction is None
            or est.statistics.rotation_angle_degrees is None
            or evaluation.relative_pose_error.rotation_error_degrees is None
        ):
            reason = est.failure_reason or "Unknown pose recovery failure"
            raise RuntimeError(f"Relative pose estimation failed: {reason}")

        ref_gt = evaluation.reference_ground_truth
        cur_gt = evaluation.current_ground_truth
        gt_rel = evaluation.ground_truth_relative_pose
        err = evaluation.relative_pose_error
        pose_stats = est.statistics

        raw_report = {
            "sequence_path": sequence_root,
            "ground_truth_csv_path": ground_truth.csv_path.resolve(),
            "calibration_file": calibration.calibration_file.resolve(),
            "reference_frame_index": evaluation.reference_frame_index,
            "current_frame_index": evaluation.current_frame_index,
            "frame_gap": evaluation.frame_gap,
            "reference_timestamp_ns": evaluation.reference_timestamp_ns,
            "current_timestamp_ns": evaluation.current_timestamp_ns,
            "reference_gt_lower_timestamp_ns": ref_gt.lower_timestamp_ns,
            "reference_gt_upper_timestamp_ns": ref_gt.upper_timestamp_ns,
            "reference_interpolation_fraction": ref_gt.interpolation_fraction,
            "reference_gt_exact_sample": ref_gt.exact_sample,
            "current_gt_lower_timestamp_ns": cur_gt.lower_timestamp_ns,
            "current_gt_upper_timestamp_ns": cur_gt.upper_timestamp_ns,
            "current_interpolation_fraction": cur_gt.interpolation_fraction,
            "current_gt_exact_sample": cur_gt.exact_sample,
            "estimated_pose_success": est.success,
            "pose_quality_valid": evaluation.pose_quality_valid,
            "pose_quality_reason": evaluation.pose_quality_reason,
            "minimum_cheirality_count": pose_stats.minimum_cheirality_count,
            "minimum_cheirality_ratio": pose_stats.minimum_cheirality_ratio,
            "calibrated_correspondence_count": pose_stats.input_correspondence_count,
            "essential_inlier_count": pose_stats.essential_inlier_count,
            "cheirality_inlier_count": pose_stats.cheirality_inlier_count,
            "cheirality_inlier_ratio": pose_stats.cheirality_inlier_ratio,
            "estimated_relative_rotation_matrix": est.rotation_matrix,
            "ground_truth_relative_rotation_matrix": gt_rel.rotation_current_reference,
            "estimated_rotation_angle_degrees": pose_stats.rotation_angle_degrees,
            "ground_truth_rotation_angle_degrees": gt_rel.rotation_angle_degrees,
            "rotation_error_degrees": err.rotation_error_degrees,
            "estimated_unit_translation_direction": est.translation_direction,
            "ground_truth_translation_vector_m": gt_rel.translation_current_reference,
            "ground_truth_unit_translation_direction": (
                gt_rel.translation_direction_current_reference
            ),
            "ground_truth_translation_magnitude_m": gt_rel.translation_magnitude_m,
            "translation_direction_error_degrees": (
                err.translation_direction_error_degrees
            ),
            "translation_direction_defined": err.translation_direction_defined,
            "translation_direction_undefined_reason": (
                gt_rel.translation_direction_undefined_reason
            ),
            "translation_epsilon_m": float(args.translation_epsilon),
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

    gt_unit_list = (
        gt_rel.translation_direction_current_reference.tolist()
        if gt_rel.translation_direction_current_reference is not None
        else None
    )
    trans_err_str = (
        f"{err.translation_direction_error_degrees:.6f}"
        if err.translation_direction_error_degrees is not None
        else "undefined"
    )

    print(f"Sequence path: {sequence_root}")
    print(f"Ground-truth CSV path: {ground_truth.csv_path.resolve()}")
    print(f"Calibration file: {calibration.calibration_file.resolve()}")
    print(f"Reference frame index: {evaluation.reference_frame_index}")
    print(f"Current frame index: {evaluation.current_frame_index}")
    print(f"Frame gap: {evaluation.frame_gap}")
    print(f"Reference timestamp (ns): {evaluation.reference_timestamp_ns}")
    print(f"Current timestamp (ns): {evaluation.current_timestamp_ns}")
    print(
        f"Reference GT bounds (ns): [{ref_gt.lower_timestamp_ns}, "
        f"{ref_gt.upper_timestamp_ns}] "
        f"(alpha={ref_gt.interpolation_fraction:.6f}, exact={ref_gt.exact_sample})"
    )
    print(
        f"Current GT bounds (ns): [{cur_gt.lower_timestamp_ns}, "
        f"{cur_gt.upper_timestamp_ns}] "
        f"(alpha={cur_gt.interpolation_fraction:.6f}, exact={cur_gt.exact_sample})"
    )
    print(f"Estimated pose success: {est.success}")
    print(f"Pose quality valid: {evaluation.pose_quality_valid}")
    print(f"Pose quality reason: {evaluation.pose_quality_reason}")
    print(
        f"Calibrated correspondence count: {pose_stats.input_correspondence_count}"
    )
    print(f"Essential-matrix inlier count: {pose_stats.essential_inlier_count}")
    print(f"Cheirality-inlier count: {pose_stats.cheirality_inlier_count}")
    print(f"Cheirality inlier ratio: {pose_stats.cheirality_inlier_ratio:.4f}")
    print(
        f"Estimated relative rotation:\n{est.rotation_matrix.tolist()}"
    )
    print(
        f"Ground-truth relative rotation:\n"
        f"{gt_rel.rotation_current_reference.tolist()}"
    )
    print(
        f"Estimated rotation angle (deg): "
        f"{pose_stats.rotation_angle_degrees:.6f}"
    )
    print(
        f"Ground-truth rotation angle (deg): "
        f"{gt_rel.rotation_angle_degrees:.6f}"
    )
    print(f"Rotation angular error (deg): {err.rotation_error_degrees:.6f}")
    print(
        f"Estimated unit translation direction: "
        f"{est.translation_direction.tolist()}"
    )
    print(f"Ground-truth unit translation direction: {gt_unit_list}")
    print(f"Translation-direction defined: {err.translation_direction_defined}")
    print(f"Translation-direction angular error (deg): {trans_err_str}")
    print(
        f"Ground-truth translation magnitude (m): "
        f"{gt_rel.translation_magnitude_m:.6f}"
    )
    print(f"Translation scale status: {TRANSLATION_SCALE_STATUS}")
    print(f"Output JSON path: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
