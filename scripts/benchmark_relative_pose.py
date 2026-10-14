"""Run a diagnostic relative-pose benchmark across frame gaps on one EuRoC sequence."""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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

CSV_COLUMNS = (
    "reference_index",
    "current_index",
    "frame_gap",
    "reference_timestamp_ns",
    "current_timestamp_ns",
    "ground_truth_available",
    "estimation_success",
    "pose_quality_valid",
    "pose_quality_reason",
    "calibrated_correspondence_count",
    "essential_inlier_count",
    "cheirality_inlier_count",
    "cheirality_inlier_ratio",
    "estimated_rotation_angle_degrees",
    "ground_truth_rotation_angle_degrees",
    "rotation_error_degrees",
    "ground_truth_translation_magnitude_m",
    "translation_direction_error_degrees",
    "translation_direction_defined",
    "failure_category",
    "failure_reason",
)


@dataclass(frozen=True)
class BenchmarkPairRecord:
    """One evaluated or skipped pair row in the Milestone 4A benchmark."""

    reference_index: int
    current_index: int
    frame_gap: int
    reference_timestamp_ns: int
    current_timestamp_ns: int
    ground_truth_available: bool
    estimation_success: bool
    pose_quality_valid: bool
    pose_quality_reason: str
    calibrated_correspondence_count: int | None
    essential_inlier_count: int | None
    cheirality_inlier_count: int | None
    cheirality_inlier_ratio: float | None
    estimated_rotation_angle_degrees: float | None
    ground_truth_rotation_angle_degrees: float | None
    rotation_error_degrees: float | None
    ground_truth_translation_magnitude_m: float | None
    translation_direction_error_degrees: float | None
    translation_direction_defined: bool
    failure_category: str
    failure_reason: str


def _summary_stats(values: list[float]) -> dict[str, float | None]:
    """Compute median, mean, and 90th-percentile for a list of finite floats."""
    if not values:
        return {
            "median": None,
            "mean": None,
            "p90": None,
        }
    arr = np.array(values, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90.0)),
    }


def _subset_error_metrics(records: list[BenchmarkPairRecord]) -> dict[str, Any]:
    """Compute error summaries for a subset of successfully recovered pose records."""
    rot_errors = [
        rec.rotation_error_degrees
        for rec in records
        if rec.rotation_error_degrees is not None
    ]
    trans_errors = [
        rec.translation_direction_error_degrees
        for rec in records
        if rec.translation_direction_defined
        and rec.translation_direction_error_degrees is not None
    ]
    rot_stats = _summary_stats(rot_errors)
    trans_stats = _summary_stats(trans_errors)
    return {
        "pair_count": len(records),
        "translation_direction_defined_count": len(trans_errors),
        "median_rotation_error_degrees": rot_stats["median"],
        "mean_rotation_error_degrees": rot_stats["mean"],
        "p90_rotation_error_degrees": rot_stats["p90"],
        "median_translation_direction_error_degrees": trans_stats["median"],
        "mean_translation_direction_error_degrees": trans_stats["mean"],
        "p90_translation_direction_error_degrees": trans_stats["p90"],
    }


def _aggregate_for_gap(
    frame_gap: int, records: list[BenchmarkPairRecord]
) -> dict[str, Any]:
    """Aggregate benchmark counts and error statistics for one frame gap."""
    total_requested = len(records)
    gt_unavailable = [r for r in records if not r.ground_truth_available]
    est_failed = [
        r for r in records if r.ground_truth_available and not r.estimation_success
    ]
    evaluated = [
        r for r in records if r.ground_truth_available and r.estimation_success
    ]
    accepted = [r for r in evaluated if r.pose_quality_valid]
    rejected = [r for r in evaluated if not r.pose_quality_valid]

    all_metrics = _subset_error_metrics(evaluated)
    accepted_metrics = _subset_error_metrics(accepted)
    rejected_metrics = _subset_error_metrics(rejected)

    return {
        "frame_gap": frame_gap,
        "total_requested_pairs": total_requested,
        "evaluated_pairs": len(evaluated),
        "ground_truth_unavailable_count": len(gt_unavailable),
        "estimation_failure_count": len(est_failed),
        "quality_gate_accepted_count": len(accepted),
        "quality_gate_rejected_count": len(rejected),
        "translation_direction_defined_count": all_metrics[
            "translation_direction_defined_count"
        ],
        "median_rotation_error_degrees": all_metrics[
            "median_rotation_error_degrees"
        ],
        "mean_rotation_error_degrees": all_metrics["mean_rotation_error_degrees"],
        "p90_rotation_error_degrees": all_metrics["p90_rotation_error_degrees"],
        "median_translation_direction_error_degrees": all_metrics[
            "median_translation_direction_error_degrees"
        ],
        "mean_translation_direction_error_degrees": all_metrics[
            "mean_translation_direction_error_degrees"
        ],
        "p90_translation_direction_error_degrees": all_metrics[
            "p90_translation_direction_error_degrees"
        ],
        "quality_gate_accepted": accepted_metrics,
        "quality_gate_rejected": rejected_metrics,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Milestone 4A relative-pose benchmark across selected frame "
            "ranges and frame gaps."
        )
    )
    parser.add_argument(
        "--sequence",
        required=True,
        type=Path,
        help="EuRoC sequence root containing mav0",
    )
    parser.add_argument(
        "--start-frame",
        required=True,
        type=int,
        help="Inclusive start reference frame index",
    )
    parser.add_argument(
        "--end-frame",
        required=True,
        type=int,
        help="Inclusive end reference frame index",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Positive sampling stride for reference frames (default: 10)",
    )
    parser.add_argument(
        "--frame-gaps",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10],
        help="One or more positive integer frame gaps (default: 1 2 5 10)",
    )
    parser.add_argument(
        "--csv-output",
        required=True,
        type=Path,
        help="Output path for per-pair CSV results",
    )
    parser.add_argument(
        "--json-output",
        required=True,
        type=Path,
        help="Output path for aggregate JSON results",
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


def _format_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> int:
    """Execute the Milestone 4A diagnostic relative-pose benchmark."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        sequence_root = args.sequence.expanduser().resolve(strict=True)
        csv_output = args.csv_output.expanduser().resolve(strict=False)
        json_output = args.json_output.expanduser().resolve(strict=False)
        for out_path in (csv_output, json_output):
            if out_path == sequence_root or sequence_root in out_path.parents:
                raise ValueError(
                    f"Refusing to write benchmark output inside dataset directory: "
                    f"{out_path}"
                )

        if args.stride < 1:
            raise ValueError(
                f"stride must be a positive integer (>= 1), got {args.stride}"
            )
        if not args.frame_gaps or any(gap < 1 for gap in args.frame_gaps):
            raise ValueError(
                f"All frame_gaps must be positive integers (>= 1), got "
                f"{args.frame_gaps}"
            )

        sequence = EurocSequence(sequence_root)
        calibration = load_cam0_calibration(sequence_root)
        ground_truth = EuRoCGroundTruthSequence(sequence_root)

        pair_count = sequence.synchronized_pair_count
        max_gap = max(args.frame_gaps)
        if args.start_frame < 0 or args.end_frame < args.start_frame:
            raise ValueError(
                f"Invalid frame range [{args.start_frame}, {args.end_frame}]"
            )
        if args.end_frame + max_gap >= pair_count:
            raise IndexError(
                f"end_frame ({args.end_frame}) + max_frame_gap ({max_gap}) = "
                f"{args.end_frame + max_gap} exceeds available synchronized pairs "
                f"({pair_count})"
            )

        reference_indices = list(
            range(args.start_frame, args.end_frame + 1, args.stride)
        )
        all_records: list[BenchmarkPairRecord] = []
        records_by_gap: dict[int, list[BenchmarkPairRecord]] = {
            gap: [] for gap in args.frame_gaps
        }

        for gap in args.frame_gaps:
            for ref_idx in reference_indices:
                cur_idx = ref_idx + gap
                ref_ts = sequence.frames[ref_idx].timestamp_ns
                cur_ts = sequence.frames[cur_idx].timestamp_ns

                if not (
                    ground_truth.contains_timestamp(ref_ts)
                    and ground_truth.contains_timestamp(cur_ts)
                ):
                    unavail_reason = (
                        f"Camera pair timestamps ({ref_ts}, {cur_ts}) ns lie outside "
                        f"ground-truth range [{ground_truth.first_timestamp_ns}, "
                        f"{ground_truth.last_timestamp_ns}] ns"
                    )
                    rec = BenchmarkPairRecord(
                        reference_index=ref_idx,
                        current_index=cur_idx,
                        frame_gap=gap,
                        reference_timestamp_ns=ref_ts,
                        current_timestamp_ns=cur_ts,
                        ground_truth_available=False,
                        estimation_success=False,
                        pose_quality_valid=False,
                        pose_quality_reason="Ground truth unavailable",
                        calibrated_correspondence_count=None,
                        essential_inlier_count=None,
                        cheirality_inlier_count=None,
                        cheirality_inlier_ratio=None,
                        estimated_rotation_angle_degrees=None,
                        ground_truth_rotation_angle_degrees=None,
                        rotation_error_degrees=None,
                        ground_truth_translation_magnitude_m=None,
                        translation_direction_error_degrees=None,
                        translation_direction_defined=False,
                        failure_category="ground_truth_unavailable",
                        failure_reason=unavail_reason,
                    )
                    all_records.append(rec)
                    records_by_gap[gap].append(rec)
                    continue

                eval_res = evaluate_frame_pair_with_loaded_context(
                    sequence,
                    calibration,
                    ground_truth,
                    ref_idx,
                    frame_gap=gap,
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

                est = eval_res.estimated_result
                pose_stats = est.statistics
                gt_rel = eval_res.ground_truth_relative_pose
                err = eval_res.relative_pose_error

                if not est.success:
                    failure_category = "estimation_failure"
                    failure_reason = (
                        est.failure_reason or "Relative pose estimation failed"
                    )
                elif not eval_res.pose_quality_valid:
                    failure_category = "quality_gate_rejected"
                    failure_reason = eval_res.pose_quality_reason
                elif not err.translation_direction_defined:
                    failure_category = "translation_direction_undefined"
                    failure_reason = err.failure_reason or ""
                else:
                    failure_category = "none"
                    failure_reason = ""

                rec = BenchmarkPairRecord(
                    reference_index=ref_idx,
                    current_index=cur_idx,
                    frame_gap=gap,
                    reference_timestamp_ns=ref_ts,
                    current_timestamp_ns=cur_ts,
                    ground_truth_available=True,
                    estimation_success=est.success,
                    pose_quality_valid=eval_res.pose_quality_valid,
                    pose_quality_reason=eval_res.pose_quality_reason,
                    calibrated_correspondence_count=(
                        pose_stats.input_correspondence_count
                    ),
                    essential_inlier_count=pose_stats.essential_inlier_count,
                    cheirality_inlier_count=pose_stats.cheirality_inlier_count,
                    cheirality_inlier_ratio=pose_stats.cheirality_inlier_ratio,
                    estimated_rotation_angle_degrees=(
                        pose_stats.rotation_angle_degrees
                    ),
                    ground_truth_rotation_angle_degrees=gt_rel.rotation_angle_degrees,
                    rotation_error_degrees=err.rotation_error_degrees,
                    ground_truth_translation_magnitude_m=(
                        gt_rel.translation_magnitude_m
                    ),
                    translation_direction_error_degrees=(
                        err.translation_direction_error_degrees
                    ),
                    translation_direction_defined=err.translation_direction_defined,
                    failure_category=failure_category,
                    failure_reason=failure_reason,
                )
                all_records.append(rec)
                records_by_gap[gap].append(rec)

        per_gap_summaries = [
            _aggregate_for_gap(gap, records_by_gap[gap]) for gap in args.frame_gaps
        ]
        overall_summary = _aggregate_for_gap(0, all_records)
        overall_summary.pop("frame_gap", None)

        raw_json = {
            "sequence_path": sequence_root,
            "ground_truth_csv_path": ground_truth.csv_path.resolve(),
            "calibration_file": calibration.calibration_file.resolve(),
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "stride": args.stride,
            "frame_gaps": list(args.frame_gaps),
            "minimum_cheirality_count": args.min_cheirality_count,
            "minimum_cheirality_ratio": float(args.min_cheirality_ratio),
            "translation_epsilon_m": float(args.translation_epsilon),
            "transform_convention": TRANSFORM_CONVENTION,
            "translation_scale_status": TRANSLATION_SCALE_STATUS,
            "overall": overall_summary,
            "per_frame_gap": per_gap_summaries,
        }
        serializable_json = to_json_compatible(raw_json)
        json_text = json.dumps(serializable_json, indent=2, allow_nan=False)

        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with csv_output.open("w", newline="", encoding="utf-8") as csv_handle:
            writer = csv.writer(csv_handle)
            writer.writerow(CSV_COLUMNS)
            for rec in all_records:
                writer.writerow(
                    [
                        _format_csv_value(getattr(rec, col))
                        for col in CSV_COLUMNS
                    ]
                )

        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json_text + "\n", encoding="utf-8")
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        if error.__cause__ is not None:
            cause = error.__cause__
            detail += f" (caused by {type(cause).__name__}: {cause})"
        print(f"Error: {detail}", file=sys.stderr)
        return 1

    print(f"Sequence path: {sequence_root}")
    print(
        f"Reference frame range: [{args.start_frame}, {args.end_frame}] "
        f"(stride={args.stride}, gaps={args.frame_gaps})"
    )
    for gap_summary in per_gap_summaries:
        gap = gap_summary["frame_gap"]
        acc = gap_summary["quality_gate_accepted"]
        rej = gap_summary["quality_gate_rejected"]
        print(
            f"Gap {gap}: requested={gap_summary['total_requested_pairs']}, "
            f"evaluated={gap_summary['evaluated_pairs']}, "
            f"gt_unavail={gap_summary['ground_truth_unavailable_count']}, "
            f"est_fail={gap_summary['estimation_failure_count']}, "
            f"accepted={gap_summary['quality_gate_accepted_count']}, "
            f"rejected={gap_summary['quality_gate_rejected_count']}"
        )
        print(
            f"  All evaluated: "
            f"med_rot_err={gap_summary['median_rotation_error_degrees']}, "
            f"med_trans_err={gap_summary['median_translation_direction_error_degrees']}"
        )
        print(
            f"  Accepted ({acc['pair_count']}): "
            f"med_rot_err={acc['median_rotation_error_degrees']}, "
            f"med_trans_err={acc['median_translation_direction_error_degrees']}"
        )
        print(
            f"  Rejected ({rej['pair_count']}): "
            f"med_rot_err={rej['median_rotation_error_degrees']}, "
            f"med_trans_err={rej['median_translation_direction_error_degrees']}"
        )
    print(f"Per-pair CSV output: {csv_output}")
    print(f"Aggregate JSON output: {json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
