"""Ground-truth relative-pose computation and Milestone 4A accuracy evaluation."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aeronav_vio.calibration.euroc import CameraCalibration, load_cam0_calibration
from aeronav_vio.data.euroc import EurocSequence
from aeronav_vio.features.orb import TemporalMatchResult
from aeronav_vio.geometry.relative_pose import (
    DEFAULT_MIN_CHEIRALITY_COUNT,
    DEFAULT_MIN_CHEIRALITY_RATIO,
    MIN_CALIBRATED_CORRESPONDENCES,
    RelativePoseResult,
    compute_rotation_angle_degrees,
    estimate_calibrated_relative_pose,
    validate_rotation_matrix,
)
from aeronav_vio.ground_truth.euroc import (
    CameraPose,
    EuRoCGroundTruthSequence,
    InterpolatedGroundTruthPose,
    invert_rigid_transform,
    validate_homogeneous_transform,
)

DEFAULT_TRANSLATION_EPSILON_M = 1e-4
_UNIT_NORM_ATOL = 1e-4


def _readonly_float64(array: np.ndarray) -> np.ndarray:
    """Return a read-only float64 copy of array."""
    out = np.array(array, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


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


@dataclass(frozen=True)
class GroundTruthRelativePose:
    """Ground-truth cam0 relative motion in the Milestone 3 convention.

    Transform convention:
        X_current =
            rotation_current_reference @ X_reference
            + translation_current_reference
    where X_reference is a 3D point in the reference cam0 frame and X_current is
    the same 3D point in the current cam0 frame.
    """

    reference_timestamp_ns: int
    current_timestamp_ns: int
    rotation_current_reference: np.ndarray
    translation_current_reference: np.ndarray
    translation_direction_current_reference: np.ndarray | None
    translation_magnitude_m: float
    transform_current_reference: np.ndarray
    rotation_angle_degrees: float
    translation_direction_defined: bool
    translation_direction_undefined_reason: str | None = None


@dataclass(frozen=True)
class RelativePoseError:
    """Angular rotation and translation-direction errors against ground truth."""

    rotation_error_degrees: float | None
    translation_direction_error_degrees: float | None
    ground_truth_translation_magnitude_m: float
    translation_direction_defined: bool
    failure_reason: str | None


@dataclass(frozen=True)
class RelativePoseEvaluation:
    """Complete Milestone 4A evaluation for one reference/current cam0 frame pair."""

    estimated_result: RelativePoseResult
    ground_truth_relative_pose: GroundTruthRelativePose
    relative_pose_error: RelativePoseError
    pose_quality_valid: bool
    pose_quality_reason: str
    frame_gap: int
    reference_frame_index: int
    current_frame_index: int
    reference_timestamp_ns: int
    current_timestamp_ns: int
    reference_ground_truth: InterpolatedGroundTruthPose
    current_ground_truth: InterpolatedGroundTruthPose
    reference_camera_pose: CameraPose
    current_camera_pose: CameraPose
    match_result: TemporalMatchResult


def _validate_translation_epsilon(translation_epsilon_m: float) -> float:
    """Validate that the near-zero ground-truth translation epsilon is positive."""
    if isinstance(translation_epsilon_m, bool) or not isinstance(
        translation_epsilon_m, (int, float)
    ):
        raise TypeError("translation_epsilon_m must be a positive real number")
    eps = float(translation_epsilon_m)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(
            f"translation_epsilon_m must be finite and positive, got "
            f"{translation_epsilon_m!r}"
        )
    return eps


def compute_ground_truth_relative_camera_pose(
    reference_camera_pose: CameraPose,
    current_camera_pose: CameraPose,
    *,
    translation_epsilon_m: float = DEFAULT_TRANSLATION_EPSILON_M,
) -> GroundTruthRelativePose:
    """Compute ground-truth relative cam0 pose in the Milestone 3 convention.

    Given world-to-camera poses T_WC_ref and T_WC_cur satisfying:
        X_W = R_WC_ref @ X_C_ref + p_WC_ref
        X_W = R_WC_cur @ X_C_cur + p_WC_cur
    the relative transform mapping reference-camera coordinates X_C_ref into
    current-camera coordinates X_C_cur is:
        T_cur_ref = inv(T_WC_cur) @ T_WC_ref
        R_cur_ref = R_WC_cur.T @ R_WC_ref
        t_cur_ref = R_WC_cur.T @ (p_WC_ref - p_WC_cur)
        X_C_cur   = R_cur_ref @ X_C_ref + t_cur_ref
    """
    eps = _validate_translation_epsilon(translation_epsilon_m)
    validate_homogeneous_transform(
        reference_camera_pose.transform_world_camera,
        label="reference_camera_pose.transform_world_camera",
    )
    validate_homogeneous_transform(
        current_camera_pose.transform_world_camera,
        label="current_camera_pose.transform_world_camera",
    )

    t_wc_ref = reference_camera_pose.transform_world_camera
    t_wc_cur = current_camera_pose.transform_world_camera
    t_cw_cur = invert_rigid_transform(t_wc_cur)

    t_cur_ref = t_cw_cur @ t_wc_ref
    validate_homogeneous_transform(t_cur_ref, label="transform_current_reference")

    r_cur_ref = _readonly_float64(t_cur_ref[:3, :3])
    trans_cur_ref = _readonly_float64(t_cur_ref[:3, 3])
    rot_angle_deg = compute_rotation_angle_degrees(r_cur_ref)

    magnitude_m = float(np.linalg.norm(trans_cur_ref))
    if not math.isfinite(magnitude_m) or magnitude_m < 0.0:
        raise ValueError("Computed nonfinite ground-truth translation magnitude")

    if magnitude_m < eps:
        direction = None
        direction_defined = False
        undefined_reason = (
            f"Ground-truth translation magnitude ({magnitude_m:.6e} m) is below "
            f"near-zero epsilon ({eps:.6e} m); unit translation direction is undefined"
        )
    else:
        direction = _readonly_float64(trans_cur_ref / magnitude_m)
        direction_defined = True
        undefined_reason = None

    return GroundTruthRelativePose(
        reference_timestamp_ns=reference_camera_pose.timestamp_ns,
        current_timestamp_ns=current_camera_pose.timestamp_ns,
        rotation_current_reference=r_cur_ref,
        translation_current_reference=trans_cur_ref,
        translation_direction_current_reference=direction,
        translation_magnitude_m=magnitude_m,
        transform_current_reference=_readonly_float64(t_cur_ref),
        rotation_angle_degrees=rot_angle_deg,
        translation_direction_defined=direction_defined,
        translation_direction_undefined_reason=undefined_reason,
    )


def compute_rotation_error_degrees(
    estimated_rotation: np.ndarray,
    ground_truth_rotation: np.ndarray,
) -> float:
    """Compute angular error in [0, 180] degrees via R_error = R_est @ R_gt.T."""
    validate_rotation_matrix(estimated_rotation)
    validate_rotation_matrix(ground_truth_rotation)

    r_error = estimated_rotation @ ground_truth_rotation.T
    validate_rotation_matrix(r_error)
    return compute_rotation_angle_degrees(r_error)


def compute_translation_direction_error_degrees(
    estimated_translation_direction: np.ndarray,
    ground_truth_translation_direction: np.ndarray,
) -> float:
    """Compute angular error in [0, 180] degrees between unit translation directions.

    Opposite directions (+t and -t) are NOT treated as equivalent; no absolute
    value is applied to the dot product.
    """
    if not isinstance(estimated_translation_direction, np.ndarray) or (
        estimated_translation_direction.shape != (3,)
    ):
        raise ValueError(
            "Expected estimated_translation_direction to be a NumPy array of shape (3,)"
        )
    if not isinstance(ground_truth_translation_direction, np.ndarray) or (
        ground_truth_translation_direction.shape != (3,)
    ):
        raise ValueError(
            "Expected ground_truth_translation_direction to be a NumPy array of "
            "shape (3,)"
        )
    if not np.all(np.isfinite(estimated_translation_direction)):
        raise ValueError("estimated_translation_direction contains nonfinite values")
    if not np.all(np.isfinite(ground_truth_translation_direction)):
        raise ValueError(
            "ground_truth_translation_direction contains nonfinite values"
        )

    est_norm = float(np.linalg.norm(estimated_translation_direction))
    gt_norm = float(np.linalg.norm(ground_truth_translation_direction))
    if not math.isclose(est_norm, 1.0, abs_tol=_UNIT_NORM_ATOL):
        raise ValueError(
            f"estimated_translation_direction must have unit norm, got {est_norm:.6f}"
        )
    if not math.isclose(gt_norm, 1.0, abs_tol=_UNIT_NORM_ATOL):
        raise ValueError(
            f"ground_truth_translation_direction must have unit norm, got {gt_norm:.6f}"
        )

    est_unit = estimated_translation_direction / est_norm
    gt_unit = ground_truth_translation_direction / gt_norm
    cos_angle = float(np.clip(np.dot(est_unit, gt_unit), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def compare_relative_pose(
    estimated_pose: RelativePoseResult,
    ground_truth_relative_pose: GroundTruthRelativePose,
    *,
    translation_epsilon_m: float = DEFAULT_TRANSLATION_EPSILON_M,
) -> RelativePoseError:
    """Compare a Milestone 3 RelativePoseResult with ground-truth relative cam0 pose."""
    eps = _validate_translation_epsilon(translation_epsilon_m)
    validate_rotation_matrix(ground_truth_relative_pose.rotation_current_reference)

    gt_mag = float(ground_truth_relative_pose.translation_magnitude_m)
    if not math.isfinite(gt_mag) or gt_mag < 0.0:
        raise ValueError("ground_truth_relative_pose.translation_magnitude_m invalid")

    direction_defined = (
        ground_truth_relative_pose.translation_direction_defined
        and ground_truth_relative_pose.translation_direction_current_reference
        is not None
        and gt_mag >= eps
    )
    gt_undef_reason = (
        ground_truth_relative_pose.translation_direction_undefined_reason
        or (
            f"Ground-truth translation magnitude ({gt_mag:.6e} m) is below "
            f"near-zero epsilon ({eps:.6e} m); unit translation direction is undefined"
        )
    )

    if (
        not estimated_pose.success
        or estimated_pose.rotation_matrix is None
        or estimated_pose.translation_direction is None
    ):
        est_reason = (
            estimated_pose.failure_reason or "Estimated relative pose was not recovered"
        )
        return RelativePoseError(
            rotation_error_degrees=None,
            translation_direction_error_degrees=None,
            ground_truth_translation_magnitude_m=gt_mag,
            translation_direction_defined=direction_defined,
            failure_reason=est_reason,
        )

    rotation_error_deg = compute_rotation_error_degrees(
        estimated_pose.rotation_matrix,
        ground_truth_relative_pose.rotation_current_reference,
    )

    if not direction_defined or (
        ground_truth_relative_pose.translation_direction_current_reference is None
    ):
        return RelativePoseError(
            rotation_error_degrees=rotation_error_deg,
            translation_direction_error_degrees=None,
            ground_truth_translation_magnitude_m=gt_mag,
            translation_direction_defined=False,
            failure_reason=gt_undef_reason,
        )

    translation_error_deg = compute_translation_direction_error_degrees(
        estimated_pose.translation_direction,
        ground_truth_relative_pose.translation_direction_current_reference,
    )
    return RelativePoseError(
        rotation_error_degrees=rotation_error_deg,
        translation_direction_error_degrees=translation_error_deg,
        ground_truth_translation_magnitude_m=gt_mag,
        translation_direction_defined=True,
        failure_reason=None,
    )


def evaluate_frame_pair_with_loaded_context(
    sequence: EurocSequence,
    calibration: CameraCalibration,
    ground_truth: EuRoCGroundTruthSequence,
    reference_frame_index: int,
    *,
    frame_gap: int = 1,
    translation_epsilon_m: float = DEFAULT_TRANSLATION_EPSILON_M,
    max_features: int = 1000,
    ratio_threshold: float = 0.75,
    fundamental_ransac_threshold: float = 1.0,
    fundamental_ransac_confidence: float = 0.99,
    essential_ransac_probability: float = 0.999,
    essential_ransac_threshold: float = 1e-3,
    min_correspondences: int = MIN_CALIBRATED_CORRESPONDENCES,
    minimum_cheirality_count: int = DEFAULT_MIN_CHEIRALITY_COUNT,
    minimum_cheirality_ratio: float = DEFAULT_MIN_CHEIRALITY_RATIO,
) -> RelativePoseEvaluation:
    """Evaluate one cam0 frame pair using pre-loaded sequence, calibration, and GT."""
    if isinstance(frame_gap, bool) or not isinstance(frame_gap, int) or frame_gap < 1:
        raise ValueError(
            f"frame_gap must be a positive integer (>= 1), got {frame_gap!r}"
        )
    _validate_translation_epsilon(translation_epsilon_m)

    pair_count = sequence.synchronized_pair_count
    if reference_frame_index < 0 or reference_frame_index >= pair_count:
        raise IndexError(
            f"Reference frame index {reference_frame_index} is out of range for "
            f"sequence with {pair_count} synchronized pairs"
        )
    current_frame_index = reference_frame_index + frame_gap
    if current_frame_index >= pair_count:
        raise IndexError(
            f"Reference frame index {reference_frame_index} with frame_gap "
            f"{frame_gap} requires current frame index {current_frame_index}, "
            f"but sequence has {pair_count} synchronized pairs"
        )

    ref_frame = sequence.frames[reference_frame_index]
    cur_frame = sequence.frames[current_frame_index]

    ref_gt, ref_cam_pose = ground_truth.interpolate_camera_pose(
        ref_frame.timestamp_ns, calibration.body_to_sensor_transform
    )
    cur_gt, cur_cam_pose = ground_truth.interpolate_camera_pose(
        cur_frame.timestamp_ns, calibration.body_to_sensor_transform
    )

    gt_relative_pose = compute_ground_truth_relative_camera_pose(
        ref_cam_pose,
        cur_cam_pose,
        translation_epsilon_m=translation_epsilon_m,
    )

    ref_image = sequence.load_left_image(reference_frame_index)
    cur_image = sequence.load_left_image(current_frame_index)
    match_result, estimated_pose = estimate_calibrated_relative_pose(
        ref_image,
        cur_image,
        calibration,
        max_features=max_features,
        ratio_threshold=ratio_threshold,
        fundamental_ransac_threshold=fundamental_ransac_threshold,
        fundamental_ransac_confidence=fundamental_ransac_confidence,
        essential_ransac_probability=essential_ransac_probability,
        essential_ransac_threshold=essential_ransac_threshold,
        min_correspondences=min_correspondences,
        minimum_cheirality_count=minimum_cheirality_count,
        minimum_cheirality_ratio=minimum_cheirality_ratio,
    )

    pose_error = compare_relative_pose(
        estimated_pose,
        gt_relative_pose,
        translation_epsilon_m=translation_epsilon_m,
    )

    return RelativePoseEvaluation(
        estimated_result=estimated_pose,
        ground_truth_relative_pose=gt_relative_pose,
        relative_pose_error=pose_error,
        pose_quality_valid=estimated_pose.statistics.pose_quality_valid,
        pose_quality_reason=estimated_pose.statistics.pose_quality_reason,
        frame_gap=frame_gap,
        reference_frame_index=reference_frame_index,
        current_frame_index=current_frame_index,
        reference_timestamp_ns=ref_frame.timestamp_ns,
        current_timestamp_ns=cur_frame.timestamp_ns,
        reference_ground_truth=ref_gt,
        current_ground_truth=cur_gt,
        reference_camera_pose=ref_cam_pose,
        current_camera_pose=cur_cam_pose,
        match_result=match_result,
    )


def evaluate_sequence_relative_pose(
    sequence_root: Path,
    reference_frame_index: int,
    *,
    frame_gap: int = 1,
    translation_epsilon_m: float = DEFAULT_TRANSLATION_EPSILON_M,
    max_features: int = 1000,
    ratio_threshold: float = 0.75,
    fundamental_ransac_threshold: float = 1.0,
    fundamental_ransac_confidence: float = 0.99,
    essential_ransac_probability: float = 0.999,
    essential_ransac_threshold: float = 1e-3,
    min_correspondences: int = MIN_CALIBRATED_CORRESPONDENCES,
    minimum_cheirality_count: int = DEFAULT_MIN_CHEIRALITY_COUNT,
    minimum_cheirality_ratio: float = DEFAULT_MIN_CHEIRALITY_RATIO,
) -> RelativePoseEvaluation:
    """High-level Milestone 4A relative-pose evaluation for a single EuRoC pair."""
    if not isinstance(sequence_root, Path):
        sequence_root = Path(sequence_root)
    sequence = EurocSequence(sequence_root)
    calibration = load_cam0_calibration(sequence_root)
    ground_truth = EuRoCGroundTruthSequence(sequence_root)

    return evaluate_frame_pair_with_loaded_context(
        sequence,
        calibration,
        ground_truth,
        reference_frame_index,
        frame_gap=frame_gap,
        translation_epsilon_m=translation_epsilon_m,
        max_features=max_features,
        ratio_threshold=ratio_threshold,
        fundamental_ransac_threshold=fundamental_ransac_threshold,
        fundamental_ransac_confidence=fundamental_ransac_confidence,
        essential_ransac_probability=essential_ransac_probability,
        essential_ransac_threshold=essential_ransac_threshold,
        min_correspondences=min_correspondences,
        minimum_cheirality_count=minimum_cheirality_count,
        minimum_cheirality_ratio=minimum_cheirality_ratio,
    )
