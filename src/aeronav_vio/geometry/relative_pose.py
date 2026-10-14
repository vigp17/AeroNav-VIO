"""Calibrated essential-matrix and relative-pose estimation for EuRoC cam0."""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from aeronav_vio.calibration.euroc import (
    CameraCalibration,
    validate_image_calibration_resolution,
)
from aeronav_vio.features.orb import TemporalMatchResult, match_temporal_pair

MIN_CALIBRATED_CORRESPONDENCES = 8
DEFAULT_MIN_CHEIRALITY_COUNT = 30
DEFAULT_MIN_CHEIRALITY_RATIO = 0.15
TRANSFORM_CONVENTION = (
    "X_current = R_current_reference @ X_reference + t_current_reference"
)
TRANSLATION_SCALE_STATUS = (
    "unknown (unit direction vector only; monocular two-view pose recovery "
    "cannot recover metric translation scale)"
)
_ROTATION_ORTHONORMAL_ATOL = 1e-4
_ROTATION_DET_ATOL = 1e-4
_TRANSLATION_MIN_NORM = 1e-6


@dataclass(frozen=True)
class RelativePoseStatistics:
    """Inlier counts, ratios, and angular diagnostics for two-view pose recovery."""

    input_correspondence_count: int
    essential_inlier_count: int
    cheirality_inlier_count: int
    essential_inlier_ratio: float
    cheirality_inlier_ratio: float
    rotation_angle_degrees: float | None
    translation_norm: float | None
    pose_quality_valid: bool = False
    pose_quality_reason: str = ""
    minimum_cheirality_count: int = DEFAULT_MIN_CHEIRALITY_COUNT
    minimum_cheirality_ratio: float = DEFAULT_MIN_CHEIRALITY_RATIO


@dataclass(frozen=True)
class RelativePoseResult:
    """Structured output of calibrated two-view relative pose estimation."""

    success: bool
    failure_reason: str | None
    essential_matrix: np.ndarray | None
    rotation_matrix: np.ndarray | None
    translation_direction: np.ndarray | None
    essential_inlier_mask: np.ndarray
    cheirality_inlier_mask: np.ndarray
    normalized_reference_points: np.ndarray
    normalized_current_points: np.ndarray
    statistics: RelativePoseStatistics


def _readonly_float64(array: np.ndarray) -> np.ndarray:
    """Return a read-only float64 copy of array."""
    out = np.array(array, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


def _readonly_bool(array: np.ndarray) -> np.ndarray:
    """Return a read-only 1D bool copy of array."""
    out = np.array(array, dtype=bool, copy=True).reshape(-1)
    out.setflags(write=False)
    return out


def _validate_quality_thresholds(
    minimum_cheirality_count: int,
    minimum_cheirality_ratio: float,
) -> None:
    """Validate configurable cheirality pose-quality thresholds."""
    if isinstance(minimum_cheirality_count, bool) or not isinstance(
        minimum_cheirality_count, int
    ):
        raise TypeError("minimum_cheirality_count must be an integer")
    if minimum_cheirality_count < 1:
        raise ValueError(
            "minimum_cheirality_count must be >= 1, got "
            f"{minimum_cheirality_count}"
        )
    if (
        isinstance(minimum_cheirality_ratio, bool)
        or not isinstance(minimum_cheirality_ratio, (int, float))
        or not math.isfinite(float(minimum_cheirality_ratio))
        or not (0.0 < float(minimum_cheirality_ratio) <= 1.0)
    ):
        raise ValueError(
            "minimum_cheirality_ratio must be in (0.0, 1.0], got "
            f"{minimum_cheirality_ratio!r}"
        )


def evaluate_pose_quality(
    cheirality_inlier_count: int,
    cheirality_inlier_ratio: float,
    *,
    minimum_cheirality_count: int = DEFAULT_MIN_CHEIRALITY_COUNT,
    minimum_cheirality_ratio: float = DEFAULT_MIN_CHEIRALITY_RATIO,
) -> tuple[bool, str]:
    """Evaluate whether a recovered pose meets minimum cheirality support thresholds."""
    _validate_quality_thresholds(
        minimum_cheirality_count, minimum_cheirality_ratio
    )
    min_ratio = float(minimum_cheirality_ratio)

    failures: list[str] = []
    if cheirality_inlier_count < minimum_cheirality_count:
        failures.append(
            f"cheirality_inlier_count ({cheirality_inlier_count}) < "
            f"minimum_cheirality_count ({minimum_cheirality_count})"
        )
    if cheirality_inlier_ratio < min_ratio:
        failures.append(
            f"cheirality_inlier_ratio ({cheirality_inlier_ratio:.4f}) < "
            f"minimum_cheirality_ratio ({min_ratio:.4f})"
        )

    if failures:
        return (
            False,
            "Low-confidence relative pose (insufficient cheirality support): "
            + "; ".join(failures),
        )
    return (
        True,
        (
            f"Cheirality support meets quality thresholds "
            f"(count={cheirality_inlier_count} >= {minimum_cheirality_count}, "
            f"ratio={cheirality_inlier_ratio:.4f} >= {min_ratio:.4f})"
        ),
    )


def _failed_pose_result(
    reason: str,
    normalized_reference_points: np.ndarray,
    normalized_current_points: np.ndarray,
    *,
    essential_inlier_mask: np.ndarray | None = None,
    cheirality_inlier_mask: np.ndarray | None = None,
    minimum_cheirality_count: int = DEFAULT_MIN_CHEIRALITY_COUNT,
    minimum_cheirality_ratio: float = DEFAULT_MIN_CHEIRALITY_RATIO,
) -> RelativePoseResult:
    """Construct a RelativePoseResult representing a failed estimation."""
    count = int(normalized_reference_points.shape[0])
    if essential_inlier_mask is None:
        e_mask = _readonly_bool(np.zeros(count, dtype=bool))
    else:
        e_mask = _readonly_bool(essential_inlier_mask)

    if cheirality_inlier_mask is None:
        c_mask = _readonly_bool(np.zeros(count, dtype=bool))
    else:
        c_mask = _readonly_bool(cheirality_inlier_mask)

    e_count = int(np.count_nonzero(e_mask))
    c_count = int(np.count_nonzero(c_mask))
    e_ratio = e_count / count if count > 0 else 0.0
    c_ratio = c_count / count if count > 0 else 0.0

    stats = RelativePoseStatistics(
        input_correspondence_count=count,
        essential_inlier_count=e_count,
        cheirality_inlier_count=c_count,
        essential_inlier_ratio=e_ratio,
        cheirality_inlier_ratio=c_ratio,
        rotation_angle_degrees=None,
        translation_norm=None,
        pose_quality_valid=False,
        pose_quality_reason=f"Pose recovery failed: {reason}",
        minimum_cheirality_count=minimum_cheirality_count,
        minimum_cheirality_ratio=float(minimum_cheirality_ratio),
    )
    return RelativePoseResult(
        success=False,
        failure_reason=reason,
        essential_matrix=None,
        rotation_matrix=None,
        translation_direction=None,
        essential_inlier_mask=e_mask,
        cheirality_inlier_mask=c_mask,
        normalized_reference_points=_readonly_float64(normalized_reference_points),
        normalized_current_points=_readonly_float64(normalized_current_points),
        statistics=stats,
    )


def _validate_point_pair_arrays(
    reference_points: np.ndarray,
    current_points: np.ndarray,
    *,
    ref_label: str = "reference_points",
    cur_label: str = "current_points",
) -> tuple[np.ndarray, np.ndarray]:
    """Validate that reference and current point arrays are finite with shape (N, 2)."""
    if not isinstance(reference_points, np.ndarray):
        raise TypeError(f"Expected {ref_label} to be a NumPy array")
    if not isinstance(current_points, np.ndarray):
        raise TypeError(f"Expected {cur_label} to be a NumPy array")

    if reference_points.ndim != 2 or reference_points.shape[1] != 2:
        raise ValueError(
            f"Expected {ref_label} with shape (N, 2), got {reference_points.shape}"
        )
    if current_points.ndim != 2 or current_points.shape[1] != 2:
        raise ValueError(
            f"Expected {cur_label} with shape (N, 2), got {current_points.shape}"
        )
    if reference_points.shape[0] != current_points.shape[0]:
        raise ValueError(
            f"Unequal correspondence counts: {ref_label} has "
            f"{reference_points.shape[0]} points, {cur_label} has "
            f"{current_points.shape[0]} points"
        )

    try:
        ref_float = np.asarray(reference_points, dtype=np.float64)
        cur_float = np.asarray(current_points, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Point coordinates must be numeric") from error

    if not np.all(np.isfinite(ref_float)):
        raise ValueError(f"{ref_label} contains nonfinite coordinate values")
    if not np.all(np.isfinite(cur_float)):
        raise ValueError(f"{cur_label} contains nonfinite coordinate values")

    return ref_float, cur_float


def extract_inlier_correspondences(
    match_result: TemporalMatchResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (N, 2) float64 reference and current pixel coordinates from inliers."""
    inlier_matches = match_result.geometric_inlier_matches
    if not inlier_matches:
        empty_ref = _readonly_float64(np.empty((0, 2), dtype=np.float64))
        empty_cur = _readonly_float64(np.empty((0, 2), dtype=np.float64))
        return empty_ref, empty_cur

    ref_keypoints = match_result.reference_features.keypoints
    cur_keypoints = match_result.current_features.keypoints
    ref_count = len(ref_keypoints)
    cur_count = len(cur_keypoints)

    ref_coords: list[tuple[float, float]] = []
    cur_coords: list[tuple[float, float]] = []
    for match in inlier_matches:
        if match.queryIdx < 0 or match.queryIdx >= ref_count:
            raise IndexError(
                f"Inlier match queryIdx {match.queryIdx} out of range for "
                f"{ref_count} reference keypoints"
            )
        if match.trainIdx < 0 or match.trainIdx >= cur_count:
            raise IndexError(
                f"Inlier match trainIdx {match.trainIdx} out of range for "
                f"{cur_count} current keypoints"
            )
        ref_pt = ref_keypoints[match.queryIdx].pt
        cur_pt = cur_keypoints[match.trainIdx].pt
        ref_coords.append((float(ref_pt[0]), float(ref_pt[1])))
        cur_coords.append((float(cur_pt[0]), float(cur_pt[1])))

    ref_array = np.array(ref_coords, dtype=np.float64).reshape((-1, 2))
    cur_array = np.array(cur_coords, dtype=np.float64).reshape((-1, 2))
    _validate_point_pair_arrays(
        ref_array,
        cur_array,
        ref_label="extracted reference inlier coordinates",
        cur_label="extracted current inlier coordinates",
    )
    ref_array.setflags(write=False)
    cur_array.setflags(write=False)
    return ref_array, cur_array


def undistort_matched_points(
    reference_points: np.ndarray,
    current_points: np.ndarray,
    calibration: CameraCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    """Undistort pixel coordinates (u, v) to normalized camera coordinates (x, y)."""
    ref_float, cur_float = _validate_point_pair_arrays(
        reference_points, current_points
    )
    count = int(ref_float.shape[0])
    if count == 0:
        empty_ref = _readonly_float64(np.empty((0, 2), dtype=np.float64))
        empty_cur = _readonly_float64(np.empty((0, 2), dtype=np.float64))
        return empty_ref, empty_cur

    ref_reshaped = np.ascontiguousarray(ref_float.reshape((count, 1, 2)))
    cur_reshaped = np.ascontiguousarray(cur_float.reshape((count, 1, 2)))

    undistorted_ref = cv2.undistortPoints(
        ref_reshaped,
        calibration.camera_matrix,
        calibration.distortion_array,
    )
    undistorted_cur = cv2.undistortPoints(
        cur_reshaped,
        calibration.camera_matrix,
        calibration.distortion_array,
    )
    if undistorted_ref is None or undistorted_cur is None:
        raise ValueError("cv2.undistortPoints failed to return normalized coordinates")

    norm_ref = np.asarray(undistorted_ref, dtype=np.float64).reshape((count, 2))
    norm_cur = np.asarray(undistorted_cur, dtype=np.float64).reshape((count, 2))
    if not np.all(np.isfinite(norm_ref)) or not np.all(np.isfinite(norm_cur)):
        raise ValueError("Undistorted normalized coordinates contain nonfinite values")

    return _readonly_float64(norm_ref), _readonly_float64(norm_cur)


def validate_rotation_matrix(rotation_matrix: np.ndarray) -> float:
    """Validate that rotation_matrix is a finite 3x3 SO(3) matrix and return det(R)."""
    if not isinstance(rotation_matrix, np.ndarray):
        raise TypeError("Expected rotation_matrix to be a NumPy array")
    if rotation_matrix.shape != (3, 3):
        raise ValueError(
            f"Expected rotation_matrix shape (3, 3), got {rotation_matrix.shape}"
        )
    if not np.all(np.isfinite(rotation_matrix)):
        raise ValueError("rotation_matrix contains nonfinite values")

    ortho_err = float(
        np.max(np.abs(rotation_matrix.T @ rotation_matrix - np.eye(3)))
    )
    if ortho_err > _ROTATION_ORTHONORMAL_ATOL:
        raise ValueError(
            f"rotation_matrix is not orthonormal within tolerance "
            f"(max error={ortho_err:.3e})"
        )

    determinant = float(np.linalg.det(rotation_matrix))
    if abs(determinant - 1.0) > _ROTATION_DET_ATOL:
        raise ValueError(
            f"rotation_matrix determinant {determinant:.6f} is not close to +1"
        )
    return determinant


def compute_rotation_angle_degrees(rotation_matrix: np.ndarray) -> float:
    """Compute the rotation angle in [0, 180] degrees using clipped matrix trace."""
    validate_rotation_matrix(rotation_matrix)
    trace_val = float(np.trace(rotation_matrix))
    cos_theta = float(np.clip((trace_val - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def validate_essential_inlier_mask(
    mask: np.ndarray | list[int] | tuple[int, ...] | None,
    correspondence_count: int,
) -> np.ndarray:
    """Validate an essential-matrix RANSAC mask and return a 1D bool array."""
    if correspondence_count < 0:
        raise ValueError("correspondence_count must be non-negative")
    if correspondence_count == 0:
        return _readonly_bool(np.zeros(0, dtype=bool))
    if mask is None:
        raise ValueError("Essential-matrix RANSAC mask is None")

    mask_array = np.asarray(mask)
    if mask_array.ndim > 2 or (
        mask_array.ndim == 2 and mask_array.shape[1] != 1
    ):
        raise ValueError(
            f"Invalid RANSAC mask shape {mask_array.shape}: expected (N,) or (N, 1)"
        )
    flat_mask = mask_array.reshape(-1)
    if flat_mask.shape[0] != correspondence_count:
        raise ValueError(
            f"Essential-matrix mask length {flat_mask.shape[0]} does not match "
            f"correspondence count {correspondence_count}"
        )
    if not np.all(np.isfinite(flat_mask)):
        raise ValueError("Essential-matrix mask contains nonfinite values")

    return _readonly_bool(flat_mask != 0)


def split_essential_matrix_candidates(
    essential_matrix: np.ndarray | None,
) -> tuple[np.ndarray, ...]:
    """Validate and split a 3x3 or vertically stacked (3k x 3) essential matrix."""
    if essential_matrix is None:
        return ()
    if not isinstance(essential_matrix, np.ndarray):
        raise TypeError("Expected essential_matrix to be a NumPy array or None")
    if (
        essential_matrix.ndim != 2
        or essential_matrix.shape[1] != 3
        or essential_matrix.shape[0] < 3
        or essential_matrix.shape[0] % 3 != 0
    ):
        raise ValueError(
            f"Malformed essential matrix shape {essential_matrix.shape}: "
            "expected (3 * k, 3) with k >= 1"
        )
    if not np.all(np.isfinite(essential_matrix)):
        raise ValueError("Essential matrix contains nonfinite values")

    num_candidates = essential_matrix.shape[0] // 3
    candidates: list[np.ndarray] = []
    for idx in range(num_candidates):
        candidate = _readonly_float64(essential_matrix[3 * idx : 3 * (idx + 1), :])
        if float(np.linalg.norm(candidate)) <= 1e-12:
            raise ValueError("Essential matrix candidate has near-zero norm")
        candidates.append(candidate)
    return tuple(candidates)


def estimate_essential_matrix(
    normalized_reference_points: np.ndarray,
    normalized_current_points: np.ndarray,
    *,
    ransac_probability: float = 0.999,
    ransac_threshold: float = 1e-3,
    min_correspondences: int = MIN_CALIBRATED_CORRESPONDENCES,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, str | None]:
    """Estimate essential-matrix candidate(s) and inlier mask from normalized points."""
    ref_float, cur_float = _validate_point_pair_arrays(
        normalized_reference_points,
        normalized_current_points,
        ref_label="normalized_reference_points",
        cur_label="normalized_current_points",
    )
    if not math.isfinite(ransac_probability) or not (0.0 < ransac_probability < 1.0):
        raise ValueError(
            f"ransac_probability must be strictly between 0 and 1, "
            f"got {ransac_probability}"
        )
    if not math.isfinite(ransac_threshold) or ransac_threshold <= 0.0:
        raise ValueError(
            f"ransac_threshold must be finite and positive, got {ransac_threshold}"
        )
    if min_correspondences < 5:
        raise ValueError(
            f"min_correspondences must be at least 5, got {min_correspondences}"
        )

    count = int(ref_float.shape[0])
    empty_mask = _readonly_bool(np.zeros(count, dtype=bool))
    if count < min_correspondences:
        return (
            (),
            empty_mask,
            (
                f"Insufficient correspondences for essential-matrix estimation: "
                f"got {count}, require at least {min_correspondences}"
            ),
        )

    if (
        float(np.max(np.ptp(ref_float, axis=0))) <= 1e-9
        or float(np.max(np.ptp(cur_float, axis=0))) <= 1e-9
    ):
        return (
            (),
            empty_mask,
            "Degenerate correspondence geometry: points have near-zero spatial spread",
        )

    identity_k = np.eye(3, dtype=np.float64)
    essential_raw, mask_raw = cv2.findEssentialMat(
        np.ascontiguousarray(ref_float),
        np.ascontiguousarray(cur_float),
        cameraMatrix=identity_k,
        method=cv2.RANSAC,
        prob=ransac_probability,
        threshold=ransac_threshold,
    )

    if essential_raw is None or mask_raw is None:
        return (
            (),
            empty_mask,
            "cv2.findEssentialMat failed to estimate a valid essential matrix",
        )

    try:
        inlier_mask = validate_essential_inlier_mask(mask_raw, count)
        candidates = split_essential_matrix_candidates(essential_raw)
    except ValueError as error:
        return (), empty_mask, f"Invalid essential-matrix output: {error}"

    if not candidates or int(np.count_nonzero(inlier_mask)) == 0:
        return (
            (),
            inlier_mask,
            "Essential-matrix estimation produced zero RANSAC inliers",
        )

    return candidates, inlier_mask, None


def recover_pose_from_essential_matrix(
    essential_matrix: np.ndarray,
    normalized_reference_points: np.ndarray,
    normalized_current_points: np.ndarray,
    essential_inlier_mask: np.ndarray | list[bool] | tuple[bool, ...],
    *,
    minimum_cheirality_count: int = DEFAULT_MIN_CHEIRALITY_COUNT,
    minimum_cheirality_ratio: float = DEFAULT_MIN_CHEIRALITY_RATIO,
) -> RelativePoseResult:
    """Recover relative rotation and unit translation direction from E(s)."""
    _validate_quality_thresholds(
        minimum_cheirality_count, minimum_cheirality_ratio
    )
    ref_float, cur_float = _validate_point_pair_arrays(
        normalized_reference_points,
        normalized_current_points,
        ref_label="normalized_reference_points",
        cur_label="normalized_current_points",
    )
    count = int(ref_float.shape[0])
    inlier_mask = validate_essential_inlier_mask(essential_inlier_mask, count)

    try:
        candidates = split_essential_matrix_candidates(essential_matrix)
    except (TypeError, ValueError) as error:
        return _failed_pose_result(
            f"Invalid essential matrix: {error}",
            ref_float,
            cur_float,
            essential_inlier_mask=inlier_mask,
            minimum_cheirality_count=minimum_cheirality_count,
            minimum_cheirality_ratio=minimum_cheirality_ratio,
        )

    if not candidates:
        return _failed_pose_result(
            "No essential-matrix candidate supplied",
            ref_float,
            cur_float,
            essential_inlier_mask=inlier_mask,
            minimum_cheirality_count=minimum_cheirality_count,
            minimum_cheirality_ratio=minimum_cheirality_ratio,
        )

    essential_inlier_count = int(np.count_nonzero(inlier_mask))
    if essential_inlier_count == 0:
        return _failed_pose_result(
            "Cannot recover relative pose with zero essential-matrix inliers",
            ref_float,
            cur_float,
            essential_inlier_mask=inlier_mask,
            minimum_cheirality_count=minimum_cheirality_count,
            minimum_cheirality_ratio=minimum_cheirality_ratio,
        )

    identity_k = np.eye(3, dtype=np.float64)
    best_result: RelativePoseResult | None = None
    best_cheirality_count = -1
    last_failure_reason = "No essential-matrix candidate produced a valid relative pose"

    for candidate in candidates:
        mask_uint8 = np.where(inlier_mask, 1, 0).astype(np.uint8).reshape((-1, 1))
        try:
            _, rot_raw, trans_raw, pose_mask_raw = cv2.recoverPose(
                candidate,
                np.ascontiguousarray(ref_float),
                np.ascontiguousarray(cur_float),
                cameraMatrix=identity_k,
                mask=mask_uint8,
            )
        except cv2.error as error:
            last_failure_reason = f"cv2.recoverPose failed: {error}"
            continue

        if rot_raw is None or trans_raw is None or pose_mask_raw is None:
            last_failure_reason = "cv2.recoverPose returned None output"
            continue

        rot_arr = np.asarray(rot_raw, dtype=np.float64)
        trans_arr = np.asarray(trans_raw, dtype=np.float64).reshape(-1)
        if trans_arr.shape != (3,) or not np.all(np.isfinite(trans_arr)):
            last_failure_reason = (
                f"Invalid recovered translation shape {trans_arr.shape} or nonfinite"
            )
            continue

        raw_norm = float(np.linalg.norm(trans_arr))
        if not math.isfinite(raw_norm) or raw_norm <= _TRANSLATION_MIN_NORM:
            last_failure_reason = (
                f"Recovered translation vector has near-zero norm ({raw_norm:.3e})"
            )
            continue

        unit_translation = _readonly_float64(trans_arr / raw_norm)
        unit_norm = float(np.linalg.norm(unit_translation))
        if not np.isclose(unit_norm, 1.0, atol=1e-6):
            last_failure_reason = (
                f"Normalized translation norm {unit_norm:.6f} is not unit length"
            )
            continue

        try:
            angle_deg = compute_rotation_angle_degrees(rot_arr)
            cheirality_raw = validate_essential_inlier_mask(pose_mask_raw, count)
        except (TypeError, ValueError) as error:
            last_failure_reason = str(error)
            continue

        cheirality_mask = _readonly_bool(inlier_mask & cheirality_raw)
        cheirality_count = int(np.count_nonzero(cheirality_mask))
        if cheirality_count == 0:
            last_failure_reason = (
                "Zero correspondences passed cheirality check in front of both cameras"
            )
            continue

        essential_ratio = essential_inlier_count / count if count > 0 else 0.0
        cheirality_ratio = cheirality_count / count if count > 0 else 0.0
        quality_valid, quality_reason = evaluate_pose_quality(
            cheirality_count,
            cheirality_ratio,
            minimum_cheirality_count=minimum_cheirality_count,
            minimum_cheirality_ratio=minimum_cheirality_ratio,
        )

        candidate_result = RelativePoseResult(
            success=True,
            failure_reason=None,
            essential_matrix=candidate,
            rotation_matrix=_readonly_float64(rot_arr),
            translation_direction=unit_translation,
            essential_inlier_mask=inlier_mask,
            cheirality_inlier_mask=cheirality_mask,
            normalized_reference_points=_readonly_float64(ref_float),
            normalized_current_points=_readonly_float64(cur_float),
            statistics=RelativePoseStatistics(
                input_correspondence_count=count,
                essential_inlier_count=essential_inlier_count,
                cheirality_inlier_count=cheirality_count,
                essential_inlier_ratio=essential_ratio,
                cheirality_inlier_ratio=cheirality_ratio,
                rotation_angle_degrees=angle_deg,
                translation_norm=unit_norm,
                pose_quality_valid=quality_valid,
                pose_quality_reason=quality_reason,
                minimum_cheirality_count=minimum_cheirality_count,
                minimum_cheirality_ratio=float(minimum_cheirality_ratio),
            ),
        )

        # Deterministic selection rule: strictly higher cheirality-inlier count wins;
        # ties preserve the earliest candidate index in the stacked matrix.
        if cheirality_count > best_cheirality_count:
            best_cheirality_count = cheirality_count
            best_result = candidate_result

    if best_result is None:
        return _failed_pose_result(
            last_failure_reason,
            ref_float,
            cur_float,
            essential_inlier_mask=inlier_mask,
            minimum_cheirality_count=minimum_cheirality_count,
            minimum_cheirality_ratio=minimum_cheirality_ratio,
        )
    return best_result


def estimate_relative_pose_from_normalized_points(
    normalized_reference_points: np.ndarray,
    normalized_current_points: np.ndarray,
    *,
    ransac_probability: float = 0.999,
    ransac_threshold: float = 1e-3,
    min_correspondences: int = MIN_CALIBRATED_CORRESPONDENCES,
    minimum_cheirality_count: int = DEFAULT_MIN_CHEIRALITY_COUNT,
    minimum_cheirality_ratio: float = DEFAULT_MIN_CHEIRALITY_RATIO,
) -> RelativePoseResult:
    """Estimate essential matrix and recover relative pose from normalized points."""
    _validate_quality_thresholds(
        minimum_cheirality_count, minimum_cheirality_ratio
    )
    ref_float, cur_float = _validate_point_pair_arrays(
        normalized_reference_points,
        normalized_current_points,
        ref_label="normalized_reference_points",
        cur_label="normalized_current_points",
    )
    candidates, inlier_mask, failure_reason = estimate_essential_matrix(
        ref_float,
        cur_float,
        ransac_probability=ransac_probability,
        ransac_threshold=ransac_threshold,
        min_correspondences=min_correspondences,
    )
    if failure_reason is not None or not candidates:
        return _failed_pose_result(
            failure_reason or "Essential-matrix estimation failed",
            ref_float,
            cur_float,
            essential_inlier_mask=inlier_mask,
            minimum_cheirality_count=minimum_cheirality_count,
            minimum_cheirality_ratio=minimum_cheirality_ratio,
        )

    stacked = np.vstack(candidates)
    return recover_pose_from_essential_matrix(
        stacked,
        ref_float,
        cur_float,
        inlier_mask,
        minimum_cheirality_count=minimum_cheirality_count,
        minimum_cheirality_ratio=minimum_cheirality_ratio,
    )


def estimate_calibrated_relative_pose(
    reference_image: np.ndarray,
    current_image: np.ndarray,
    calibration: CameraCalibration,
    *,
    max_features: int = 1000,
    ratio_threshold: float = 0.75,
    fundamental_ransac_threshold: float = 1.0,
    fundamental_ransac_confidence: float = 0.99,
    essential_ransac_probability: float = 0.999,
    essential_ransac_threshold: float = 1e-3,
    min_correspondences: int = MIN_CALIBRATED_CORRESPONDENCES,
    minimum_cheirality_count: int = DEFAULT_MIN_CHEIRALITY_COUNT,
    minimum_cheirality_ratio: float = DEFAULT_MIN_CHEIRALITY_RATIO,
) -> tuple[TemporalMatchResult, RelativePoseResult]:
    """Run Milestone 2 temporal matching and Milestone 3 calibrated pose recovery."""
    validate_image_calibration_resolution(
        calibration, reference_image, label="reference_image"
    )
    validate_image_calibration_resolution(
        calibration, current_image, label="current_image"
    )
    _validate_quality_thresholds(
        minimum_cheirality_count, minimum_cheirality_ratio
    )

    match_result = match_temporal_pair(
        reference_image,
        current_image,
        max_features=max_features,
        ratio_threshold=ratio_threshold,
        ransac_reprojection_threshold=fundamental_ransac_threshold,
        ransac_confidence=fundamental_ransac_confidence,
    )
    ref_pixels, cur_pixels = extract_inlier_correspondences(match_result)
    norm_ref, norm_cur = undistort_matched_points(
        ref_pixels, cur_pixels, calibration
    )
    pose_result = estimate_relative_pose_from_normalized_points(
        norm_ref,
        norm_cur,
        ransac_probability=essential_ransac_probability,
        ransac_threshold=essential_ransac_threshold,
        min_correspondences=min_correspondences,
        minimum_cheirality_count=minimum_cheirality_count,
        minimum_cheirality_ratio=minimum_cheirality_ratio,
    )
    return match_result, pose_result
