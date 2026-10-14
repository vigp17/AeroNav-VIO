"""Unit tests for body-to-cam0 transform composition and relative-pose evaluation."""

import math

import numpy as np
import pytest

from aeronav_vio.evaluation.relative_pose import (
    compare_relative_pose,
    compute_ground_truth_relative_camera_pose,
    compute_rotation_error_degrees,
    compute_translation_direction_error_degrees,
)
from aeronav_vio.geometry.relative_pose import (
    RelativePoseResult,
    RelativePoseStatistics,
)
from aeronav_vio.ground_truth.euroc import (
    GroundTruthPose,
    body_pose_to_camera_pose,
    invert_rigid_transform,
    normalize_quaternion_wxyz,
    quaternion_wxyz_to_rotation_matrix,
    validate_homogeneous_transform,
)


def _make_body_pose(
    timestamp_ns: int,
    position: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> GroundTruthPose:
    q = normalize_quaternion_wxyz(quat_wxyz)
    rot = quaternion_wxyz_to_rotation_matrix(q)
    pos = np.array(position, dtype=np.float64)
    pos.setflags(write=False)
    return GroundTruthPose(
        timestamp_ns=timestamp_ns,
        position_world_body=pos,
        quaternion_world_body_wxyz=q,
        rotation_world_body=rot,
    )


def _make_se3(
    rotation: np.ndarray, translation: list[float] | np.ndarray
) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rotation
    mat[:3, 3] = np.asarray(translation, dtype=np.float64)
    return mat


def _rot_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def test_identity_translated_and_rotated_body_pose_to_camera_pose() -> None:
    # 1. Identity body pose with nonidentity camera extrinsic T_BS
    r_bc = _rot_x(math.pi / 2.0)
    p_bc = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    t_bs = _make_se3(r_bc, p_bc)
    t_bs_before = t_bs.copy()

    identity_body = _make_body_pose(100, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    cam_pose_1 = body_pose_to_camera_pose(identity_body, t_bs)
    assert np.array_equal(t_bs, t_bs_before)
    np.testing.assert_allclose(cam_pose_1.rotation_world_camera, r_bc, atol=1e-12)
    np.testing.assert_allclose(cam_pose_1.position_world_camera, p_bc, atol=1e-12)
    np.testing.assert_allclose(cam_pose_1.transform_world_camera, t_bs, atol=1e-12)

    # 2. Translated body pose with identity rotation
    trans_body = _make_body_pose(200, (1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0))
    cam_pose_2 = body_pose_to_camera_pose(trans_body, t_bs)
    np.testing.assert_allclose(cam_pose_2.rotation_world_camera, r_bc, atol=1e-12)
    np.testing.assert_allclose(
        cam_pose_2.position_world_camera, [1.1, 1.8, 3.3], atol=1e-12
    )

    # 3. Rotated body pose (90 deg about Z) + translated body + non-commuting extrinsic
    s = math.sqrt(0.5)
    rot_body = _make_body_pose(300, (1.0, 0.0, 2.0), (s, 0.0, 0.0, s))
    cam_pose_3 = body_pose_to_camera_pose(rot_body, t_bs)

    # R_WB is 90 deg about Z: R_WB @ [0.1, -0.2, 0.3] = [0.2, 0.1, 0.3]
    expected_r_wc = _rot_z(math.pi / 2.0) @ r_bc
    expected_p_wc = np.array([1.0 + 0.2, 0.0 + 0.1, 2.0 + 0.3], dtype=np.float64)
    np.testing.assert_allclose(
        cam_pose_3.rotation_world_camera, expected_r_wc, atol=1e-12
    )
    np.testing.assert_allclose(
        cam_pose_3.position_world_camera, expected_p_wc, atol=1e-12
    )

    # Detect reversed multiplication order T_BS @ T_WB vs T_WB @ T_BS
    t_wb = _make_se3(rot_body.rotation_world_body, rot_body.position_world_body)
    reversed_product = t_bs @ t_wb
    assert not np.allclose(cam_pose_3.transform_world_camera, reversed_product)


def test_transform_inversion_and_validation() -> None:
    r = _rot_y(0.35) @ _rot_z(-0.2)
    t = _make_se3(r, [1.5, -0.75, 2.25])
    t_inv = invert_rigid_transform(t)
    np.testing.assert_allclose(t @ t_inv, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(t_inv @ t, np.eye(4), atol=1e-12)

    bad_row = t.copy()
    bad_row[3, 3] = 2.0
    with pytest.raises(ValueError, match="Invalid final row"):
        validate_homogeneous_transform(bad_row)


def test_ground_truth_relative_camera_pose_matches_milestone3_convention() -> None:
    # Construct two non-commuting body poses and a non-identity camera extrinsic
    r_bc = _rot_x(0.4) @ _rot_y(-0.25)
    t_bs = _make_se3(r_bc, [0.05, -0.04, 0.02])

    pose_ref = _make_body_pose(
        1000, (1.2, -0.5, 0.8), (0.96, 0.1, -0.2, 0.15)
    )
    pose_cur = _make_body_pose(
        2000, (1.6, -0.2, 1.1), (0.92, 0.18, -0.12, 0.32)
    )

    cam_ref = body_pose_to_camera_pose(pose_ref, t_bs)
    cam_cur = body_pose_to_camera_pose(pose_cur, t_bs)

    gt_rel = compute_ground_truth_relative_camera_pose(cam_ref, cam_cur)

    # Verify against a 3D world landmark X_W projected into both camera frames:
    # X_W = R_WC_ref @ X_C_ref + p_WC_ref  and  X_W = R_WC_cur @ X_C_cur + p_WC_cur
    landmark_world = np.array([3.5, 1.2, 4.8], dtype=np.float64)
    x_c_ref = cam_ref.rotation_world_camera.T @ (
        landmark_world - cam_ref.position_world_camera
    )
    x_c_cur_true = cam_cur.rotation_world_camera.T @ (
        landmark_world - cam_cur.position_world_camera
    )
    x_c_cur_from_rel = (
        gt_rel.rotation_current_reference @ x_c_ref
        + gt_rel.translation_current_reference
    )
    np.testing.assert_allclose(x_c_cur_from_rel, x_c_cur_true, atol=1e-12)

    # Verify that reversing relative order (inv(T_WC_ref) @ T_WC_cur)
    # gives a different result
    reversed_rel = (
        invert_rigid_transform(cam_ref.transform_world_camera)
        @ cam_cur.transform_world_camera
    )
    assert not np.allclose(gt_rel.transform_current_reference, reversed_rel)


def test_rotation_and_translation_direction_error_metrics() -> None:
    r_gt = _rot_z(math.radians(15.0))
    # Identical rotation -> 0 deg error
    assert compute_rotation_error_degrees(r_gt, r_gt) == pytest.approx(0.0, abs=1e-7)

    # Known 10 deg rotation difference -> 10 deg error
    r_est = _rot_x(math.radians(10.0)) @ r_gt
    assert compute_rotation_error_degrees(r_est, r_gt) == pytest.approx(
        10.0, abs=1e-6
    )

    # Matching unit translation directions -> 0 deg error
    t_gt = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    assert compute_translation_direction_error_degrees(t_gt, t_gt) == pytest.approx(
        0.0, abs=1e-7
    )

    # Known 60 deg angle between unit translation directions
    t_60 = np.array([math.sin(math.pi / 3.0), 0.0, math.cos(math.pi / 3.0)])
    assert compute_translation_direction_error_degrees(t_60, t_gt) == pytest.approx(
        60.0, abs=1e-6
    )

    # Opposite translation direction (-t_gt) -> 180 deg error (sign is NOT ignored)
    assert compute_translation_direction_error_degrees(-t_gt, t_gt) == pytest.approx(
        180.0, abs=1e-6
    )


def _dummy_pose_result(
    *,
    success: bool,
    rotation: np.ndarray | None,
    translation_dir: np.ndarray | None,
    quality_valid: bool = True,
) -> RelativePoseResult:
    pts = np.zeros((10, 2), dtype=np.float64)
    mask = np.ones(10, dtype=bool)
    return RelativePoseResult(
        success=success,
        failure_reason=None if success else "Estimation failed",
        essential_matrix=np.eye(3, dtype=np.float64) if success else None,
        rotation_matrix=rotation,
        translation_direction=translation_dir,
        essential_inlier_mask=mask,
        cheirality_inlier_mask=mask,
        normalized_reference_points=pts,
        normalized_current_points=pts,
        statistics=RelativePoseStatistics(
            input_correspondence_count=10,
            essential_inlier_count=10 if success else 0,
            cheirality_inlier_count=10 if success else 0,
            essential_inlier_ratio=1.0 if success else 0.0,
            cheirality_inlier_ratio=1.0 if success else 0.0,
            rotation_angle_degrees=0.0 if success else None,
            translation_norm=1.0 if success else None,
            pose_quality_valid=quality_valid,
            pose_quality_reason="ok" if quality_valid else "low cheirality",
        ),
    )


def test_near_zero_ground_truth_translation_marks_direction_undefined() -> None:
    t_bs = np.eye(4, dtype=np.float64)
    pose_ref = _make_body_pose(100, (1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0))
    # Pure rotation with zero translation
    s = math.sin(math.radians(2.5))
    c = math.cos(math.radians(2.5))
    pose_cur = _make_body_pose(200, (1.0, 2.0, 3.0), (c, 0.0, 0.0, s))

    cam_ref = body_pose_to_camera_pose(pose_ref, t_bs)
    cam_cur = body_pose_to_camera_pose(pose_cur, t_bs)

    gt_rel = compute_ground_truth_relative_camera_pose(
        cam_ref, cam_cur, translation_epsilon_m=1e-4
    )
    assert gt_rel.translation_magnitude_m == pytest.approx(0.0, abs=1e-12)
    assert gt_rel.translation_direction_defined is False
    assert gt_rel.translation_direction_current_reference is None
    assert gt_rel.translation_direction_undefined_reason is not None

    est_pose = _dummy_pose_result(
        success=True,
        rotation=gt_rel.rotation_current_reference,
        translation_dir=np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )
    err = compare_relative_pose(est_pose, gt_rel, translation_epsilon_m=1e-4)
    assert err.rotation_error_degrees == pytest.approx(0.0, abs=1e-7)
    assert err.translation_direction_defined is False
    assert err.translation_direction_error_degrees is None
    assert (
        err.failure_reason is not None
        and "below near-zero epsilon" in err.failure_reason
    )


def test_compare_relative_pose_rejects_nonfinite_estimated_pose() -> None:
    t_bs = np.eye(4, dtype=np.float64)
    cam_ref = body_pose_to_camera_pose(
        _make_body_pose(100, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)), t_bs
    )
    cam_cur = body_pose_to_camera_pose(
        _make_body_pose(200, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0)), t_bs
    )
    gt_rel = compute_ground_truth_relative_camera_pose(cam_ref, cam_cur)

    bad_rot = np.eye(3, dtype=np.float64)
    bad_rot[0, 0] = float("nan")
    bad_est = _dummy_pose_result(
        success=True,
        rotation=bad_rot,
        translation_dir=np.array([0.0, 0.0, -1.0], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="nonfinite"):
        compare_relative_pose(bad_est, gt_rel)

    # Also verify failed estimated pose produces None errors without raising
    failed_est = _dummy_pose_result(
        success=False, rotation=None, translation_dir=None, quality_valid=False
    )
    failed_err = compare_relative_pose(failed_est, gt_rel)
    assert failed_err.rotation_error_degrees is None
    assert failed_err.translation_direction_error_degrees is None
    assert failed_err.failure_reason == "Estimation failed"
