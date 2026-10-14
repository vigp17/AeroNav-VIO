"""Synthetic 3D geometry unit tests for calibrated relative pose estimation."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from aeronav_vio.calibration.euroc import CameraCalibration
from aeronav_vio.features.orb import (
    ORB_DESCRIPTOR_WIDTH,
    FeatureSet,
    MatchStatistics,
    TemporalMatchResult,
)
from aeronav_vio.geometry.relative_pose import (
    DEFAULT_MIN_CHEIRALITY_COUNT,
    DEFAULT_MIN_CHEIRALITY_RATIO,
    compute_rotation_angle_degrees,
    estimate_calibrated_relative_pose,
    estimate_essential_matrix,
    estimate_relative_pose_from_normalized_points,
    evaluate_pose_quality,
    extract_inlier_correspondences,
    recover_pose_from_essential_matrix,
    split_essential_matrix_candidates,
    undistort_matched_points,
    validate_essential_inlier_mask,
    validate_rotation_matrix,
)


def _synthetic_calibration(
    *, with_distortion: bool = True
) -> CameraCalibration:
    fx, fy, cx, cy = 460.0, 458.0, 376.0, 240.0
    coeffs = (
        (-0.28, 0.074, 0.00019, 1.7e-5)
        if with_distortion
        else (0.0, 0.0, 0.0, 0.0)
    )
    k_mat = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    k_mat.setflags(write=False)
    dist_arr = np.array(coeffs, dtype=np.float64)
    dist_arr.setflags(write=False)
    t_bs = np.eye(4, dtype=np.float64)
    t_bs.setflags(write=False)
    return CameraCalibration(
        sensor_label="cam0",
        sensor_type="camera",
        camera_model="pinhole",
        distortion_model="radial-tangential",
        resolution=(752, 480),
        rate_hz=20.0,
        intrinsics=(fx, fy, cx, cy),
        distortion_coefficients=coeffs,
        camera_matrix=k_mat,
        distortion_array=dist_arr,
        body_to_sensor_transform=t_bs,
        calibration_file=Path("synthetic/mav0/cam0/sensor.yaml"),
    )


def _generate_synthetic_two_view_scene(
    calibration: CameraCalibration,
    *,
    num_points: int = 80,
    rotation_vec: tuple[float, float, float] = (0.03, -0.05, 0.02),
    translation_vec: tuple[float, float, float] = (0.25, -0.08, 0.12),
    pixel_noise_std: float = 0.0,
    num_outliers: int = 0,
    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate 3D points in front of both cameras and project to pixel coordinates."""
    rng = np.random.default_rng(seed)
    points_ref = np.column_stack(
        (
            rng.uniform(-1.5, 1.5, size=num_points),
            rng.uniform(-1.0, 1.0, size=num_points),
            rng.uniform(3.0, 7.5, size=num_points),
        )
    )

    r_gt, _ = cv2.Rodrigues(np.array(rotation_vec, dtype=np.float64))
    t_gt = np.array(translation_vec, dtype=np.float64)
    points_cur = (r_gt @ points_ref.T).T + t_gt
    assert np.all(points_ref[:, 2] > 0.5)
    assert np.all(points_cur[:, 2] > 0.5)

    zero_rvec = np.zeros((3, 1), dtype=np.float64)
    zero_tvec = np.zeros((3, 1), dtype=np.float64)
    proj_ref, _ = cv2.projectPoints(
        points_ref,
        zero_rvec,
        zero_tvec,
        calibration.camera_matrix,
        calibration.distortion_array,
    )
    proj_cur, _ = cv2.projectPoints(
        points_cur,
        zero_rvec,
        zero_tvec,
        calibration.camera_matrix,
        calibration.distortion_array,
    )
    pixels_ref = proj_ref.reshape((-1, 2)).astype(np.float64)
    pixels_cur = proj_cur.reshape((-1, 2)).astype(np.float64)

    if pixel_noise_std > 0.0:
        pixels_ref = pixels_ref + rng.normal(
            0.0, pixel_noise_std, size=pixels_ref.shape
        )
        pixels_cur = pixels_cur + rng.normal(
            0.0, pixel_noise_std, size=pixels_cur.shape
        )

    if num_outliers > 0:
        out_ref = np.column_stack(
            (
                rng.uniform(40.0, 700.0, size=num_outliers),
                rng.uniform(40.0, 440.0, size=num_outliers),
            )
        )
        out_cur = np.column_stack(
            (
                rng.uniform(40.0, 700.0, size=num_outliers),
                rng.uniform(40.0, 440.0, size=num_outliers),
            )
        )
        pixels_ref = np.vstack((pixels_ref, out_ref))
        pixels_cur = np.vstack((pixels_cur, out_cur))

    t_norm = float(np.linalg.norm(t_gt))
    t_unit_gt = t_gt / t_norm if t_norm > 0.0 else t_gt
    return pixels_ref, pixels_cur, r_gt, t_unit_gt


def _make_temporal_match_result(
    ref_coords: list[tuple[float, float]],
    cur_coords: list[tuple[float, float]],
    ratio_matches: tuple[cv2.DMatch, ...],
    inlier_matches: tuple[cv2.DMatch, ...],
) -> TemporalMatchResult:
    ref_kps = tuple(cv2.KeyPoint(float(u), float(v), 12.0) for u, v in ref_coords)
    cur_kps = tuple(cv2.KeyPoint(float(u), float(v), 12.0) for u, v in cur_coords)
    ref_features = FeatureSet(
        keypoints=ref_kps,
        descriptors=np.zeros((len(ref_kps), ORB_DESCRIPTOR_WIDTH), dtype=np.uint8),
        image_shape=(480, 752),
    )
    cur_features = FeatureSet(
        keypoints=cur_kps,
        descriptors=np.zeros((len(cur_kps), ORB_DESCRIPTOR_WIDTH), dtype=np.uint8),
        image_shape=(480, 752),
    )
    stats = MatchStatistics(
        reference_keypoint_count=len(ref_kps),
        current_keypoint_count=len(cur_kps),
        raw_knn_match_count=len(ratio_matches),
        ratio_filtered_match_count=len(ratio_matches),
        geometric_inlier_count=len(inlier_matches),
        geometric_outlier_count=len(ratio_matches) - len(inlier_matches),
        inlier_ratio=(
            len(inlier_matches) / len(ratio_matches) if ratio_matches else 0.0
        ),
    )
    return TemporalMatchResult(
        reference_features=ref_features,
        current_features=cur_features,
        ratio_filtered_matches=ratio_matches,
        geometric_inlier_matches=inlier_matches,
        fundamental_matrix=np.eye(3, dtype=np.float64),
        statistics=stats,
    )


def test_extract_inlier_correspondences_uses_only_geometric_inliers() -> None:
    ref_pts = [(10.5, 20.25), (100.0, 200.0), (300.5, 150.75)]
    cur_pts = [(12.5, 21.25), (400.0, 50.0), (304.5, 149.75)]
    ratio_matches = (
        cv2.DMatch(0, 0, 5.0),
        cv2.DMatch(1, 1, 18.0),  # ratio-filtered outlier, NOT in geometric inliers
        cv2.DMatch(2, 2, 7.0),
    )
    inlier_matches = (cv2.DMatch(0, 0, 5.0), cv2.DMatch(2, 2, 7.0))
    match_result = _make_temporal_match_result(
        ref_pts, cur_pts, ratio_matches, inlier_matches
    )

    extracted_ref, extracted_cur = extract_inlier_correspondences(match_result)

    assert extracted_ref.shape == (2, 2)
    assert extracted_cur.shape == (2, 2)
    np.testing.assert_allclose(extracted_ref, [[10.5, 20.25], [300.5, 150.75]])
    np.testing.assert_allclose(extracted_cur, [[12.5, 21.25], [304.5, 149.75]])

    # Empty inlier matches return shape (0, 2)
    empty_match_result = _make_temporal_match_result(
        ref_pts, cur_pts, ratio_matches, ()
    )
    empty_ref, empty_cur = extract_inlier_correspondences(empty_match_result)
    assert empty_ref.shape == (0, 2)
    assert empty_cur.shape == (0, 2)


def test_extract_inlier_correspondences_rejects_invalid_indices_and_nonfinite() -> None:
    bad_idx_result = _make_temporal_match_result(
        [(10.0, 20.0)],
        [(11.0, 21.0)],
        (cv2.DMatch(5, 0, 1.0),),
        (cv2.DMatch(5, 0, 1.0),),
    )
    with pytest.raises(IndexError, match="queryIdx"):
        extract_inlier_correspondences(bad_idx_result)

    nan_result = _make_temporal_match_result(
        [(float("nan"), 20.0)],
        [(11.0, 21.0)],
        (cv2.DMatch(0, 0, 1.0),),
        (cv2.DMatch(0, 0, 1.0),),
    )
    with pytest.raises(ValueError, match="nonfinite"):
        extract_inlier_correspondences(nan_result)


def test_undistort_matched_points_recovers_normalized_coords_without_mutation() -> None:
    calib = _synthetic_calibration(with_distortion=True)
    pixels_ref, pixels_cur, _, _ = _generate_synthetic_two_view_scene(
        calib, num_points=25, pixel_noise_std=0.0
    )
    ref_snapshot = pixels_ref.copy()
    cur_snapshot = pixels_cur.copy()

    norm_ref, norm_cur = undistort_matched_points(pixels_ref, pixels_cur, calib)

    assert np.array_equal(pixels_ref, ref_snapshot)
    assert np.array_equal(pixels_cur, cur_snapshot)
    assert norm_ref.shape == (25, 2)
    assert norm_cur.shape == (25, 2)
    assert np.all(np.isfinite(norm_ref))
    assert np.all(np.isfinite(norm_cur))

    # Zero-distortion check against (u - cx) / fx, (v - cy) / fy
    pinhole_calib = _synthetic_calibration(with_distortion=False)
    norm_ref_pinhole, _ = undistort_matched_points(
        pixels_ref, pixels_cur, pinhole_calib
    )
    expected_x = (pixels_ref[:, 0] - pinhole_calib.cx) / pinhole_calib.fx
    expected_y = (pixels_ref[:, 1] - pinhole_calib.cy) / pinhole_calib.fy
    np.testing.assert_allclose(
        norm_ref_pinhole, np.column_stack((expected_x, expected_y)), atol=1e-7
    )


def test_undistort_matched_points_validates_shapes_counts_and_finiteness() -> None:
    calib = _synthetic_calibration()
    valid = np.ones((5, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="shape"):
        undistort_matched_points(np.ones((5, 3)), valid, calib)
    with pytest.raises(ValueError, match="Unequal correspondence counts"):
        undistort_matched_points(valid, np.ones((4, 2)), calib)

    nonfinite = valid.copy()
    nonfinite[2, 1] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        undistort_matched_points(nonfinite, valid, calib)


def test_essential_matrix_validation_and_insufficient_correspondences() -> None:
    pts = np.ones((6, 2), dtype=np.float64)
    candidates, mask, reason = estimate_essential_matrix(
        pts, pts, min_correspondences=8
    )
    assert candidates == ()
    assert mask.shape == (6,)
    assert not np.any(mask)
    assert reason is not None and "Insufficient correspondences" in reason

    with pytest.raises(ValueError, match="ransac_probability"):
        estimate_essential_matrix(pts, pts, ransac_probability=1.0)
    with pytest.raises(ValueError, match="ransac_threshold"):
        estimate_essential_matrix(pts, pts, ransac_threshold=0.0)


def test_mask_and_candidate_splitting_validation() -> None:
    mask = validate_essential_inlier_mask(np.array([[1], [0], [1]], dtype=np.uint8), 3)
    assert mask.tolist() == [True, False, True]
    with pytest.raises(ValueError, match="mask length"):
        validate_essential_inlier_mask(np.array([1, 0], dtype=np.uint8), 3)

    assert split_essential_matrix_candidates(None) == ()
    with pytest.raises(ValueError, match="Malformed essential matrix shape"):
        split_essential_matrix_candidates(np.eye(4))
    with pytest.raises(ValueError, match="nonfinite"):
        bad_e = np.eye(3)
        bad_e[0, 0] = float("nan")
        split_essential_matrix_candidates(bad_e)


def test_stacked_essential_matrix_candidates_selected_deterministically() -> None:
    calib = _synthetic_calibration(with_distortion=False)
    pixels_ref, pixels_cur, r_gt, t_unit_gt = _generate_synthetic_two_view_scene(
        calib, num_points=40
    )
    norm_ref, norm_cur = undistort_matched_points(pixels_ref, pixels_cur, calib)

    candidates, inlier_mask, reason = estimate_essential_matrix(norm_ref, norm_cur)
    assert reason is None
    assert len(candidates) >= 1
    true_e = candidates[0]

    # Construct a bogus 3x3 candidate and stack [bogus_e; true_e; true_e]
    bogus_e = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64
    )
    stacked = np.vstack((bogus_e, true_e, true_e))
    split = split_essential_matrix_candidates(stacked)
    assert len(split) == 3

    pose = recover_pose_from_essential_matrix(stacked, norm_ref, norm_cur, inlier_mask)
    assert pose.success
    assert pose.essential_matrix is not None
    np.testing.assert_allclose(pose.essential_matrix, true_e)

    # Also verify missing/malformed essential matrix handling in recover_pose
    failed_pose = recover_pose_from_essential_matrix(
        np.zeros((2, 3)), norm_ref, norm_cur, inlier_mask
    )
    assert not failed_pose.success
    assert failed_pose.rotation_matrix is None
    assert failed_pose.translation_direction is None


def test_synthetic_two_view_pose_recovery_accuracy_with_noise_and_outliers() -> None:
    calib = _synthetic_calibration(with_distortion=True)

    # 1. Noiseless distorted scene: verify near-exact recovery
    clean_ref, clean_cur, r_clean_gt, t_clean_gt = _generate_synthetic_two_view_scene(
        calib,
        num_points=80,
        rotation_vec=(0.04, -0.06, 0.03),
        translation_vec=(0.30, -0.10, 0.15),
        pixel_noise_std=0.0,
        num_outliers=0,
        seed=2026,
    )
    clean_norm_ref, clean_norm_cur = undistort_matched_points(
        clean_ref, clean_cur, calib
    )
    clean_pose = estimate_relative_pose_from_normalized_points(
        clean_norm_ref, clean_norm_cur
    )
    assert clean_pose.success
    assert clean_pose.rotation_matrix is not None
    assert clean_pose.translation_direction is not None
    clean_rot_err = compute_rotation_angle_degrees(
        clean_pose.rotation_matrix @ r_clean_gt.T
    )
    clean_cos_t = float(
        np.clip(np.dot(clean_pose.translation_direction, t_clean_gt), -1.0, 1.0)
    )
    clean_trans_err = float(np.degrees(np.arccos(clean_cos_t)))
    assert clean_rot_err < 0.05
    assert clean_trans_err < 0.1

    # 2. Noisy distorted scene with gross correspondence outliers
    pixels_ref, pixels_cur, r_gt, t_unit_gt = _generate_synthetic_two_view_scene(
        calib,
        num_points=90,
        rotation_vec=(0.04, -0.06, 0.03),
        translation_vec=(0.30, -0.10, 0.15),
        pixel_noise_std=0.08,
        num_outliers=8,
        seed=2026,
    )
    norm_ref, norm_cur = undistort_matched_points(pixels_ref, pixels_cur, calib)
    ref_copy = norm_ref.copy()
    cur_copy = norm_cur.copy()

    pose = estimate_relative_pose_from_normalized_points(
        norm_ref,
        norm_cur,
        ransac_probability=0.999,
        ransac_threshold=1e-3,
    )

    assert np.array_equal(norm_ref, ref_copy)
    assert np.array_equal(norm_cur, cur_copy)
    assert pose.success, pose.failure_reason
    assert pose.failure_reason is None
    assert pose.essential_matrix is not None and pose.essential_matrix.shape == (3, 3)
    assert pose.rotation_matrix is not None and pose.rotation_matrix.shape == (3, 3)
    assert (
        pose.translation_direction is not None
        and pose.translation_direction.shape == (3,)
    )

    det_r = validate_rotation_matrix(pose.rotation_matrix)
    assert det_r == pytest.approx(1.0, abs=1e-5)
    assert float(np.linalg.norm(pose.translation_direction)) == pytest.approx(
        1.0, abs=1e-6
    )

    stats = pose.statistics
    assert stats.input_correspondence_count == 98
    assert 0 < stats.essential_inlier_count <= stats.input_correspondence_count
    assert 0 < stats.cheirality_inlier_count <= stats.essential_inlier_count
    assert stats.essential_inlier_ratio == pytest.approx(
        stats.essential_inlier_count / stats.input_correspondence_count
    )
    assert stats.cheirality_inlier_ratio == pytest.approx(
        stats.cheirality_inlier_count / stats.input_correspondence_count
    )
    assert pose.essential_inlier_mask.shape == (98,)
    assert pose.cheirality_inlier_mask.shape == (98,)

    # Angular error between recovered rotation and ground-truth rotation
    r_err_mat = pose.rotation_matrix @ r_gt.T
    rot_err_deg = compute_rotation_angle_degrees(r_err_mat)
    assert rot_err_deg < 1.0

    # Angular error between unit translation directions
    cos_t = float(np.clip(np.dot(pose.translation_direction, t_unit_gt), -1.0, 1.0))
    trans_err_deg = float(np.degrees(np.arccos(cos_t)))
    assert trans_err_deg < 2.5


def test_degenerate_identical_correspondences_and_pure_rotation() -> None:
    identical = np.full((16, 2), 0.1, dtype=np.float64)
    ident_pose = estimate_relative_pose_from_normalized_points(identical, identical)
    assert not ident_pose.success
    assert ident_pose.failure_reason is not None
    assert ident_pose.rotation_matrix is None
    assert ident_pose.translation_direction is None

    # Pure rotation (zero translation): verify structured response without exceptions
    calib = _synthetic_calibration(with_distortion=False)
    pixels_ref, pixels_cur, r_gt, _ = _generate_synthetic_two_view_scene(
        calib,
        num_points=40,
        rotation_vec=(0.0, 0.04, 0.0),
        translation_vec=(0.0, 0.0, 0.0),
    )
    norm_ref, norm_cur = undistort_matched_points(pixels_ref, pixels_cur, calib)
    pure_rot_pose = estimate_relative_pose_from_normalized_points(norm_ref, norm_cur)
    assert isinstance(pure_rot_pose.success, bool)
    if pure_rot_pose.success:
        assert pure_rot_pose.rotation_matrix is not None
        validate_rotation_matrix(pure_rot_pose.rotation_matrix)
    else:
        assert pure_rot_pose.failure_reason is not None
        assert pure_rot_pose.rotation_matrix is None


def test_high_level_estimate_calibrated_relative_pose_preserves_inputs() -> None:
    calib = _synthetic_calibration(with_distortion=False)
    rng = np.random.default_rng(77)
    canvas = np.full((500, 780), 30, dtype=np.uint8)
    for row in range(48, 440, 36):
        for col in range(48, 710, 36):
            val = int(rng.integers(85, 245))
            cv2.rectangle(canvas, (col, row), (col + 16, row + 16), val, -1)
            cv2.circle(canvas, (col + 8, row + 8), 4, 255 - val // 2, 1)

    ref_img = canvas[10:490, 14:766].copy()
    cur_img = canvas[8:488, 10:762].copy()
    ref_before = ref_img.copy()
    cur_before = cur_img.copy()

    match_res, pose_res = estimate_calibrated_relative_pose(
        ref_img,
        cur_img,
        calib,
        minimum_cheirality_count=1,
        minimum_cheirality_ratio=0.01,
    )

    assert np.array_equal(ref_img, ref_before)
    assert np.array_equal(cur_img, cur_before)
    assert match_res.statistics.geometric_inlier_count > 0
    assert (
        pose_res.statistics.input_correspondence_count
        == match_res.statistics.geometric_inlier_count
    )
    assert pose_res.success
    assert pose_res.rotation_matrix is not None
    assert pose_res.translation_direction is not None
    assert pose_res.statistics.pose_quality_valid
    assert pose_res.statistics.minimum_cheirality_count == 1
    assert pose_res.statistics.minimum_cheirality_ratio == pytest.approx(0.01)


def test_pose_quality_classification_marks_low_cheirality_without_dropping_pose() -> (
    None
):
    calib = _synthetic_calibration(with_distortion=False)
    pixels_ref, pixels_cur, _, _ = _generate_synthetic_two_view_scene(
        calib, num_points=45, seed=909
    )
    norm_ref, norm_cur = undistort_matched_points(pixels_ref, pixels_cur, calib)

    # Demand a higher cheirality count than the 45 points available
    low_conf_pose = estimate_relative_pose_from_normalized_points(
        norm_ref,
        norm_cur,
        minimum_cheirality_count=100,
        minimum_cheirality_ratio=0.95,
    )
    assert low_conf_pose.success
    assert low_conf_pose.rotation_matrix is not None
    assert low_conf_pose.translation_direction is not None
    assert not low_conf_pose.statistics.pose_quality_valid
    assert (
        "Low-confidence relative pose"
        in low_conf_pose.statistics.pose_quality_reason
    )
    assert low_conf_pose.statistics.minimum_cheirality_count == 100
    assert low_conf_pose.statistics.minimum_cheirality_ratio == pytest.approx(0.95)

    valid_flag, valid_reason = evaluate_pose_quality(
        40,
        0.50,
        minimum_cheirality_count=DEFAULT_MIN_CHEIRALITY_COUNT,
        minimum_cheirality_ratio=DEFAULT_MIN_CHEIRALITY_RATIO,
    )
    assert valid_flag
    assert "meets quality thresholds" in valid_reason

    with pytest.raises(ValueError, match="minimum_cheirality_count"):
        evaluate_pose_quality(10, 0.5, minimum_cheirality_count=0)
    with pytest.raises(ValueError, match="minimum_cheirality_ratio"):
        evaluate_pose_quality(10, 0.5, minimum_cheirality_ratio=0.0)
    with pytest.raises(ValueError, match="minimum_cheirality_ratio"):
        evaluate_pose_quality(10, 0.5, minimum_cheirality_ratio=1.5)
