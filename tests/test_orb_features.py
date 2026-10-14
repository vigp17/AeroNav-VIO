"""Unit tests for ORB feature detection and temporal matching."""

from dataclasses import FrozenInstanceError

import cv2
import numpy as np
import pytest

from aeronav_vio.features.orb import (
    ORB_DESCRIPTOR_WIDTH,
    FeatureSet,
    apply_inlier_mask,
    extract_orb_features,
    filter_knn_matches,
    match_orb_descriptors,
    match_temporal_pair,
    render_inlier_matches,
    verify_matches_ransac,
)


def _make_textured_grayscale_pair(
    height: int = 320, width: int = 480, shift_x: int = 4, shift_y: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic textured grayscale image pair related by a 2D shift."""
    rng = np.random.default_rng(42)
    canvas = np.full((height + 20, width + 20), 30, dtype=np.uint8)

    for row in range(40, height - 40, 36):
        for col in range(40, width - 40, 36):
            intensity = int(rng.integers(90, 245))
            patch_h = int(rng.integers(10, 22))
            patch_w = int(rng.integers(10, 22))
            cv2.rectangle(
                canvas,
                (col, row),
                (col + patch_w, row + patch_h),
                intensity,
                thickness=-1,
            )
            cv2.circle(
                canvas,
                (col + patch_w // 2, row + patch_h // 2),
                4,
                255 - intensity // 2,
                thickness=1,
            )

    ref = canvas[10 : 10 + height, 10 : 10 + width].copy()
    cur = canvas[
        10 - shift_y : 10 - shift_y + height,
        10 - shift_x : 10 - shift_x + width,
    ].copy()
    return ref, cur


def test_extract_orb_features_valid_grayscale_preserves_shape_and_input() -> None:
    ref_image, _ = _make_textured_grayscale_pair(height=240, width=360)
    original_snapshot = ref_image.copy()

    features = extract_orb_features(ref_image, max_features=300)

    assert np.array_equal(ref_image, original_snapshot)
    assert features.image_shape == (240, 360)
    assert len(features.keypoints) > 0
    assert features.descriptors is not None
    assert features.descriptors.dtype == np.uint8
    assert features.descriptors.ndim == 2
    assert features.descriptors.shape == (
        len(features.keypoints),
        ORB_DESCRIPTOR_WIDTH,
    )
    with pytest.raises(FrozenInstanceError):
        features.image_shape = (100, 100)  # type: ignore[misc]


def test_extract_orb_features_rejects_empty_image() -> None:
    empty_image = np.zeros((0, 128), dtype=np.uint8)
    with pytest.raises(ValueError, match="nonempty 2D grayscale"):
        extract_orb_features(empty_image)


def test_extract_orb_features_rejects_color_image() -> None:
    color_image = np.zeros((120, 160, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="2D grayscale"):
        extract_orb_features(color_image)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.uint16])
def test_extract_orb_features_rejects_unsupported_dtype(dtype: type) -> None:
    bad_image = np.zeros((120, 160), dtype=dtype)
    with pytest.raises(ValueError, match="Unsupported dtype"):
        extract_orb_features(bad_image)


def test_extract_orb_features_handles_blank_no_keypoint_image() -> None:
    blank = np.full((180, 240), 128, dtype=np.uint8)

    features = extract_orb_features(blank)

    assert features.keypoints == ()
    assert features.descriptors is None
    assert features.image_shape == (180, 240)


def test_match_orb_descriptors_handles_missing_descriptors() -> None:
    ref_image, _ = _make_textured_grayscale_pair()
    non_empty = extract_orb_features(ref_image, max_features=150)
    empty = FeatureSet(keypoints=(), descriptors=None, image_shape=ref_image.shape)

    raw_count_a, matches_a = match_orb_descriptors(empty, non_empty)
    raw_count_b, matches_b = match_orb_descriptors(non_empty, empty)

    assert raw_count_a == 0
    assert matches_a == ()
    assert raw_count_b == 0
    assert matches_b == ()


def test_filter_knn_matches_applies_ratio_and_handles_short_entries() -> None:
    knn_candidates = [
        (cv2.DMatch(0, 3, 20.0), cv2.DMatch(0, 4, 50.0)),  # 20 < 0.75 * 50 -> keep
        (cv2.DMatch(1, 2, 40.0), cv2.DMatch(1, 5, 50.0)),  # 40 >= 37.5 -> reject
        (cv2.DMatch(2, 1, 10.0),),  # fewer than 2 candidates -> skip
        (),  # empty entry -> not counted as raw KNN match
        (cv2.DMatch(3, 0, 12.0), cv2.DMatch(3, 6, 40.0)),  # 12 < 30 -> keep
    ]

    raw_count, filtered = filter_knn_matches(knn_candidates, ratio_threshold=0.75)

    assert raw_count == 4
    assert len(filtered) == 2
    assert [m.queryIdx for m in filtered] == [3, 0]
    assert [m.trainIdx for m in filtered] == [0, 3]


def test_match_orb_descriptors_valid_knn_does_not_mutate_inputs() -> None:
    ref_image, cur_image = _make_textured_grayscale_pair()
    ref_features = extract_orb_features(ref_image, max_features=250)
    cur_features = extract_orb_features(cur_image, max_features=250)
    assert ref_features.descriptors is not None
    assert cur_features.descriptors is not None
    ref_desc_before = ref_features.descriptors.copy()
    cur_desc_before = cur_features.descriptors.copy()

    raw_count, filtered = match_orb_descriptors(
        ref_features, cur_features, ratio_threshold=0.75
    )

    assert np.array_equal(ref_features.descriptors, ref_desc_before)
    assert np.array_equal(cur_features.descriptors, cur_desc_before)
    assert raw_count > 0
    assert 0 < len(filtered) <= raw_count
    for match in filtered:
        assert 0 <= match.queryIdx < len(ref_features.keypoints)
        assert 0 <= match.trainIdx < len(cur_features.keypoints)


def test_verify_matches_ransac_insufficient_correspondences() -> None:
    keypoints = tuple(
        cv2.KeyPoint(float(10 * idx + 5), float(12 * idx + 7), 16.0)
        for idx in range(6)
    )
    descriptors = np.zeros((6, ORB_DESCRIPTOR_WIDTH), dtype=np.uint8)
    features = FeatureSet(
        keypoints=keypoints, descriptors=descriptors, image_shape=(200, 200)
    )
    matches = tuple(cv2.DMatch(idx, idx, 5.0) for idx in range(6))

    result = verify_matches_ransac(features, features, matches)

    assert result.fundamental_matrix is None
    assert result.inlier_matches == ()
    assert result.inlier_count == 0
    assert result.outlier_count == 6
    assert result.inlier_ratio == 0.0
    assert result.inlier_mask == (False,) * 6


def test_verify_matches_ransac_identifies_inliers_on_synthetic_correspondences() -> (
    None
):
    rng = np.random.default_rng(7)
    inlier_count = 24
    ref_pts = rng.uniform(25.0, 275.0, size=(inlier_count, 2))
    cur_pts = ref_pts + np.array([6.0, -3.0])

    # Append 4 gross outliers
    outlier_ref = np.array(
        [[30.0, 40.0], [260.0, 50.0], [45.0, 250.0], [230.0, 220.0]]
    )
    outlier_cur = np.array(
        [[250.0, 240.0], [25.0, 230.0], [240.0, 35.0], [40.0, 45.0]]
    )

    all_ref = np.vstack((ref_pts, outlier_ref))
    all_cur = np.vstack((cur_pts, outlier_cur))
    total = all_ref.shape[0]

    ref_kps = tuple(
        cv2.KeyPoint(float(pt[0]), float(pt[1]), 12.0) for pt in all_ref
    )
    cur_kps = tuple(
        cv2.KeyPoint(float(pt[0]), float(pt[1]), 12.0) for pt in all_cur
    )
    ref_set = FeatureSet(
        keypoints=ref_kps,
        descriptors=np.zeros((total, ORB_DESCRIPTOR_WIDTH), dtype=np.uint8),
        image_shape=(300, 300),
    )
    cur_set = FeatureSet(
        keypoints=cur_kps,
        descriptors=np.zeros((total, ORB_DESCRIPTOR_WIDTH), dtype=np.uint8),
        image_shape=(300, 300),
    )
    matches = tuple(cv2.DMatch(idx, idx, 4.0) for idx in range(total))

    result = verify_matches_ransac(
        ref_set,
        cur_set,
        matches,
        reprojection_threshold=1.0,
        confidence=0.999,
    )

    assert result.fundamental_matrix is not None
    assert result.fundamental_matrix.shape == (3, 3)
    assert result.inlier_count >= 20
    assert result.outlier_count == total - result.inlier_count
    assert len(result.inlier_mask) == total
    assert result.inlier_ratio == pytest.approx(result.inlier_count / total)


def test_verify_matches_ransac_handles_estimation_failure() -> None:
    # Degenerate identical coordinates cannot define a valid fundamental matrix
    degenerate_kps = tuple(cv2.KeyPoint(50.0, 50.0, 10.0) for _ in range(10))
    features = FeatureSet(
        keypoints=degenerate_kps,
        descriptors=np.zeros((10, ORB_DESCRIPTOR_WIDTH), dtype=np.uint8),
        image_shape=(100, 100),
    )
    matches = tuple(cv2.DMatch(idx, idx, 0.0) for idx in range(10))

    result = verify_matches_ransac(features, features, matches)

    assert result.fundamental_matrix is None
    assert result.inlier_matches == ()
    assert result.inlier_count == 0
    assert result.outlier_count == 10
    assert result.inlier_ratio == 0.0


def test_apply_inlier_mask_validates_length_and_zero_matches() -> None:
    empty_inliers, empty_mask, in_cnt, out_cnt, ratio = apply_inlier_mask((), None)
    assert empty_inliers == ()
    assert empty_mask == ()
    assert (in_cnt, out_cnt, ratio) == (0, 0, 0.0)

    matches = (cv2.DMatch(0, 0, 1.0), cv2.DMatch(1, 1, 2.0), cv2.DMatch(2, 2, 3.0))
    inliers, flags, in_cnt, out_cnt, ratio = apply_inlier_mask(
        matches, np.array([[1], [0], [1]], dtype=np.uint8)
    )
    assert len(inliers) == 2
    assert flags == (True, False, True)
    assert in_cnt == 2
    assert out_cnt == 1
    assert ratio == pytest.approx(2.0 / 3.0)

    with pytest.raises(ValueError, match="RANSAC mask length"):
        apply_inlier_mask(matches, np.array([[1], [0]], dtype=np.uint8))


def test_high_level_temporal_matching_and_visualization() -> None:
    ref_image, cur_image = _make_textured_grayscale_pair(height=280, width=400)
    ref_before = ref_image.copy()
    cur_before = cur_image.copy()

    result = match_temporal_pair(ref_image, cur_image, max_features=400)

    assert np.array_equal(ref_image, ref_before)
    assert np.array_equal(cur_image, cur_before)
    stats = result.statistics
    assert stats.reference_keypoint_count > 0
    assert stats.current_keypoint_count > 0
    assert 0 < stats.ratio_filtered_match_count <= stats.raw_knn_match_count
    assert 0 < stats.geometric_inlier_count <= stats.ratio_filtered_match_count
    assert (
        stats.geometric_inlier_count + stats.geometric_outlier_count
        == stats.ratio_filtered_match_count
    )
    assert stats.inlier_ratio == pytest.approx(
        stats.geometric_inlier_count / stats.ratio_filtered_match_count
    )

    vis = render_inlier_matches(
        ref_image,
        cur_image,
        result,
        reference_timestamp_ns=1000,
        current_timestamp_ns=2000,
    )
    assert np.array_equal(ref_image, ref_before)
    assert np.array_equal(cur_image, cur_before)
    assert isinstance(vis, np.ndarray)
    assert vis.dtype == np.uint8
    assert vis.ndim == 3
    assert vis.shape[1] == 800
    assert vis.shape[0] > 280
    assert vis.shape[2] == 3


def test_visualization_with_zero_inlier_matches() -> None:
    blank_ref = np.full((120, 160), 60, dtype=np.uint8)
    blank_cur = np.full((120, 160), 90, dtype=np.uint8)

    result = match_temporal_pair(blank_ref, blank_cur)
    assert result.statistics.geometric_inlier_count == 0
    assert result.statistics.inlier_ratio == 0.0

    vis = render_inlier_matches(blank_ref, blank_cur, result)
    assert isinstance(vis, np.ndarray)
    assert vis.shape[1] == 320
    assert vis.shape[0] > 120
    assert vis.shape[2] == 3
