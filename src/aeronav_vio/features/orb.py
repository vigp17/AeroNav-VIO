"""ORB feature extraction and geometrically verified temporal matching."""

from dataclasses import dataclass

import cv2
import numpy as np

ORB_DESCRIPTOR_WIDTH = 32
MIN_FUNDAMENTAL_CORRESPONDENCES = 8


@dataclass(frozen=True)
class FeatureSet:
    """Detected keypoints and binary descriptors for one grayscale image."""

    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray | None
    image_shape: tuple[int, int]


@dataclass(frozen=True)
class MatchStatistics:
    """Summary counts and ratios across temporal matching stages."""

    reference_keypoint_count: int
    current_keypoint_count: int
    raw_knn_match_count: int
    ratio_filtered_match_count: int
    geometric_inlier_count: int
    geometric_outlier_count: int
    inlier_ratio: float


@dataclass(frozen=True)
class GeometricVerificationResult:
    """Fundamental-matrix RANSAC verification output."""

    inlier_matches: tuple[cv2.DMatch, ...]
    fundamental_matrix: np.ndarray | None
    inlier_mask: tuple[bool, ...]
    inlier_count: int
    outlier_count: int
    inlier_ratio: float


@dataclass(frozen=True)
class TemporalMatchResult:
    """Complete result of matching two consecutive monocular frames."""

    reference_features: FeatureSet
    current_features: FeatureSet
    ratio_filtered_matches: tuple[cv2.DMatch, ...]
    geometric_inlier_matches: tuple[cv2.DMatch, ...]
    fundamental_matrix: np.ndarray | None
    statistics: MatchStatistics


def _validate_grayscale_image(image: np.ndarray, label: str = "image") -> None:
    """Validate that an input is a nonempty 2D uint8 grayscale NumPy array."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected {label} to be a NumPy array, got {type(image)!r}")
    if image.dtype != np.uint8:
        raise ValueError(
            f"Unsupported dtype {image.dtype} for {label}: expected uint8 grayscale"
        )
    if image.ndim == 3:
        raise ValueError(
            f"Expected 2D grayscale array for {label}, got color or 3D shape "
            f"{image.shape}"
        )
    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale array for {label}, got {image.ndim}D shape "
            f"{image.shape}"
        )
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(
            f"Expected nonempty 2D grayscale array for {label}, got shape {image.shape}"
        )


def _validate_descriptors(
    descriptors: np.ndarray | None, keypoint_count: int, label: str
) -> None:
    """Validate binary ORB descriptor array shape and dtype when present."""
    if descriptors is None:
        return
    if not isinstance(descriptors, np.ndarray):
        raise TypeError(f"Expected {label} descriptors to be a NumPy array or None")
    if descriptors.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 binary descriptors for {label}, got {descriptors.dtype}"
        )
    if descriptors.ndim != 2 or descriptors.shape[1] != ORB_DESCRIPTOR_WIDTH:
        raise ValueError(
            f"Expected {label} descriptors with shape (N, {ORB_DESCRIPTOR_WIDTH}), "
            f"got {descriptors.shape}"
        )
    if descriptors.shape[0] != keypoint_count:
        raise ValueError(
            f"Descriptor row count {descriptors.shape[0]} does not match "
            f"{label} keypoint count {keypoint_count}"
        )


def extract_orb_features(
    image: np.ndarray,
    *,
    max_features: int = 1000,
    scale_factor: float = 1.2,
    n_levels: int = 8,
    edge_threshold: int = 31,
    fast_threshold: int = 20,
) -> FeatureSet:
    """Detect ORB keypoints and compute 256-bit binary descriptors on a 2D image."""
    _validate_grayscale_image(image)
    if max_features <= 0:
        raise ValueError(f"max_features must be positive, got {max_features}")
    if scale_factor <= 1.0:
        raise ValueError(f"scale_factor must be > 1.0, got {scale_factor}")
    if n_levels <= 0:
        raise ValueError(f"n_levels must be positive, got {n_levels}")
    if edge_threshold < 0:
        raise ValueError(f"edge_threshold must be non-negative, got {edge_threshold}")
    if fast_threshold <= 0:
        raise ValueError(f"fast_threshold must be positive, got {fast_threshold}")

    orb = cv2.ORB_create(
        nfeatures=max_features,
        scaleFactor=scale_factor,
        nlevels=n_levels,
        edgeThreshold=edge_threshold,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=max(edge_threshold, 2),
        fastThreshold=fast_threshold,
    )
    keypoints, descriptors = orb.detectAndCompute(image, None)
    height, width = int(image.shape[0]), int(image.shape[1])

    if not keypoints or descriptors is None or descriptors.size == 0:
        return FeatureSet(
            keypoints=tuple(keypoints) if keypoints else (),
            descriptors=None,
            image_shape=(height, width),
        )

    descriptor_copy = np.array(descriptors, dtype=np.uint8, copy=True)
    descriptor_copy.setflags(write=False)
    return FeatureSet(
        keypoints=tuple(keypoints),
        descriptors=descriptor_copy,
        image_shape=(height, width),
    )


def filter_knn_matches(
    knn_matches: list[tuple[cv2.DMatch, ...]] | tuple[tuple[cv2.DMatch, ...], ...],
    *,
    ratio_threshold: float = 0.75,
) -> tuple[int, tuple[cv2.DMatch, ...]]:
    """Apply Lowe's ratio test to k=2 KNN matches and sort deterministically."""
    if not (0.0 < ratio_threshold < 1.0):
        raise ValueError(
            f"ratio_threshold must be strictly between 0 and 1, got {ratio_threshold}"
        )

    raw_knn_count = sum(1 for candidates in knn_matches if len(candidates) > 0)
    accepted: list[cv2.DMatch] = []
    for candidates in knn_matches:
        if len(candidates) < 2:
            continue
        best, second_best = candidates[0], candidates[1]
        if best.distance < ratio_threshold * second_best.distance:
            accepted.append(
                cv2.DMatch(
                    best.queryIdx,
                    best.trainIdx,
                    best.imgIdx,
                    float(best.distance),
                )
            )

    accepted.sort(key=lambda match: (match.distance, match.queryIdx, match.trainIdx))
    return raw_knn_count, tuple(accepted)


def match_orb_descriptors(
    reference_features: FeatureSet,
    current_features: FeatureSet,
    *,
    ratio_threshold: float = 0.75,
) -> tuple[int, tuple[cv2.DMatch, ...]]:
    """Match binary ORB descriptors using Hamming KNN (k=2) and ratio filtering."""
    if not (0.0 < ratio_threshold < 1.0):
        raise ValueError(
            f"ratio_threshold must be strictly between 0 and 1, got {ratio_threshold}"
        )

    _validate_descriptors(
        reference_features.descriptors,
        len(reference_features.keypoints),
        "reference",
    )
    _validate_descriptors(
        current_features.descriptors,
        len(current_features.keypoints),
        "current",
    )

    if (
        not reference_features.keypoints
        or not current_features.keypoints
        or reference_features.descriptors is None
        or current_features.descriptors is None
    ):
        return 0, ()

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(
        reference_features.descriptors,
        current_features.descriptors,
        k=2,
    )
    return filter_knn_matches(knn_matches, ratio_threshold=ratio_threshold)


def apply_inlier_mask(
    matches: tuple[cv2.DMatch, ...] | list[cv2.DMatch],
    mask: np.ndarray | list[int] | tuple[int, ...] | None,
) -> tuple[tuple[cv2.DMatch, ...], tuple[bool, ...], int, int, float]:
    """Extract inliers from a RANSAC mask after validating mask length."""
    match_tuple = tuple(matches)
    filtered_count = len(match_tuple)
    if filtered_count == 0:
        return (), (), 0, 0, 0.0

    if mask is None:
        return (), (False,) * filtered_count, 0, filtered_count, 0.0

    mask_array = np.asarray(mask).reshape(-1)
    if mask_array.shape[0] != filtered_count:
        raise ValueError(
            f"RANSAC mask length {mask_array.shape[0]} does not match "
            f"match count {filtered_count}"
        )

    inlier_flags = tuple(bool(int(flag) != 0) for flag in mask_array)
    inliers = tuple(
        match
        for match, is_inlier in zip(match_tuple, inlier_flags, strict=True)
        if is_inlier
    )
    inlier_count = len(inliers)
    outlier_count = filtered_count - inlier_count
    inlier_ratio = inlier_count / filtered_count
    return inliers, inlier_flags, inlier_count, outlier_count, inlier_ratio


def verify_matches_ransac(
    reference_features: FeatureSet,
    current_features: FeatureSet,
    matches: tuple[cv2.DMatch, ...] | list[cv2.DMatch],
    *,
    reprojection_threshold: float = 1.0,
    confidence: float = 0.99,
) -> GeometricVerificationResult:
    """Verify matched keypoints with fundamental-matrix RANSAC."""
    if reprojection_threshold <= 0.0:
        raise ValueError(
            f"reprojection_threshold must be > 0, got {reprojection_threshold}"
        )
    if not (0.0 < confidence < 1.0):
        raise ValueError(
            f"confidence must be strictly between 0 and 1, got {confidence}"
        )

    match_tuple = tuple(matches)
    filtered_count = len(match_tuple)
    if filtered_count == 0:
        return GeometricVerificationResult(
            inlier_matches=(),
            fundamental_matrix=None,
            inlier_mask=(),
            inlier_count=0,
            outlier_count=0,
            inlier_ratio=0.0,
        )

    ref_count = len(reference_features.keypoints)
    cur_count = len(current_features.keypoints)
    for match in match_tuple:
        if match.queryIdx < 0 or match.queryIdx >= ref_count:
            raise IndexError(
                f"Match queryIdx {match.queryIdx} out of range for {ref_count} "
                "reference keypoints"
            )
        if match.trainIdx < 0 or match.trainIdx >= cur_count:
            raise IndexError(
                f"Match trainIdx {match.trainIdx} out of range for {cur_count} "
                "current keypoints"
            )

    if filtered_count < MIN_FUNDAMENTAL_CORRESPONDENCES:
        return GeometricVerificationResult(
            inlier_matches=(),
            fundamental_matrix=None,
            inlier_mask=(False,) * filtered_count,
            inlier_count=0,
            outlier_count=filtered_count,
            inlier_ratio=0.0,
        )

    ref_points = np.float32(
        [reference_features.keypoints[m.queryIdx].pt for m in match_tuple]
    )
    cur_points = np.float32(
        [current_features.keypoints[m.trainIdx].pt for m in match_tuple]
    )

    fundamental_matrix, mask = cv2.findFundamentalMat(
        ref_points,
        cur_points,
        cv2.FM_RANSAC,
        reprojection_threshold,
        confidence,
    )

    if (
        fundamental_matrix is None
        or mask is None
        or fundamental_matrix.shape != (3, 3)
        or not np.all(np.isfinite(fundamental_matrix))
    ):
        return GeometricVerificationResult(
            inlier_matches=(),
            fundamental_matrix=None,
            inlier_mask=(False,) * filtered_count,
            inlier_count=0,
            outlier_count=filtered_count,
            inlier_ratio=0.0,
        )

    inliers, inlier_flags, inlier_count, outlier_count, inlier_ratio = (
        apply_inlier_mask(match_tuple, mask)
    )
    if inlier_count == 0:
        return GeometricVerificationResult(
            inlier_matches=(),
            fundamental_matrix=None,
            inlier_mask=inlier_flags,
            inlier_count=0,
            outlier_count=filtered_count,
            inlier_ratio=0.0,
        )

    f_copy = np.array(fundamental_matrix, dtype=np.float64, copy=True)
    f_copy.setflags(write=False)
    return GeometricVerificationResult(
        inlier_matches=inliers,
        fundamental_matrix=f_copy,
        inlier_mask=inlier_flags,
        inlier_count=inlier_count,
        outlier_count=outlier_count,
        inlier_ratio=inlier_ratio,
    )


def match_temporal_pair(
    reference_image: np.ndarray,
    current_image: np.ndarray,
    *,
    max_features: int = 1000,
    ratio_threshold: float = 0.75,
    ransac_reprojection_threshold: float = 1.0,
    ransac_confidence: float = 0.99,
) -> TemporalMatchResult:
    """Detect ORB features, match with ratio test, and verify with RANSAC."""
    _validate_grayscale_image(reference_image, "reference_image")
    _validate_grayscale_image(current_image, "current_image")

    ref_features = extract_orb_features(reference_image, max_features=max_features)
    cur_features = extract_orb_features(current_image, max_features=max_features)
    raw_knn_count, ratio_matches = match_orb_descriptors(
        ref_features,
        cur_features,
        ratio_threshold=ratio_threshold,
    )
    verification = verify_matches_ransac(
        ref_features,
        cur_features,
        ratio_matches,
        reprojection_threshold=ransac_reprojection_threshold,
        confidence=ransac_confidence,
    )

    stats = MatchStatistics(
        reference_keypoint_count=len(ref_features.keypoints),
        current_keypoint_count=len(cur_features.keypoints),
        raw_knn_match_count=raw_knn_count,
        ratio_filtered_match_count=len(ratio_matches),
        geometric_inlier_count=verification.inlier_count,
        geometric_outlier_count=verification.outlier_count,
        inlier_ratio=verification.inlier_ratio,
    )
    return TemporalMatchResult(
        reference_features=ref_features,
        current_features=cur_features,
        ratio_filtered_matches=ratio_matches,
        geometric_inlier_matches=verification.inlier_matches,
        fundamental_matrix=verification.fundamental_matrix,
        statistics=stats,
    )


def render_inlier_matches(
    reference_image: np.ndarray,
    current_image: np.ndarray,
    match_result: TemporalMatchResult,
    *,
    reference_timestamp_ns: int | None = None,
    current_timestamp_ns: int | None = None,
) -> np.ndarray:
    """Render side-by-side images with only geometrically verified inlier matches."""
    _validate_grayscale_image(reference_image, "reference_image")
    _validate_grayscale_image(current_image, "current_image")

    if reference_image.shape != match_result.reference_features.image_shape:
        raise ValueError(
            f"reference_image shape {reference_image.shape} does not match "
            f"feature shape {match_result.reference_features.image_shape}"
        )
    if current_image.shape != match_result.current_features.image_shape:
        raise ValueError(
            f"current_image shape {current_image.shape} does not match "
            f"feature shape {match_result.current_features.image_shape}"
        )

    matched_canvas = cv2.drawMatches(
        reference_image,
        list(match_result.reference_features.keypoints),
        current_image,
        list(match_result.current_features.keypoints),
        list(match_result.geometric_inlier_matches),
        None,
        matchColor=(0, 220, 0),
        singlePointColor=(140, 140, 140),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    stats = match_result.statistics
    ref_ts_label = (
        str(reference_timestamp_ns) if reference_timestamp_ns is not None else "N/A"
    )
    cur_ts_label = (
        str(current_timestamp_ns) if current_timestamp_ns is not None else "N/A"
    )

    lines = (
        f"Ref ts (ns): {ref_ts_label} | Cur ts (ns): {cur_ts_label}",
        (
            f"Keypoints ref/cur: {stats.reference_keypoint_count} / "
            f"{stats.current_keypoint_count} | "
            f"Raw KNN: {stats.raw_knn_match_count} | "
            f"Ratio-filtered: {stats.ratio_filtered_match_count}"
        ),
        (
            f"RANSAC inliers: {stats.geometric_inlier_count} | "
            f"Outliers: {stats.geometric_outlier_count} | "
            f"Inlier ratio: {stats.inlier_ratio:.4f}"
        ),
    )

    banner_height = 84
    canvas_width = matched_canvas.shape[1]
    banner = np.full((banner_height, canvas_width, 3), 24, dtype=np.uint8)
    for line_idx, text in enumerate(lines):
        y_pos = 24 + line_idx * 24
        cv2.putText(
            banner,
            text,
            (12, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    return np.vstack((banner, matched_canvas))
