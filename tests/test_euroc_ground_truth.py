"""Unit tests for EuRoC ground-truth CSV parsing and quaternion SLERP."""

import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from aeronav_vio.ground_truth.euroc import (
    EuRoCGroundTruthSequence,
    normalize_quaternion_wxyz,
    quaternion_wxyz_to_rotation_matrix,
    slerp_quaternion_wxyz,
)

HEADER_17 = (
    "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], "
    "q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], "
    "v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], "
    "b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], "
    "b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]\n"
)


def _format_row_17(
    timestamp_ns: int | str,
    pos: tuple[float | str, float | str, float | str],
    quat_wxyz: tuple[float | str, float | str, float | str, float | str],
) -> str:
    extras = ",".join(["0.0"] * 9)
    return (
        f"{timestamp_ns},{pos[0]},{pos[1]},{pos[2]},"
        f"{quat_wxyz[0]},{quat_wxyz[1]},{quat_wxyz[2]},{quat_wxyz[3]},{extras}\n"
    )


def _write_gt_csv(sequence_root: Path, content: str) -> Path:
    gt_dir = sequence_root / "mav0" / "state_groundtruth_estimate0"
    gt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = gt_dir / "data.csv"
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


def test_parse_ground_truth_csv_preserves_precision_and_wxyz_order(
    tmp_path: Path,
) -> None:
    sequence_root = tmp_path / "synthetic_seq"
    ts0 = 1403636580838555648
    ts1 = 1403636580843555328
    # Unnormalized quaternion along z-axis rotation of 90 deg:
    # [cos(45), 0, 0, sin(45)] * 2
    s = math.sqrt(0.5)
    content = (
        HEADER_17
        + "\n"
        + "# comment line\n"
        + _format_row_17(ts0, (1.25, -2.5, 0.75), (2.0, 0.0, 0.0, 0.0))
        + _format_row_17(ts1, (3.25, 1.5, -0.25), (2.0 * s, 0.0, 0.0, 2.0 * s))
    )
    _write_gt_csv(sequence_root, content)

    gt = EuRoCGroundTruthSequence(sequence_root)

    assert gt.sample_count == 2
    assert gt.first_timestamp_ns == ts0
    assert gt.last_timestamp_ns == ts1
    assert isinstance(gt.poses[0].timestamp_ns, int)
    assert gt.poses[0].timestamp_ns == ts0
    assert gt.poses[1].timestamp_ns == ts1

    np.testing.assert_allclose(gt.poses[0].position_world_body, [1.25, -2.5, 0.75])
    np.testing.assert_allclose(
        gt.poses[0].quaternion_world_body_wxyz, [1.0, 0.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(gt.poses[0].rotation_world_body, np.eye(3), atol=1e-12)

    np.testing.assert_allclose(
        gt.poses[1].quaternion_world_body_wxyz, [s, 0.0, 0.0, s]
    )
    expected_rot_z90 = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    np.testing.assert_allclose(
        gt.poses[1].rotation_world_body, expected_rot_z90, atol=1e-12
    )

    with pytest.raises(FrozenInstanceError):
        gt.poses[0].timestamp_ns = 0  # type: ignore[misc]


def test_quaternion_normalization_sign_equivalence_and_rotation_conversion() -> None:
    raw = np.array([2.0, -1.0, 3.0, -0.5], dtype=np.float64)
    raw_copy = raw.copy()

    q_norm = normalize_quaternion_wxyz(raw)
    assert np.array_equal(raw, raw_copy)
    assert float(np.linalg.norm(q_norm)) == pytest.approx(1.0, abs=1e-12)

    rot_pos = quaternion_wxyz_to_rotation_matrix(q_norm)
    rot_neg = quaternion_wxyz_to_rotation_matrix(-q_norm)
    assert np.array_equal(raw, raw_copy)
    assert rot_pos.shape == (3, 3)
    np.testing.assert_allclose(rot_pos, rot_neg, atol=1e-12)
    np.testing.assert_allclose(rot_pos.T @ rot_pos, np.eye(3), atol=1e-12)
    assert float(np.linalg.det(rot_pos)) == pytest.approx(1.0, abs=1e-12)


def test_slerp_shortest_path_and_near_identical_quaternions() -> None:
    # 0 deg to 90 deg about Z: midpoint (alpha=0.5) is 45 deg about Z
    q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q1 = np.array([math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)])
    q0_before = q0.copy()
    q1_before = q1.copy()

    q_mid_pos = slerp_quaternion_wxyz(q0, q1, 0.5)
    # Negated q1 has negative dot product with q0; shortest-path SLERP must flip sign
    q_mid_neg = slerp_quaternion_wxyz(q0, -q1, 0.5)
    assert np.array_equal(q0, q0_before)
    assert np.array_equal(q1, q1_before)

    expected_mid = np.array(
        [math.cos(math.pi / 8.0), 0.0, 0.0, math.sin(math.pi / 8.0)], dtype=np.float64
    )
    np.testing.assert_allclose(q_mid_pos, expected_mid, atol=1e-12)
    np.testing.assert_allclose(q_mid_neg, expected_mid, atol=1e-12)

    # Near-identical quaternions (dot > 0.9995) exercise linear fallback safely
    q_close = normalize_quaternion_wxyz(np.array([1.0, 1e-5, -1e-5, 2e-5]))
    q_interp = slerp_quaternion_wxyz(q0, q_close, 0.35)
    assert float(np.linalg.norm(q_interp)) == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.isfinite(q_interp))


def test_exact_midpoint_and_boundary_pose_interpolation(tmp_path: Path) -> None:
    sequence_root = tmp_path / "synthetic_seq"
    t0 = 1000
    t1 = 2000
    s = math.sqrt(0.5)
    content = (
        HEADER_17
        + _format_row_17(t0, (0.0, 2.0, -4.0), (1.0, 0.0, 0.0, 0.0))
        + _format_row_17(t1, (10.0, 6.0, 4.0), (s, 0.0, 0.0, s))
    )
    _write_gt_csv(sequence_root, content)
    gt = EuRoCGroundTruthSequence(sequence_root)

    # First valid timestamp (exact sample)
    first_interp = gt.interpolate_pose(t0)
    assert first_interp.exact_sample is True
    assert first_interp.timestamp_ns == t0
    assert first_interp.lower_timestamp_ns == t0
    assert first_interp.upper_timestamp_ns == t0
    assert first_interp.interpolation_fraction == pytest.approx(0.0)
    np.testing.assert_allclose(first_interp.position_world_body, [0.0, 2.0, -4.0])

    # Last valid timestamp (exact sample)
    last_interp = gt.interpolate_pose(t1)
    assert last_interp.exact_sample is True
    assert last_interp.timestamp_ns == t1
    assert last_interp.lower_timestamp_ns == t1
    assert last_interp.upper_timestamp_ns == t1
    assert last_interp.interpolation_fraction == pytest.approx(0.0)
    np.testing.assert_allclose(last_interp.position_world_body, [10.0, 6.0, 4.0])

    # Midpoint interpolation (t = 1500 -> alpha = 0.5)
    mid_interp = gt.interpolate_pose(1500)
    assert mid_interp.exact_sample is False
    assert mid_interp.timestamp_ns == 1500
    assert mid_interp.lower_timestamp_ns == t0
    assert mid_interp.upper_timestamp_ns == t1
    assert mid_interp.interpolation_fraction == pytest.approx(0.5)
    np.testing.assert_allclose(mid_interp.position_world_body, [5.0, 4.0, 0.0])

    expected_rot_z45 = np.array(
        [
            [math.cos(math.pi / 4.0), -math.sin(math.pi / 4.0), 0.0],
            [math.sin(math.pi / 4.0), math.cos(math.pi / 4.0), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        mid_interp.rotation_world_body, expected_rot_z45, atol=1e-12
    )

    # Out-of-range before first timestamp
    with pytest.raises(ValueError, match="precedes first available ground-truth"):
        gt.interpolate_pose(999)

    # Out-of-range after last timestamp
    with pytest.raises(ValueError, match="exceeds last available ground-truth"):
        gt.interpolate_pose(2001)


def test_single_sample_ground_truth_sequence(tmp_path: Path) -> None:
    sequence_root = tmp_path / "single_sample"
    _write_gt_csv(
        sequence_root,
        HEADER_17 + _format_row_17(5000, (1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0)),
    )
    gt = EuRoCGroundTruthSequence(sequence_root)
    exact = gt.interpolate_pose(5000)
    assert exact.exact_sample is True
    with pytest.raises(ValueError, match="precedes first available"):
        gt.interpolate_pose(4999)
    with pytest.raises(ValueError, match="exceeds last available"):
        gt.interpolate_pose(5001)


def test_missing_and_empty_ground_truth_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing ground-truth directory"):
        EuRoCGroundTruthSequence(tmp_path / "missing_seq")

    seq_no_csv = tmp_path / "no_csv"
    (seq_no_csv / "mav0" / "state_groundtruth_estimate0").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Missing ground-truth CSV file"):
        EuRoCGroundTruthSequence(seq_no_csv)

    seq_empty = tmp_path / "empty_csv"
    _write_gt_csv(seq_empty, HEADER_17 + "\n# only comments\n")
    with pytest.raises(ValueError, match="Empty ground-truth sequence"):
        EuRoCGroundTruthSequence(seq_empty)


@pytest.mark.parametrize(
    ("csv_body", "match_pattern"),
    [
        (
            _format_row_17(100, (0, 0, 0), (1, 0, 0, 0))
            + _format_row_17(100, (1, 0, 0), (1, 0, 0, 0)),
            "Duplicate ground-truth timestamp",
        ),
        (
            _format_row_17(200, (0, 0, 0), (1, 0, 0, 0))
            + _format_row_17(150, (1, 0, 0), (1, 0, 0, 0)),
            "Unordered ground-truth timestamp",
        ),
        (
            "100,1.0,2.0,3.0\n",
            "Malformed ground-truth CSV row",
        ),
        (
            "100,1.0,2.0,3.0,1.0,0.0,0.0,0.0,0.0\n",
            "Malformed ground-truth CSV row",
        ),
        (
            '100,"unterminated\n',
            "Malformed CSV",
        ),
        (
            _format_row_17("bad_ts", (0, 0, 0), (1, 0, 0, 0)),
            "Noninteger ground-truth timestamp",
        ),
        (
            _format_row_17(100, ("nan", 0, 0), (1, 0, 0, 0)),
            "Nonfinite ground-truth position",
        ),
        (
            _format_row_17(100, (0, "inf", 0), (1, 0, 0, 0)),
            "Nonfinite ground-truth position",
        ),
        (
            _format_row_17(100, (0, 0, 0), ("nan", 0, 0, 0)),
            "Nonfinite ground-truth quaternion",
        ),
        (
            _format_row_17(100, (0, 0, 0), (0.0, 0.0, 0.0, 0.0)),
            "Zero or near-zero quaternion norm",
        ),
    ],
)
def test_invalid_ground_truth_csv_contents(
    tmp_path: Path, csv_body: str, match_pattern: str
) -> None:
    sequence_root = tmp_path / "bad_seq"
    _write_gt_csv(sequence_root, HEADER_17 + csv_body)
    with pytest.raises(ValueError, match=match_pattern):
        EuRoCGroundTruthSequence(sequence_root)
