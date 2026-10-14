"""Script-level tests for scripts/estimate_relative_pose.py using synthetic data."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from scripts.estimate_relative_pose import to_json_compatible

EXPECTED_JSON_KEYS = {
    "sequence_path",
    "calibration_file",
    "reference_frame_index",
    "current_frame_index",
    "frame_gap",
    "reference_timestamp_ns",
    "current_timestamp_ns",
    "timestamp_difference_ns",
    "camera_model",
    "distortion_model",
    "resolution",
    "rate_hz",
    "fx",
    "fy",
    "cx",
    "cy",
    "camera_matrix",
    "distortion_coefficients",
    "body_to_sensor_transform",
    "reference_keypoint_count",
    "current_keypoint_count",
    "ratio_filtered_match_count",
    "milestone2_geometric_inlier_count",
    "calibrated_input_correspondence_count",
    "essential_inlier_count",
    "cheirality_inlier_count",
    "essential_inlier_ratio",
    "cheirality_inlier_ratio",
    "minimum_cheirality_count",
    "minimum_cheirality_ratio",
    "pose_quality_valid",
    "pose_quality_reason",
    "essential_matrix",
    "relative_rotation_matrix",
    "rotation_determinant",
    "unit_translation_direction",
    "translation_norm",
    "rotation_angle_degrees",
    "transform_convention",
    "translation_scale_status",
}


def _make_textured_sequence_frames(
    height: int = 480, width: int = 752
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(314)
    canvas = np.full((height + 24, width + 24), 28, dtype=np.uint8)
    for row in range(44, height - 44, 36):
        for col in range(44, width - 44, 36):
            val = int(rng.integers(90, 245))
            cv2.rectangle(canvas, (col, row), (col + 16, row + 16), val, -1)
            cv2.circle(canvas, (col + 8, row + 8), 4, 255 - val // 2, 1)
    frame0 = canvas[12 : 12 + height, 12 : 12 + width].copy()
    frame1 = canvas[10 : 10 + height, 7 : 7 + width].copy()
    frame2 = canvas[10 : 10 + height, 7 : 7 + width].copy()
    return frame0, frame1, frame2


@pytest.fixture
def synthetic_calibrated_sequence(tmp_path: Path) -> Path:
    sequence_root = tmp_path / "synthetic_sequence"
    for camera in ("cam0", "cam1"):
        (sequence_root / "mav0" / camera / "data").mkdir(parents=True)

    frame0, frame1, frame2 = _make_textured_sequence_frames()
    ts0 = 1403636579763555584
    ts1 = 1403636579813555456
    ts2 = 1403636579863555328

    for camera in ("cam0", "cam1"):
        (sequence_root / "mav0" / camera / "data.csv").write_text(
            "#timestamp [ns],filename\n"
            f"{ts0},{ts0}.png\n"
            f"{ts1},{ts1}.png\n"
            f"{ts2},{ts2}.png\n",
            encoding="utf-8",
        )
        assert cv2.imwrite(
            str(sequence_root / "mav0" / camera / "data" / f"{ts0}.png"), frame0
        )
        assert cv2.imwrite(
            str(sequence_root / "mav0" / camera / "data" / f"{ts1}.png"), frame1
        )
        assert cv2.imwrite(
            str(sequence_root / "mav0" / camera / "data" / f"{ts2}.png"), frame2
        )

    sensor_payload = {
        "sensor_type": "camera",
        "comment": "VI-Sensor cam0",
        "T_BS": {
            "cols": 4,
            "rows": 4,
            "data": [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        },
        "rate_hz": 20,
        "resolution": [752, 480],
        "camera_model": "pinhole",
        "intrinsics": [458.654, 457.296, 367.215, 248.375],
        "distortion_model": "radial-tangential",
        "distortion_coefficients": [
            -0.28340811,
            0.07395907,
            0.00019359,
            1.76187114e-05,
        ],
    }
    (sequence_root / "mav0" / "cam0" / "sensor.yaml").write_text(
        yaml.safe_dump(sensor_payload), encoding="utf-8"
    )
    return sequence_root


def _run_pose_script(*args: str) -> subprocess.CompletedProcess[str]:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "estimate_relative_pose.py"
    )
    return subprocess.run(
        [sys.executable, str(script_path), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_estimate_relative_pose_script_writes_valid_json(
    synthetic_calibrated_sequence: Path, tmp_path: Path
) -> None:
    dataset_files_before = sorted(synthetic_calibrated_sequence.rglob("*"))
    output_json = tmp_path / "outputs" / "nested" / "pose_000000.json"

    result = _run_pose_script(
        "--sequence",
        str(synthetic_calibrated_sequence),
        "--frame-index",
        "0",
        "--output",
        str(output_json),
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert output_json.is_file()
    assert output_json.stat().st_size > 0
    assert sorted(synthetic_calibrated_sequence.rglob("*")) == dataset_files_before

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert set(report.keys()) == EXPECTED_JSON_KEYS
    assert report["reference_frame_index"] == 0
    assert report["current_frame_index"] == 1
    assert report["frame_gap"] == 1
    assert isinstance(report["pose_quality_valid"], bool)
    assert isinstance(report["pose_quality_reason"], str)
    assert "unknown" in report["translation_scale_status"].lower()
    assert (
        report["calibrated_input_correspondence_count"]
        == report["milestone2_geometric_inlier_count"]
    )
    assert (
        0
        < report["cheirality_inlier_count"]
        <= report["essential_inlier_count"]
        <= report["calibrated_input_correspondence_count"]
    )
    assert report["rotation_determinant"] == pytest.approx(1.0, abs=1e-4)
    assert report["translation_norm"] == pytest.approx(1.0, abs=1e-6)
    assert "Frame gap: 1" in result.stdout
    assert "Pose quality valid:" in result.stdout
    assert "Translation scale status: unknown" in result.stdout


def test_estimate_relative_pose_script_explicit_frame_gap_and_quality_thresholds(
    synthetic_calibrated_sequence: Path, tmp_path: Path
) -> None:
    output_json = tmp_path / "outputs" / "pose_gap2.json"

    result = _run_pose_script(
        "--sequence",
        str(synthetic_calibrated_sequence),
        "--frame-index",
        "0",
        "--frame-gap",
        "2",
        "--min-cheirality-count",
        "999",
        "--min-cheirality-ratio",
        "0.99",
        "--output",
        str(output_json),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["reference_frame_index"] == 0
    assert report["current_frame_index"] == 2
    assert report["frame_gap"] == 2
    assert report["timestamp_difference_ns"] == 99999744
    assert report["minimum_cheirality_count"] == 999
    assert report["minimum_cheirality_ratio"] == pytest.approx(0.99)
    assert report["pose_quality_valid"] is False
    assert "Low-confidence relative pose" in report["pose_quality_reason"]
    assert "Frame gap: 2" in result.stdout
    assert "Pose quality valid: False" in result.stdout


def test_estimate_relative_pose_script_failures(
    synthetic_calibrated_sequence: Path, tmp_path: Path
) -> None:
    output_json = tmp_path / "outputs" / "pose_fail.json"

    # Missing sequence
    res_missing_seq = _run_pose_script(
        "--sequence",
        str(tmp_path / "nonexistent"),
        "--frame-index",
        "0",
        "--output",
        str(output_json),
    )
    assert res_missing_seq.returncode != 0
    assert not output_json.exists()

    # Zero or negative frame-gap
    for bad_gap in ("0", "-1", "-5"):
        res_bad_gap = _run_pose_script(
            "--sequence",
            str(synthetic_calibrated_sequence),
            "--frame-index",
            "0",
            "--frame-gap",
            bad_gap,
            "--output",
            str(output_json),
        )
        assert res_bad_gap.returncode != 0
        assert "ValueError" in res_bad_gap.stderr
        assert not output_json.exists()

    # Invalid reference frame index or current frame beyond sequence range (3 frames)
    for bad_idx, gap in (("-1", "1"), ("2", "1"), ("1", "2"), ("5", "1")):
        res_bad_idx = _run_pose_script(
            "--sequence",
            str(synthetic_calibrated_sequence),
            "--frame-index",
            bad_idx,
            "--frame-gap",
            gap,
            "--output",
            str(output_json),
        )
        assert res_bad_idx.returncode != 0
        assert "IndexError" in res_bad_idx.stderr
        assert not output_json.exists()

    # Missing calibration file
    (synthetic_calibrated_sequence / "mav0" / "cam0" / "sensor.yaml").unlink()
    res_missing_cal = _run_pose_script(
        "--sequence",
        str(synthetic_calibrated_sequence),
        "--frame-index",
        "0",
        "--output",
        str(output_json),
    )
    assert res_missing_cal.returncode != 0
    assert "FileNotFoundError" in res_missing_cal.stderr
    assert not output_json.exists()


def test_to_json_compatible_converts_numpy_and_rejects_nonfinite() -> None:
    payload = {
        "path": Path("outputs/pose.json"),
        "int_val": np.int64(42),
        "float_val": np.float64(3.5),
        "arr": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
    }
    converted = to_json_compatible(payload)
    assert converted["path"] == str(Path("outputs/pose.json"))
    assert isinstance(converted["int_val"], int)
    assert isinstance(converted["float_val"], float)
    assert converted["arr"] == [[1.0, 2.0], [3.0, 4.0]]

    for bad_float in (float("nan"), float("inf"), float("-inf"), np.nan, np.inf):
        with pytest.raises(ValueError, match="Nonfinite float"):
            to_json_compatible({"bad": bad_float})
