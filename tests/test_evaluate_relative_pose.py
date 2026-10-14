"""Script-level tests for evaluate_relative_pose.py and benchmark_relative_pose.py."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from tests.test_euroc_ground_truth import HEADER_17, _format_row_17

EXPECTED_EVALUATION_JSON_KEYS = {
    "sequence_path",
    "ground_truth_csv_path",
    "calibration_file",
    "reference_frame_index",
    "current_frame_index",
    "frame_gap",
    "reference_timestamp_ns",
    "current_timestamp_ns",
    "reference_gt_lower_timestamp_ns",
    "reference_gt_upper_timestamp_ns",
    "reference_interpolation_fraction",
    "reference_gt_exact_sample",
    "current_gt_lower_timestamp_ns",
    "current_gt_upper_timestamp_ns",
    "current_interpolation_fraction",
    "current_gt_exact_sample",
    "estimated_pose_success",
    "pose_quality_valid",
    "pose_quality_reason",
    "minimum_cheirality_count",
    "minimum_cheirality_ratio",
    "calibrated_correspondence_count",
    "essential_inlier_count",
    "cheirality_inlier_count",
    "cheirality_inlier_ratio",
    "estimated_relative_rotation_matrix",
    "ground_truth_relative_rotation_matrix",
    "estimated_rotation_angle_degrees",
    "ground_truth_rotation_angle_degrees",
    "rotation_error_degrees",
    "estimated_unit_translation_direction",
    "ground_truth_translation_vector_m",
    "ground_truth_unit_translation_direction",
    "ground_truth_translation_magnitude_m",
    "translation_direction_error_degrees",
    "translation_direction_defined",
    "translation_direction_undefined_reason",
    "translation_epsilon_m",
    "transform_convention",
    "translation_scale_status",
}


def _make_textured_frames(
    height: int = 480, width: int = 752
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(314)
    canvas = np.full((height + 24, width + 24), 28, dtype=np.uint8)
    for row in range(44, height - 44, 36):
        for col in range(44, width - 44, 36):
            val = int(rng.integers(90, 245))
            cv2.rectangle(canvas, (col, row), (col + 16, row + 16), val, -1)
            cv2.circle(canvas, (col + 8, row + 8), 4, 255 - val // 2, 1)
    f0 = canvas[12 : 12 + height, 12 : 12 + width].copy()
    f1 = canvas[12 : 12 + height, 12 : 12 + width].copy()
    f2 = canvas[10 : 10 + height, 7 : 7 + width].copy()
    f3 = canvas[10 : 10 + height, 7 : 7 + width].copy()
    return f0, f1, f2, f3


@pytest.fixture
def synthetic_sequence_with_gt(tmp_path: Path) -> Path:
    """Create a 4-frame synthetic EuRoC sequence with out-of-range boundary frames."""
    sequence_root = tmp_path / "synthetic_sequence"
    for camera in ("cam0", "cam1"):
        (sequence_root / "mav0" / camera / "data").mkdir(parents=True)

    frames = _make_textured_frames()
    timestamps = (
        1000,  # frame 0: before GT start (1500)
        2000,  # frame 1: inside GT [1500, 3500]
        3000,  # frame 2: inside GT [1500, 3500]
        4000,  # frame 3: after GT end (3500)
    )

    for camera in ("cam0", "cam1"):
        rows = "".join(f"{ts},{ts}.png\n" for ts in timestamps)
        (sequence_root / "mav0" / camera / "data.csv").write_text(
            "#timestamp [ns],filename\n" + rows,
            encoding="utf-8",
        )
        for ts, img in zip(timestamps, frames, strict=True):
            assert cv2.imwrite(
                str(sequence_root / "mav0" / camera / "data" / f"{ts}.png"), img
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

    gt_dir = sequence_root / "mav0" / "state_groundtruth_estimate0"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_content = (
        HEADER_17
        + _format_row_17(1500, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        + _format_row_17(2500, (0.1, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        + _format_row_17(3500, (0.3, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    )
    (gt_dir / "data.csv").write_text(gt_content, encoding="utf-8")
    return sequence_root


def _run_eval_script(*args: str) -> subprocess.CompletedProcess[str]:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "evaluate_relative_pose.py"
    )
    return subprocess.run(
        [sys.executable, str(script_path), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _run_benchmark_script(*args: str) -> subprocess.CompletedProcess[str]:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "benchmark_relative_pose.py"
    )
    return subprocess.run(
        [sys.executable, str(script_path), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_evaluate_relative_pose_script_success(
    synthetic_sequence_with_gt: Path, tmp_path: Path
) -> None:
    dataset_files_before = sorted(synthetic_sequence_with_gt.rglob("*"))
    output_json = tmp_path / "outputs" / "eval" / "pair_001.json"

    result = _run_eval_script(
        "--sequence",
        str(synthetic_sequence_with_gt),
        "--frame-index",
        "1",
        "--frame-gap",
        "1",
        "--output",
        str(output_json),
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert output_json.is_file()
    assert output_json.stat().st_size > 0
    assert sorted(synthetic_sequence_with_gt.rglob("*")) == dataset_files_before

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert set(report.keys()) == EXPECTED_EVALUATION_JSON_KEYS
    assert report["reference_frame_index"] == 1
    assert report["current_frame_index"] == 2
    assert report["frame_gap"] == 1
    assert report["reference_interpolation_fraction"] == pytest.approx(0.5)
    assert report["current_interpolation_fraction"] == pytest.approx(0.5)
    assert report["translation_direction_defined"] is True
    assert report["ground_truth_translation_magnitude_m"] == pytest.approx(0.15)
    assert 0.0 <= report["rotation_error_degrees"] <= 180.0
    assert 0.0 <= report["translation_direction_error_degrees"] <= 180.0
    assert "unknown" in report["translation_scale_status"].lower()


def test_evaluate_relative_pose_script_rejects_out_of_range_and_invalid_inputs(
    synthetic_sequence_with_gt: Path, tmp_path: Path
) -> None:
    out_json = tmp_path / "outputs" / "fail.json"

    # Frame 0 timestamp (1000) precedes ground-truth range [1500, 3500]
    res_before = _run_eval_script(
        "--sequence",
        str(synthetic_sequence_with_gt),
        "--frame-index",
        "0",
        "--frame-gap",
        "1",
        "--output",
        str(out_json),
    )
    assert res_before.returncode != 0
    assert "precedes first available ground-truth" in res_before.stderr
    assert not out_json.exists()

    # Frame 2 -> 3: current frame timestamp (4000) exceeds GT range [1500, 3500]
    res_after = _run_eval_script(
        "--sequence",
        str(synthetic_sequence_with_gt),
        "--frame-index",
        "2",
        "--frame-gap",
        "1",
        "--output",
        str(out_json),
    )
    assert res_after.returncode != 0
    assert "exceeds last available ground-truth" in res_after.stderr
    assert not out_json.exists()

    # Invalid frame index and frame gap
    for idx, gap in (("-1", "1"), ("3", "1"), ("1", "0"), ("1", "-2")):
        res_bad = _run_eval_script(
            "--sequence",
            str(synthetic_sequence_with_gt),
            "--frame-index",
            idx,
            "--frame-gap",
            gap,
            "--output",
            str(out_json),
        )
        assert res_bad.returncode != 0
        assert not out_json.exists()

    # Missing sequence
    res_missing_seq = _run_eval_script(
        "--sequence",
        str(tmp_path / "does_not_exist"),
        "--frame-index",
        "1",
        "--output",
        str(out_json),
    )
    assert res_missing_seq.returncode != 0

    # Missing ground truth
    (
        synthetic_sequence_with_gt
        / "mav0"
        / "state_groundtruth_estimate0"
        / "data.csv"
    ).unlink()
    res_missing_gt = _run_eval_script(
        "--sequence",
        str(synthetic_sequence_with_gt),
        "--frame-index",
        "1",
        "--output",
        str(out_json),
    )
    assert res_missing_gt.returncode != 0
    assert "FileNotFoundError" in res_missing_gt.stderr


def test_benchmark_relative_pose_script_handles_gt_unavailable_and_valid_pairs(
    synthetic_sequence_with_gt: Path, tmp_path: Path
) -> None:
    csv_out = tmp_path / "outputs" / "bench.csv"
    json_out = tmp_path / "outputs" / "bench.json"

    result = _run_benchmark_script(
        "--sequence",
        str(synthetic_sequence_with_gt),
        "--start-frame",
        "0",
        "--end-frame",
        "2",
        "--stride",
        "1",
        "--frame-gaps",
        "1",
        "--csv-output",
        str(csv_out),
        "--json-output",
        str(json_out),
    )

    assert result.returncode == 0, result.stderr
    assert csv_out.is_file() and csv_out.stat().st_size > 0
    assert json_out.is_file() and json_out.stat().st_size > 0

    with csv_out.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    # Pair 0 -> 1 is ground_truth_unavailable
    assert rows[0]["ground_truth_available"] == "False"
    assert rows[0]["failure_category"] == "ground_truth_unavailable"
    # Pair 1 -> 2 is inside ground truth and evaluated
    assert rows[1]["ground_truth_available"] == "True"
    assert rows[1]["estimation_success"] == "True"
    # Pair 2 -> 3 is ground_truth_unavailable (frame 3 exceeds GT)
    assert rows[2]["ground_truth_available"] == "False"
    assert rows[2]["failure_category"] == "ground_truth_unavailable"

    summary = json.loads(json_out.read_text(encoding="utf-8"))
    gap1 = summary["per_frame_gap"][0]
    assert gap1["total_requested_pairs"] == 3
    assert gap1["ground_truth_unavailable_count"] == 2
    assert gap1["evaluated_pairs"] == 1
