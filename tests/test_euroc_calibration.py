"""Synthetic unit tests for EuRoC cam0 sensor.yaml calibration parsing."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from aeronav_vio.calibration.euroc import (
    load_cam0_calibration,
    load_camera_calibration_file,
    validate_image_calibration_resolution,
)


def _valid_calibration_dict() -> dict[str, Any]:
    return {
        "sensor_type": "camera",
        "comment": "VI-Sensor cam0 (synthetic)",
        "T_BS": {
            "cols": 4,
            "rows": 4,
            "data": [
                0.0148655429818,
                -0.999880929698,
                0.00414029679422,
                -0.0216401454975,
                0.999557249008,
                0.0149672133247,
                0.025715529948,
                -0.064676986768,
                -0.0257744366974,
                0.00375618835797,
                0.999660727178,
                0.00981073058949,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        },
        "rate_hz": 20,
        "resolution": [752, 480],
        "camera_model": "pinhole",
        "intrinsics": [
            458.6548828125123,
            457.29669189453125,
            367.21582031250006,
            248.37530517578128,
        ],
        "distortion_model": "radial-tangential",
        "distortion_coefficients": [
            -0.2834081121702931,
            0.07395907389291234,
            0.0001935950287123,
            1.761871145432e-05,
        ],
    }


def _write_sensor_yaml(sequence_root: Path, payload: Any) -> Path:
    cam0_dir = sequence_root / "mav0" / "cam0"
    cam0_dir.mkdir(parents=True, exist_ok=True)
    sensor_yaml = cam0_dir / "sensor.yaml"
    sensor_yaml.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return sensor_yaml


def test_load_cam0_calibration_success_and_precision(tmp_path: Path) -> None:
    sequence_root = tmp_path / "synthetic_sequence"
    payload = _valid_calibration_dict()
    expected_path = _write_sensor_yaml(sequence_root, payload)

    calib = load_cam0_calibration(sequence_root)

    assert calib.calibration_file == expected_path
    assert calib.sensor_type == "camera"
    assert calib.sensor_label == "VI-Sensor cam0 (synthetic)"
    assert calib.camera_model == "pinhole"
    assert calib.distortion_model == "radial-tangential"
    assert calib.resolution == (752, 480)
    assert calib.rate_hz == 20.0

    fx, fy, cx, cy = payload["intrinsics"]
    assert calib.intrinsics == (fx, fy, cx, cy)
    assert (calib.fx, calib.fy, calib.cx, calib.cy) == (fx, fy, cx, cy)
    expected_k = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    np.testing.assert_array_equal(calib.camera_matrix, expected_k)

    k1, k2, p1, p2 = payload["distortion_coefficients"]
    assert calib.distortion_coefficients == (k1, k2, p1, p2)
    np.testing.assert_array_equal(
        calib.distortion_array, np.array([k1, k2, p1, p2], dtype=np.float64)
    )

    expected_t_bs = np.array(payload["T_BS"]["data"], dtype=np.float64).reshape((4, 4))
    np.testing.assert_array_equal(calib.body_to_sensor_transform, expected_t_bs)

    with pytest.raises(FrozenInstanceError):
        calib.rate_hz = 30.0  # type: ignore[misc]
    with pytest.raises(ValueError):
        calib.camera_matrix[0, 0] = 1.0


def test_validate_image_calibration_resolution(tmp_path: Path) -> None:
    sequence_root = tmp_path / "synthetic_sequence"
    _write_sensor_yaml(sequence_root, _valid_calibration_dict())
    calib = load_cam0_calibration(sequence_root)

    valid_image = np.zeros((480, 752), dtype=np.uint8)
    validate_image_calibration_resolution(calib, valid_image)

    transposed_image = np.zeros((752, 480), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not match calibration resolution"):
        validate_image_calibration_resolution(calib, transposed_image)


def test_missing_calibration_file(tmp_path: Path) -> None:
    sequence_root = tmp_path / "missing_seq"
    with pytest.raises(FileNotFoundError, match="Missing camera calibration file"):
        load_cam0_calibration(sequence_root)


def test_invalid_yaml_syntax(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "sensor.yaml"
    bad_yaml.write_text("camera_model: [unterminated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML syntax"):
        load_camera_calibration_file(bad_yaml)


@pytest.mark.parametrize("root_content", ["[1, 2, 3]\n", "'just_a_string'\n", "42\n"])
def test_invalid_yaml_root(tmp_path: Path, root_content: str) -> None:
    bad_yaml = tmp_path / "sensor.yaml"
    bad_yaml.write_text(root_content, encoding="utf-8")
    with pytest.raises(ValueError, match="Expected YAML root mapping"):
        load_camera_calibration_file(bad_yaml)


@pytest.mark.parametrize(
    "missing_key",
    [
        "camera_model",
        "distortion_model",
        "resolution",
        "rate_hz",
        "intrinsics",
        "distortion_coefficients",
        "T_BS",
    ],
)
def test_missing_required_keys(tmp_path: Path, missing_key: str) -> None:
    payload = _valid_calibration_dict()
    del payload[missing_key]
    yaml_path = _write_sensor_yaml(tmp_path, payload)
    with pytest.raises(
        ValueError, match=f"Missing required calibration key.*{missing_key}"
    ):
        load_camera_calibration_file(yaml_path)


def test_unsupported_camera_and_distortion_models(tmp_path: Path) -> None:
    payload = _valid_calibration_dict()
    payload["camera_model"] = "omnidirectional"
    yaml_path = _write_sensor_yaml(tmp_path / "cam_model", payload)
    with pytest.raises(ValueError, match="Unsupported camera_model"):
        load_camera_calibration_file(yaml_path)

    payload2 = _valid_calibration_dict()
    payload2["distortion_model"] = "equidistant"
    yaml_path2 = _write_sensor_yaml(tmp_path / "dist_model", payload2)
    with pytest.raises(ValueError, match="Unsupported distortion_model"):
        load_camera_calibration_file(yaml_path2)


@pytest.mark.parametrize(
    ("bad_intrinsics", "match_pattern"),
    [
        ([458.0, 457.0, 367.0], "Invalid intrinsics count"),
        ([458.0, 457.0, 367.0, 248.0, 1.0], "Invalid intrinsics count"),
        (["bad", 457.0, 367.0, 248.0], "Nonnumeric intrinsic"),
        ([float("nan"), 457.0, 367.0, 248.0], "Nonfinite intrinsic"),
        ([458.0, float("inf"), 367.0, 248.0], "Nonfinite intrinsic"),
        ([0.0, 457.0, 367.0, 248.0], "Nonpositive fx"),
        ([-458.0, 457.0, 367.0, 248.0], "Nonpositive fx"),
        ([458.0, 0.0, 367.0, 248.0], "Nonpositive fy"),
        ([458.0, -457.0, 367.0, 248.0], "Nonpositive fy"),
    ],
)
def test_invalid_intrinsics(
    tmp_path: Path, bad_intrinsics: list[Any], match_pattern: str
) -> None:
    payload = _valid_calibration_dict()
    payload["intrinsics"] = bad_intrinsics
    yaml_path = _write_sensor_yaml(tmp_path, payload)
    with pytest.raises(ValueError, match=match_pattern):
        load_camera_calibration_file(yaml_path)


@pytest.mark.parametrize(
    ("bad_resolution", "match_pattern"),
    [
        ([752], "Invalid resolution length"),
        ([752, 480, 3], "Invalid resolution length"),
        ([0, 480], "Nonpositive resolution"),
        ([752, -480], "Nonpositive resolution"),
        ([752.5, 480], "Noninteger resolution"),
        (["752", 480], "Noninteger resolution"),
        ([True, 480], "Noninteger resolution"),
    ],
)
def test_invalid_resolution(
    tmp_path: Path, bad_resolution: list[Any], match_pattern: str
) -> None:
    payload = _valid_calibration_dict()
    payload["resolution"] = bad_resolution
    yaml_path = _write_sensor_yaml(tmp_path, payload)
    with pytest.raises(ValueError, match=match_pattern):
        load_camera_calibration_file(yaml_path)


@pytest.mark.parametrize(
    ("bad_distortion", "match_pattern"),
    [
        ([-0.28, 0.07, 0.0001], "Invalid distortion_coefficients count"),
        ([-0.28, 0.07, 0.0001, 1e-5, 0.0], "Invalid distortion_coefficients count"),
        ([-0.28, float("nan"), 0.0001, 1e-5], "Nonfinite distortion coefficient"),
        ([-0.28, "bad", 0.0001, 1e-5], "Nonnumeric distortion coefficient"),
    ],
)
def test_invalid_distortion_coefficients(
    tmp_path: Path, bad_distortion: list[Any], match_pattern: str
) -> None:
    payload = _valid_calibration_dict()
    payload["distortion_coefficients"] = bad_distortion
    yaml_path = _write_sensor_yaml(tmp_path, payload)
    with pytest.raises(ValueError, match=match_pattern):
        load_camera_calibration_file(yaml_path)


def test_invalid_transform_t_bs(tmp_path: Path) -> None:
    # Wrong shape metadata
    payload_shape = _valid_calibration_dict()
    payload_shape["T_BS"]["rows"] = 3
    yaml_shape = _write_sensor_yaml(tmp_path / "shape", payload_shape)
    with pytest.raises(ValueError, match="Invalid T_BS shape"):
        load_camera_calibration_file(yaml_shape)

    # Wrong data length
    payload_len = _valid_calibration_dict()
    payload_len["T_BS"]["data"] = [1.0] * 12
    yaml_len = _write_sensor_yaml(tmp_path / "len", payload_len)
    with pytest.raises(ValueError, match="Invalid T_BS data length"):
        load_camera_calibration_file(yaml_len)

    # Nonfinite value
    payload_nan = _valid_calibration_dict()
    payload_nan["T_BS"]["data"][0] = float("nan")
    yaml_nan = _write_sensor_yaml(tmp_path / "nan", payload_nan)
    with pytest.raises(ValueError, match="Nonfinite T_BS value"):
        load_camera_calibration_file(yaml_nan)

    # Invalid final row
    payload_row = _valid_calibration_dict()
    payload_row["T_BS"]["data"][15] = 2.0
    yaml_row = _write_sensor_yaml(tmp_path / "row", payload_row)
    with pytest.raises(ValueError, match="Invalid homogeneous transform final row"):
        load_camera_calibration_file(yaml_row)

    # Invalid rotation block (reflection with det == -1)
    payload_rot = _valid_calibration_dict()
    payload_rot["T_BS"]["data"] = [
        -1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    yaml_rot = _write_sensor_yaml(tmp_path / "rot", payload_rot)
    with pytest.raises(ValueError, match="Invalid rotation block in T_BS"):
        load_camera_calibration_file(yaml_rot)


@pytest.mark.parametrize(
    ("bad_rate", "match_pattern"),
    [
        ("20Hz", "Nonnumeric rate_hz"),
        (True, "Nonnumeric rate_hz"),
        (0, "Invalid rate_hz"),
        (-20.0, "Invalid rate_hz"),
        (float("inf"), "Invalid rate_hz"),
    ],
)
def test_invalid_camera_rate(
    tmp_path: Path, bad_rate: Any, match_pattern: str
) -> None:
    payload = _valid_calibration_dict()
    payload["rate_hz"] = bad_rate
    yaml_path = _write_sensor_yaml(tmp_path, payload)
    with pytest.raises(ValueError, match=match_pattern):
        load_camera_calibration_file(yaml_path)
