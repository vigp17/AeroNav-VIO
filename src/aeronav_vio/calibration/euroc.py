"""Parse and validate EuRoC cam0 sensor.yaml calibration files."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SUPPORTED_CAMERA_MODEL = "pinhole"
SUPPORTED_DISTORTION_MODEL = "radial-tangential"
REQUIRED_CALIBRATION_KEYS = (
    "camera_model",
    "distortion_model",
    "resolution",
    "rate_hz",
    "intrinsics",
    "distortion_coefficients",
    "T_BS",
)


@dataclass(frozen=True)
class CameraCalibration:
    """Validated EuRoC camera calibration parameters and matrices."""

    sensor_label: str
    sensor_type: str
    camera_model: str
    distortion_model: str
    resolution: tuple[int, int]
    rate_hz: float
    intrinsics: tuple[float, float, float, float]
    distortion_coefficients: tuple[float, ...]
    camera_matrix: np.ndarray
    distortion_array: np.ndarray
    body_to_sensor_transform: np.ndarray
    calibration_file: Path

    @property
    def fx(self) -> float:
        """Horizontal focal length in pixels."""
        return self.intrinsics[0]

    @property
    def fy(self) -> float:
        """Vertical focal length in pixels."""
        return self.intrinsics[1]

    @property
    def cx(self) -> float:
        """Horizontal principal point coordinate in pixels."""
        return self.intrinsics[2]

    @property
    def cy(self) -> float:
        """Vertical principal point coordinate in pixels."""
        return self.intrinsics[3]


def _is_real_number(value: Any) -> bool:
    """Return True if value is an int or float (excluding bool)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_strict_int(value: Any) -> bool:
    """Return True if value is an integer (excluding bool)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_resolution(raw_resolution: Any, calibration_file: Path) -> tuple[int, int]:
    """Validate and return (width, height) resolution tuple."""
    if not isinstance(raw_resolution, Sequence) or isinstance(
        raw_resolution, (str, bytes)
    ):
        raise ValueError(
            f"Invalid resolution in {calibration_file}: expected 2-element sequence"
        )
    if len(raw_resolution) != 2:
        raise ValueError(
            f"Invalid resolution length {len(raw_resolution)} in {calibration_file}: "
            "expected [width, height]"
        )
    width_raw, height_raw = raw_resolution[0], raw_resolution[1]
    if not _is_strict_int(width_raw) or not _is_strict_int(height_raw):
        raise ValueError(
            f"Noninteger resolution {raw_resolution!r} in {calibration_file}: "
            "expected positive integers"
        )
    width, height = int(width_raw), int(height_raw)
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Nonpositive resolution {(width, height)} in {calibration_file}: "
            "width and height must be > 0"
        )
    return width, height


def _parse_rate_hz(raw_rate: Any, calibration_file: Path) -> float:
    """Validate and return camera rate in Hz."""
    if not _is_real_number(raw_rate):
        raise ValueError(
            f"Nonnumeric rate_hz {raw_rate!r} in {calibration_file}: "
            "expected positive finite number"
        )
    rate_hz = float(raw_rate)
    if not math.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError(
            f"Invalid rate_hz {raw_rate!r} in {calibration_file}: "
            "expected positive finite number"
        )
    return rate_hz


def _parse_intrinsics(
    raw_intrinsics: Any, calibration_file: Path
) -> tuple[tuple[float, float, float, float], np.ndarray]:
    """Validate [fx, fy, cx, cy] and construct the 3x3 intrinsic matrix K."""
    if not isinstance(raw_intrinsics, Sequence) or isinstance(
        raw_intrinsics, (str, bytes)
    ):
        raise ValueError(
            f"Invalid intrinsics in {calibration_file}: expected [fx, fy, cx, cy]"
        )
    if len(raw_intrinsics) != 4:
        raise ValueError(
            f"Invalid intrinsics count {len(raw_intrinsics)} in {calibration_file}: "
            "expected 4 values [fx, fy, cx, cy]"
        )

    values: list[float] = []
    for index, item in enumerate(raw_intrinsics):
        if not _is_real_number(item):
            raise ValueError(
                f"Nonnumeric intrinsic at index {index} in {calibration_file}: "
                f"{item!r}"
            )
        val = float(item)
        if not math.isfinite(val):
            raise ValueError(
                f"Nonfinite intrinsic at index {index} in {calibration_file}: "
                f"{item!r}"
            )
        values.append(val)

    fx, fy, cx, cy = values[0], values[1], values[2], values[3]
    if fx <= 0.0:
        raise ValueError(
            f"Nonpositive fx ({fx}) in {calibration_file}: focal length must be > 0"
        )
    if fy <= 0.0:
        raise ValueError(
            f"Nonpositive fy ({fy}) in {calibration_file}: focal length must be > 0"
        )

    camera_matrix = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    camera_matrix.setflags(write=False)
    return (fx, fy, cx, cy), camera_matrix


def _parse_distortion(
    raw_distortion: Any, calibration_file: Path
) -> tuple[tuple[float, ...], np.ndarray]:
    """Validate radial-tangential [k1, k2, p1, p2] distortion coefficients."""
    if not isinstance(raw_distortion, Sequence) or isinstance(
        raw_distortion, (str, bytes)
    ):
        raise ValueError(
            f"Invalid distortion_coefficients in {calibration_file}: "
            "expected [k1, k2, p1, p2]"
        )
    if len(raw_distortion) != 4:
        raise ValueError(
            f"Invalid distortion_coefficients count {len(raw_distortion)} in "
            f"{calibration_file}: expected 4 values [k1, k2, p1, p2]"
        )

    coeffs: list[float] = []
    for index, item in enumerate(raw_distortion):
        if not _is_real_number(item):
            raise ValueError(
                f"Nonnumeric distortion coefficient at index {index} in "
                f"{calibration_file}: {item!r}"
            )
        val = float(item)
        if not math.isfinite(val):
            raise ValueError(
                f"Nonfinite distortion coefficient at index {index} in "
                f"{calibration_file}: {item!r}"
            )
        coeffs.append(val)

    distortion_array = np.array(coeffs, dtype=np.float64)
    distortion_array.setflags(write=False)
    return tuple(coeffs), distortion_array


def _parse_transform_t_bs(raw_t_bs: Any, calibration_file: Path) -> np.ndarray:
    """Validate and construct the 4x4 homogeneous body-to-sensor transform T_BS."""
    if not isinstance(raw_t_bs, Mapping):
        raise ValueError(
            f"Invalid T_BS in {calibration_file}: expected mapping with rows, cols, "
            "and data"
        )

    rows = raw_t_bs.get("rows")
    cols = raw_t_bs.get("cols")
    data = raw_t_bs.get("data")

    if not _is_strict_int(rows) or not _is_strict_int(cols) or rows != 4 or cols != 4:
        raise ValueError(
            f"Invalid T_BS shape ({rows!r}, {cols!r}) in {calibration_file}: "
            "expected 4x4 transform"
        )
    if (
        not isinstance(data, Sequence)
        or isinstance(data, (str, bytes))
        or len(data) != 16
    ):
        raise ValueError(
            f"Invalid T_BS data length in {calibration_file}: expected 16 numeric "
            "elements for 4x4 transform"
        )

    flat_values: list[float] = []
    for index, item in enumerate(data):
        if not _is_real_number(item):
            raise ValueError(
                f"Nonnumeric T_BS value at index {index} in {calibration_file}: "
                f"{item!r}"
            )
        val = float(item)
        if not math.isfinite(val):
            raise ValueError(
                f"Nonfinite T_BS value at index {index} in {calibration_file}: "
                f"{item!r}"
            )
        flat_values.append(val)

    transform = np.array(flat_values, dtype=np.float64).reshape((4, 4))
    if not np.allclose(transform[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise ValueError(
            f"Invalid homogeneous transform final row {transform[3, :].tolist()} "
            f"in {calibration_file}: expected [0, 0, 0, 1]"
        )

    rotation_block = transform[:3, :3]
    orthonormal_error = np.max(np.abs(rotation_block.T @ rotation_block - np.eye(3)))
    determinant = float(np.linalg.det(rotation_block))
    if orthonormal_error > 1e-3 or abs(determinant - 1.0) > 1e-3:
        raise ValueError(
            f"Invalid rotation block in T_BS of {calibration_file}: "
            f"orthonormality max error={orthonormal_error:.3e}, det={determinant:.6f}"
        )

    transform.setflags(write=False)
    return transform


def load_camera_calibration_file(
    calibration_file: Path, *, sensor_label: str = "cam0"
) -> CameraCalibration:
    """Load and validate a EuRoC camera sensor.yaml file in read-only mode."""
    if not isinstance(calibration_file, Path):
        calibration_file = Path(calibration_file)
    if not calibration_file.is_file():
        raise FileNotFoundError(
            f"Missing camera calibration file: {calibration_file}"
        )

    try:
        with calibration_file.open("r", encoding="utf-8") as handle:
            raw_data = yaml.safe_load(handle)
    except yaml.YAMLError as error:
        raise ValueError(
            f"Invalid YAML syntax in {calibration_file}: {error}"
        ) from error

    if not isinstance(raw_data, Mapping):
        raise ValueError(
            f"Expected YAML root mapping in {calibration_file}, "
            f"got {type(raw_data).__name__}"
        )

    missing_keys = [key for key in REQUIRED_CALIBRATION_KEYS if key not in raw_data]
    if missing_keys:
        missing_joined = ", ".join(missing_keys)
        raise ValueError(
            f"Missing required calibration key(s) in {calibration_file}: "
            f"{missing_joined}"
        )

    sensor_type_raw = raw_data.get("sensor_type", "camera")
    if not isinstance(sensor_type_raw, str) or sensor_type_raw != "camera":
        raise ValueError(
            f"Unsupported sensor_type {sensor_type_raw!r} in {calibration_file}: "
            "expected 'camera'"
        )

    camera_model = raw_data["camera_model"]
    if not isinstance(camera_model, str) or camera_model != SUPPORTED_CAMERA_MODEL:
        raise ValueError(
            f"Unsupported camera_model {camera_model!r} in {calibration_file}: "
            f"expected {SUPPORTED_CAMERA_MODEL!r}"
        )

    distortion_model = raw_data["distortion_model"]
    if (
        not isinstance(distortion_model, str)
        or distortion_model != SUPPORTED_DISTORTION_MODEL
    ):
        raise ValueError(
            f"Unsupported distortion_model {distortion_model!r} in "
            f"{calibration_file}: expected {SUPPORTED_DISTORTION_MODEL!r}"
        )

    resolution = _parse_resolution(raw_data["resolution"], calibration_file)
    rate_hz = _parse_rate_hz(raw_data["rate_hz"], calibration_file)
    intrinsics, camera_matrix = _parse_intrinsics(
        raw_data["intrinsics"], calibration_file
    )
    distortion_coefficients, distortion_array = _parse_distortion(
        raw_data["distortion_coefficients"], calibration_file
    )
    t_bs = _parse_transform_t_bs(raw_data["T_BS"], calibration_file)

    comment_raw = raw_data.get("comment")
    resolved_label = (
        comment_raw.strip()
        if isinstance(comment_raw, str) and comment_raw.strip()
        else sensor_label
    )

    return CameraCalibration(
        sensor_label=resolved_label,
        sensor_type=sensor_type_raw,
        camera_model=camera_model,
        distortion_model=distortion_model,
        resolution=resolution,
        rate_hz=rate_hz,
        intrinsics=intrinsics,
        distortion_coefficients=distortion_coefficients,
        camera_matrix=camera_matrix,
        distortion_array=distortion_array,
        body_to_sensor_transform=t_bs,
        calibration_file=calibration_file,
    )


def load_cam0_calibration(sequence_root: Path) -> CameraCalibration:
    """Load and validate <sequence_root>/mav0/cam0/sensor.yaml."""
    if not isinstance(sequence_root, Path):
        sequence_root = Path(sequence_root)
    calibration_file = sequence_root / "mav0" / "cam0" / "sensor.yaml"
    return load_camera_calibration_file(calibration_file, sensor_label="cam0")


def validate_image_calibration_resolution(
    calibration: CameraCalibration, image: np.ndarray, *, label: str = "image"
) -> None:
    """Validate that a grayscale image's (height, width) matches calibration."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected {label} to be a NumPy array, got {type(image)!r}")
    if image.ndim != 2 or image.size == 0:
        raise ValueError(
            f"Expected nonempty 2D grayscale array for {label}, got shape "
            f"{image.shape}"
        )
    image_height, image_width = int(image.shape[0]), int(image.shape[1])
    expected_width, expected_height = calibration.resolution
    if (image_width, image_height) != (expected_width, expected_height):
        raise ValueError(
            f"{label} resolution ({image_width}x{image_height}) does not match "
            f"calibration resolution ({expected_width}x{expected_height})"
        )

