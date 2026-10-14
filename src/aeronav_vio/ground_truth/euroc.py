"""Parse and interpolate EuRoC ground-truth poses and convert to cam0 poses."""

import csv
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aeronav_vio.geometry.relative_pose import validate_rotation_matrix

_QUATERNION_MIN_NORM = 1e-12
_SLERP_DOT_LINEAR_THRESHOLD = 0.9995
_HOMOGENEOUS_ROW_ATOL = 1e-6
_SUPPORTED_GROUND_TRUTH_COLUMN_COUNTS = (8, 17)


def _readonly_float64(array: np.ndarray) -> np.ndarray:
    """Return a read-only float64 copy of a NumPy array."""
    out = np.array(array, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class GroundTruthPose:
    """One EuRoC ground-truth body pose sample in the world reference frame.

    Transform convention:
        X_world = rotation_world_body @ X_body + position_world_body
    Quaternion component order is scalar-first:
        quaternion_world_body_wxyz = [w, x, y, z]
    """

    timestamp_ns: int
    position_world_body: np.ndarray
    quaternion_world_body_wxyz: np.ndarray
    rotation_world_body: np.ndarray


@dataclass(frozen=True)
class InterpolatedGroundTruthPose:
    """Interpolated EuRoC body pose at a requested camera timestamp.

    Transform convention:
        X_world = rotation_world_body @ X_body + position_world_body
    Quaternion component order is scalar-first:
        quaternion_world_body_wxyz = [w, x, y, z]
    """

    timestamp_ns: int
    position_world_body: np.ndarray
    quaternion_world_body_wxyz: np.ndarray
    rotation_world_body: np.ndarray
    lower_timestamp_ns: int
    upper_timestamp_ns: int
    interpolation_fraction: float
    exact_sample: bool


@dataclass(frozen=True)
class CameraPose:
    """Camera pose in the world reference frame (T_world_camera).

    Transform convention:
        X_world = rotation_world_camera @ X_camera + position_world_camera
    """

    timestamp_ns: int
    position_world_camera: np.ndarray
    rotation_world_camera: np.ndarray
    transform_world_camera: np.ndarray


def normalize_quaternion_wxyz(
    quaternion_wxyz: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Validate and normalize a scalar-first [w, x, y, z] quaternion."""
    try:
        arr = np.asarray(quaternion_wxyz, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Quaternion must contain numeric components") from error

    if arr.shape != (4,):
        raise ValueError(
            f"Expected quaternion_wxyz with shape (4,), got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Quaternion contains nonfinite values")

    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm <= _QUATERNION_MIN_NORM:
        raise ValueError(f"Zero or near-zero quaternion norm: {norm:.3e}")

    normalized = arr / norm
    final_norm = float(np.linalg.norm(normalized))
    if not math.isfinite(final_norm) or not math.isclose(
        final_norm, 1.0, abs_tol=1e-6
    ):
        raise ValueError("Failed to normalize quaternion to unit length")
    return _readonly_float64(normalized)


def quaternion_wxyz_to_rotation_matrix(
    quaternion_wxyz: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Convert a scalar-first [w, x, y, z] Hamilton quaternion to a 3x3 matrix."""
    q = normalize_quaternion_wxyz(quaternion_wxyz)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rotation = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    validate_rotation_matrix(rotation)
    return _readonly_float64(rotation)


def slerp_quaternion_wxyz(
    q0_wxyz: np.ndarray | list[float] | tuple[float, ...],
    q1_wxyz: np.ndarray | list[float] | tuple[float, ...],
    alpha: float,
) -> np.ndarray:
    """Interpolate scalar-first [w, x, y, z] quaternions via shortest-path SLERP."""
    if isinstance(alpha, bool):
        raise TypeError("Interpolation fraction alpha must be a real number")
    alpha_val = float(alpha)
    if not math.isfinite(alpha_val) or not (0.0 <= alpha_val <= 1.0):
        raise ValueError(
            f"Interpolation fraction alpha must be in [0.0, 1.0], got {alpha!r}"
        )

    q0 = normalize_quaternion_wxyz(q0_wxyz)
    q1 = np.array(normalize_quaternion_wxyz(q1_wxyz), dtype=np.float64, copy=True)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot_clamped = float(np.clip(dot, -1.0, 1.0))
    if dot_clamped > _SLERP_DOT_LINEAR_THRESHOLD:
        blended = (1.0 - alpha_val) * q0 + alpha_val * q1
        return normalize_quaternion_wxyz(blended)

    theta_0 = math.acos(dot_clamped)
    sin_theta_0 = math.sin(theta_0)
    if abs(sin_theta_0) <= _QUATERNION_MIN_NORM:
        blended = (1.0 - alpha_val) * q0 + alpha_val * q1
        return normalize_quaternion_wxyz(blended)

    theta = alpha_val * theta_0
    weight_0 = math.sin(theta_0 - theta) / sin_theta_0
    weight_1 = math.sin(theta) / sin_theta_0
    interpolated = weight_0 * q0 + weight_1 * q1
    return normalize_quaternion_wxyz(interpolated)


def validate_homogeneous_transform(
    transform: np.ndarray, *, label: str = "transform"
) -> None:
    """Validate that transform is a finite 4x4 SE(3) homogeneous matrix."""
    if transform.shape != (4, 4):
        raise ValueError(f"Expected {label} shape (4, 4), got {transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{label} contains nonfinite values")
    if not np.allclose(
        transform[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=_HOMOGENEOUS_ROW_ATOL
    ):
        raise ValueError(
            f"Invalid final row in {label}: expected [0, 0, 0, 1], "
            f"got {transform[3, :].tolist()}"
        )
    validate_rotation_matrix(transform[:3, :3])


def invert_rigid_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a validated 4x4 SE(3) rigid transform analytically."""
    validate_homogeneous_transform(transform, label="transform")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    rotation_inv = rotation.T
    translation_inv = -(rotation_inv @ translation)

    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation_inv
    inverse[:3, 3] = translation_inv
    validate_homogeneous_transform(inverse, label="inverse_transform")
    return _readonly_float64(inverse)


def body_pose_to_camera_pose(
    body_pose: GroundTruthPose | InterpolatedGroundTruthPose,
    body_to_sensor_transform: np.ndarray,
) -> CameraPose:
    """Convert a world-to-body pose (T_WB) into a world-to-camera pose (T_WC).

    EuRoC transform definitions:
        - body_pose represents T_WB (body frame B wrt world frame W):
              X_W = R_WB @ X_B + p_WB
        - body_to_sensor_transform is EuRoC T_BS (sensor frame C wrt body frame B):
              X_B = R_BC @ X_C + p_BC
    Composition:
        T_WC = T_WB @ T_BC
        R_WC = R_WB @ R_BC
        p_WC = R_WB @ p_BC + p_WB
        X_W  = R_WC @ X_C + p_WC
    """
    validate_homogeneous_transform(
        body_to_sensor_transform, label="body_to_sensor_transform"
    )
    if body_pose.position_world_body.shape != (3,):
        raise ValueError("body_pose.position_world_body must have shape (3,)")
    if not np.all(np.isfinite(body_pose.position_world_body)):
        raise ValueError("body_pose.position_world_body contains nonfinite values")
    validate_rotation_matrix(body_pose.rotation_world_body)

    t_wb = np.eye(4, dtype=np.float64)
    t_wb[:3, :3] = body_pose.rotation_world_body
    t_wb[:3, 3] = body_pose.position_world_body
    validate_homogeneous_transform(t_wb, label="transform_world_body")

    t_wc = t_wb @ body_to_sensor_transform
    validate_homogeneous_transform(t_wc, label="transform_world_camera")

    position_wc = _readonly_float64(t_wc[:3, 3])
    rotation_wc = _readonly_float64(t_wc[:3, :3])
    return CameraPose(
        timestamp_ns=body_pose.timestamp_ns,
        position_world_camera=position_wc,
        rotation_world_camera=rotation_wc,
        transform_world_camera=_readonly_float64(t_wc),
    )


def _parse_ground_truth_csv(csv_path: Path) -> tuple[GroundTruthPose, ...]:
    """Parse and validate EuRoC state_groundtruth_estimate0/data.csv."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing ground-truth CSV file: {csv_path}")

    samples: list[GroundTruthPose] = []
    expected_field_count: int | None = None
    previous_timestamp_ns: int | None = None

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, strict=True)
        try:
            for row in reader:
                if not row or all(not cell.strip() for cell in row):
                    continue
                if row[0].strip().startswith("#"):
                    continue

                col_count = len(row)
                if col_count not in _SUPPORTED_GROUND_TRUTH_COLUMN_COUNTS:
                    raise ValueError(
                        f"Malformed ground-truth CSV row in {csv_path} at line "
                        f"{reader.line_num}: expected 8 or 17 columns, got {col_count}"
                    )
                if expected_field_count is None:
                    expected_field_count = col_count
                elif col_count != expected_field_count:
                    raise ValueError(
                        f"Inconsistent column count in {csv_path} at line "
                        f"{reader.line_num}: expected {expected_field_count}, "
                        f"got {col_count}"
                    )

                if any(not cell.strip() for cell in row):
                    raise ValueError(
                        f"Empty field in ground-truth CSV {csv_path} at line "
                        f"{reader.line_num}"
                    )

                try:
                    timestamp_ns = int(row[0].strip())
                except ValueError as error:
                    raise ValueError(
                        f"Noninteger ground-truth timestamp {row[0]!r} in "
                        f"{csv_path} at line {reader.line_num}"
                    ) from error

                if previous_timestamp_ns is not None:
                    if timestamp_ns == previous_timestamp_ns:
                        raise ValueError(
                            f"Duplicate ground-truth timestamp {timestamp_ns} in "
                            f"{csv_path} at line {reader.line_num}"
                        )
                    if timestamp_ns < previous_timestamp_ns:
                        raise ValueError(
                            f"Unordered ground-truth timestamp {timestamp_ns} "
                            f"(after {previous_timestamp_ns}) in {csv_path} at "
                            f"line {reader.line_num}"
                        )
                previous_timestamp_ns = timestamp_ns

                try:
                    numeric_fields = [float(cell.strip()) for cell in row[1:]]
                except ValueError as error:
                    raise ValueError(
                        f"Nonnumeric ground-truth field in {csv_path} at line "
                        f"{reader.line_num}"
                    ) from error

                pos_vals = numeric_fields[0:3]
                quat_vals = numeric_fields[3:7]
                extra_vals = numeric_fields[7:]

                if not all(math.isfinite(val) for val in pos_vals):
                    raise ValueError(
                        f"Nonfinite ground-truth position in {csv_path} at line "
                        f"{reader.line_num}"
                    )
                if not all(math.isfinite(val) for val in quat_vals):
                    raise ValueError(
                        f"Nonfinite ground-truth quaternion in {csv_path} at line "
                        f"{reader.line_num}"
                    )
                if not all(math.isfinite(val) for val in extra_vals):
                    raise ValueError(
                        f"Nonfinite ground-truth auxiliary field in {csv_path} at "
                        f"line {reader.line_num}"
                    )

                position = _readonly_float64(np.array(pos_vals, dtype=np.float64))
                try:
                    quaternion_wxyz = normalize_quaternion_wxyz(quat_vals)
                    rotation_wb = quaternion_wxyz_to_rotation_matrix(quaternion_wxyz)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid ground-truth quaternion in {csv_path} at line "
                        f"{reader.line_num}: {error}"
                    ) from error

                samples.append(
                    GroundTruthPose(
                        timestamp_ns=timestamp_ns,
                        position_world_body=position,
                        quaternion_world_body_wxyz=quaternion_wxyz,
                        rotation_world_body=rotation_wb,
                    )
                )
        except csv.Error as error:
            raise ValueError(
                f"Malformed CSV in {csv_path} at line {reader.line_num}: {error}"
            ) from error

    if not samples:
        raise ValueError(f"Empty ground-truth sequence: {csv_path}")
    return tuple(samples)


class EuRoCGroundTruthSequence:
    """Validated EuRoC ground-truth body-pose trajectory with SLERP interpolation."""

    def __init__(self, sequence_root: Path) -> None:
        """Load and validate state_groundtruth_estimate0/data.csv."""
        gt_dir = Path(sequence_root) / "mav0" / "state_groundtruth_estimate0"
        if not gt_dir.is_dir():
            raise FileNotFoundError(
                f"Missing ground-truth directory: {gt_dir}"
            )
        self._csv_path = gt_dir / "data.csv"
        self._poses = _parse_ground_truth_csv(self._csv_path)
        self._timestamps_ns = tuple(pose.timestamp_ns for pose in self._poses)

    @property
    def csv_path(self) -> Path:
        """Path to the ground-truth data.csv file."""
        return self._csv_path

    @property
    def poses(self) -> tuple[GroundTruthPose, ...]:
        """All validated ground-truth body poses in increasing timestamp order."""
        return self._poses

    @property
    def sample_count(self) -> int:
        """Total number of ground-truth samples."""
        return len(self._poses)

    @property
    def first_timestamp_ns(self) -> int:
        """First available ground-truth timestamp in integer nanoseconds."""
        return self._timestamps_ns[0]

    @property
    def last_timestamp_ns(self) -> int:
        """Last available ground-truth timestamp in integer nanoseconds."""
        return self._timestamps_ns[-1]

    def contains_timestamp(self, timestamp_ns: int) -> bool:
        """Return True if timestamp_ns lies within the ground-truth range."""
        if isinstance(timestamp_ns, bool):
            raise TypeError("timestamp_ns must be an integer")
        return self.first_timestamp_ns <= timestamp_ns <= self.last_timestamp_ns

    def interpolate_pose(self, timestamp_ns: int) -> InterpolatedGroundTruthPose:
        """Interpolate ground-truth body pose at timestamp_ns without extrapolation."""
        if isinstance(timestamp_ns, bool):
            raise TypeError(
                f"Requested timestamp_ns must be an integer, got {timestamp_ns!r}"
            )

        first_ts = self.first_timestamp_ns
        last_ts = self.last_timestamp_ns
        if timestamp_ns < first_ts:
            delta_s = (first_ts - timestamp_ns) * 1e-9
            raise ValueError(
                f"Requested timestamp {timestamp_ns} ns precedes first available "
                f"ground-truth timestamp {first_ts} ns by {delta_s:.6f} s "
                "(ground-truth extrapolation is not permitted)"
            )
        if timestamp_ns > last_ts:
            delta_s = (timestamp_ns - last_ts) * 1e-9
            raise ValueError(
                f"Requested timestamp {timestamp_ns} ns exceeds last available "
                f"ground-truth timestamp {last_ts} ns by {delta_s:.6f} s "
                "(ground-truth extrapolation is not permitted)"
            )

        idx = bisect_left(self._timestamps_ns, timestamp_ns)
        if idx < len(self._timestamps_ns) and self._timestamps_ns[idx] == timestamp_ns:
            sample = self._poses[idx]
            return InterpolatedGroundTruthPose(
                timestamp_ns=timestamp_ns,
                position_world_body=sample.position_world_body,
                quaternion_world_body_wxyz=sample.quaternion_world_body_wxyz,
                rotation_world_body=sample.rotation_world_body,
                lower_timestamp_ns=sample.timestamp_ns,
                upper_timestamp_ns=sample.timestamp_ns,
                interpolation_fraction=0.0,
                exact_sample=True,
            )

        lower_sample = self._poses[idx - 1]
        upper_sample = self._poses[idx]
        t0 = lower_sample.timestamp_ns
        t1 = upper_sample.timestamp_ns
        alpha = (timestamp_ns - t0) / (t1 - t0)

        interp_pos = _readonly_float64(
            (1.0 - alpha) * lower_sample.position_world_body
            + alpha * upper_sample.position_world_body
        )
        interp_quat = slerp_quaternion_wxyz(
            lower_sample.quaternion_world_body_wxyz,
            upper_sample.quaternion_world_body_wxyz,
            alpha,
        )
        interp_rot = quaternion_wxyz_to_rotation_matrix(interp_quat)

        return InterpolatedGroundTruthPose(
            timestamp_ns=timestamp_ns,
            position_world_body=interp_pos,
            quaternion_world_body_wxyz=interp_quat,
            rotation_world_body=interp_rot,
            lower_timestamp_ns=t0,
            upper_timestamp_ns=t1,
            interpolation_fraction=float(alpha),
            exact_sample=False,
        )

    def interpolate_camera_pose(
        self, timestamp_ns: int, body_to_sensor_transform: np.ndarray
    ) -> tuple[InterpolatedGroundTruthPose, CameraPose]:
        """Interpolate body pose at timestamp_ns and convert it to a cam0 pose."""
        interpolated_body = self.interpolate_pose(timestamp_ns)
        camera_pose = body_pose_to_camera_pose(
            interpolated_body, body_to_sensor_transform
        )
        return interpolated_body, camera_pose
