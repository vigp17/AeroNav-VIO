"""Read synchronized stereo frames from one EuRoC sequence."""

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class StereoFrame:
    """Paths to one stereo pair at an integer nanosecond timestamp."""

    timestamp_ns: int
    left_image_path: Path
    right_image_path: Path


def _parse_camera_records(camera_dir: Path) -> dict[int, Path]:
    """Index camera image paths by timestamp without reading image pixels."""
    csv_path = camera_dir / "data.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing camera CSV file: {csv_path}")

    records: dict[int, Path] = {}
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, strict=True)
        try:
            for row in reader:
                if not row or all(not field.strip() for field in row):
                    continue
                if row[0].strip().startswith("#"):
                    continue
                if len(row) != 2 or not row[0].strip() or not row[1].strip():
                    raise ValueError(
                        f"Malformed CSV row in {csv_path} at line {reader.line_num}: "
                        "expected timestamp,filename"
                    )

                try:
                    timestamp_ns = int(row[0].strip())
                except ValueError as error:
                    raise ValueError(
                        f"Noninteger timestamp {row[0]!r} in {csv_path} "
                        f"at line {reader.line_num}"
                    ) from error
                if timestamp_ns in records:
                    raise ValueError(
                        f"Duplicate timestamp {timestamp_ns} in {csv_path} "
                        f"at line {reader.line_num}"
                    )
                records[timestamp_ns] = camera_dir / "data" / row[1].strip()
        except csv.Error as error:
            raise ValueError(
                f"Malformed CSV in {csv_path} at line {reader.line_num}: {error}"
            ) from error

    if not records:
        raise ValueError(f"Empty camera sequence: {csv_path}")
    return records


def _read_grayscale_image(image_path: Path) -> np.ndarray:
    """Read one image without resizing it."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image file: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unreadable image file: {image_path}")
    return image


class EurocSequence:
    """Index EuRoC stereo metadata; load image pairs only on request."""

    def __init__(self, sequence_root: Path) -> None:
        """Validate camera directories and index their CSV records."""
        cam0_dir = sequence_root / "mav0" / "cam0"
        cam1_dir = sequence_root / "mav0" / "cam1"
        for camera_dir in (cam0_dir, cam1_dir):
            if not camera_dir.is_dir():
                raise FileNotFoundError(f"Missing camera directory: {camera_dir}")

        cam0_records = _parse_camera_records(cam0_dir)
        cam1_records = _parse_camera_records(cam1_dir)
        self._cam0_record_count = len(cam0_records)
        self._cam1_record_count = len(cam1_records)
        common_timestamps = sorted(cam0_records.keys() & cam1_records.keys())
        if not common_timestamps:
            raise ValueError(f"No synchronized stereo pairs in {sequence_root}")

        self._frames = tuple(
    StereoFrame(
        timestamp_ns,
        cam0_records[timestamp_ns],
        cam1_records[timestamp_ns],
    )
    for timestamp_ns in common_timestamps
)
        self._unmatched_cam0_timestamps = tuple(
            sorted(cam0_records.keys() - cam1_records.keys())
        )
        self._unmatched_cam1_timestamps = tuple(
            sorted(cam1_records.keys() - cam0_records.keys())
        )

    @property
    def frames(self) -> tuple[StereoFrame, ...]:
        """Synchronized frames sorted by timestamp."""
        return self._frames

    @property
    def cam0_record_count(self) -> int:
        """Number of cam0 CSV records."""
        return self._cam0_record_count

    @property
    def cam1_record_count(self) -> int:
        """Number of cam1 CSV records."""
        return self._cam1_record_count

    @property
    def synchronized_pair_count(self) -> int:
        """Number of exact timestamp matches."""
        return len(self._frames)

    @property
    def unmatched_cam0_timestamps(self) -> tuple[int, ...]:
        """Cam0 timestamps with no matching cam1 record."""
        return self._unmatched_cam0_timestamps

    @property
    def unmatched_cam1_timestamps(self) -> tuple[int, ...]:
        """Cam1 timestamps with no matching cam0 record."""
        return self._unmatched_cam1_timestamps

    def load_left_image(self, index: int) -> np.ndarray:
        """Load the cam0 left image for a synchronized frame index as grayscale."""
        if index < 0 or index >= len(self._frames):
            raise IndexError(
                f"Frame index {index} out of range for {len(self._frames)} pairs"
            )
        return _read_grayscale_image(self._frames[index].left_image_path)

    def load_stereo_pair(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Load a selected stereo pair as original-size grayscale arrays."""
        if index < 0 or index >= len(self._frames):
            raise IndexError(
                f"Frame index {index} out of range for {len(self._frames)} pairs"
            )
        frame = self._frames[index]
        return (
            _read_grayscale_image(frame.left_image_path),
            _read_grayscale_image(frame.right_image_path),
        )