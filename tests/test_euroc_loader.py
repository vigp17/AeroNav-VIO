"""Synthetic EuRoC loader tests."""

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import cv2
import numpy as np
import pytest

from aeronav_vio.data.euroc import EurocSequence


@pytest.fixture
def sequence_root(tmp_path: Path) -> Path:
    """Create a synthetic EuRoC sequence with empty camera data folders."""
    sequence_root = tmp_path / "synthetic_sequence"
    for camera in ("cam0", "cam1"):
        (sequence_root / "mav0" / camera / "data").mkdir(parents=True)
    return sequence_root


def _write_csv(sequence_root: Path, camera: str, *rows: str) -> None:
    (sequence_root / "mav0" / camera / "data.csv").write_text(
        "#timestamp [ns],filename\n" + "".join(f"{row}\n" for row in rows),
        encoding="utf-8",
    )


def _write_pair_csvs(sequence_root: Path) -> None:
    _write_csv(sequence_root, "cam0", "100,left.png")
    _write_csv(sequence_root, "cam1", "100,right.png")


def _write_image(
    sequence_root: Path, camera: str, filename: str, pixels: np.ndarray
) -> None:
    image_path = sequence_root / "mav0" / camera / "data" / filename
    assert cv2.imwrite(str(image_path), pixels)


def _inspect(sequence_root: Path) -> subprocess.CompletedProcess[str]:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "inspect_euroc.py"
    return subprocess.run(
        [sys.executable, str(script_path), "--sequence", str(sequence_root)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_parses_pairs_in_timestamp_order(sequence_root: Path) -> None:
    """Pair exact nanosecond timestamps without loading any images."""
    _write_csv(
        sequence_root,
        "cam0",
        "1403636579763555587,late.png",
        "",
        "1403636579763555584,early.png",
        "1403636579763555586,left_only.png",
    )
    _write_csv(
        sequence_root,
        "cam1",
        "1403636579763555587,right_late.png",
        "1403636579763555585,right_only.png",
        "1403636579763555584,right_early.png",
    )

    sequence = EurocSequence(sequence_root)

    assert sequence.cam0_record_count == 3
    assert sequence.cam1_record_count == 3
    assert sequence.synchronized_pair_count == 2
    assert [frame.timestamp_ns for frame in sequence.frames] == [
        1403636579763555584,
        1403636579763555587,
    ]
    assert sequence.frames[0].left_image_path == (
        sequence_root / "mav0" / "cam0" / "data" / "early.png"
    )
    assert sequence.frames[0].right_image_path == (
        sequence_root / "mav0" / "cam1" / "data" / "right_early.png"
    )
    assert sequence.unmatched_cam0_timestamps == (1403636579763555586,)
    assert sequence.unmatched_cam1_timestamps == (1403636579763555585,)


def test_stereo_frame_and_properties_are_read_only(sequence_root: Path) -> None:
    _write_pair_csvs(sequence_root)
    sequence = EurocSequence(sequence_root)

    assert isinstance(sequence.frames, tuple)
    with pytest.raises(FrozenInstanceError):
        sequence.frames[0].timestamp_ns = 101
    with pytest.raises(AttributeError):
        sequence.frames = ()
    with pytest.raises(AttributeError):
        sequence.cam0_record_count = 0


@pytest.mark.parametrize("missing_camera", ["cam0", "cam1"])
def test_missing_camera_directory(tmp_path: Path, missing_camera: str) -> None:
    sequence_root = tmp_path / "synthetic_sequence"
    other_camera = "cam1" if missing_camera == "cam0" else "cam0"
    (sequence_root / "mav0" / other_camera / "data").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Missing camera directory") as error:
        EurocSequence(sequence_root)
    assert missing_camera in str(error.value)


@pytest.mark.parametrize("missing_camera", ["cam0", "cam1"])
def test_missing_csv_file(sequence_root: Path, missing_camera: str) -> None:
    other_camera = "cam1" if missing_camera == "cam0" else "cam0"
    _write_csv(sequence_root, other_camera, "100,image.png")

    with pytest.raises(FileNotFoundError, match="Missing camera CSV file") as error:
        EurocSequence(sequence_root)
    assert missing_camera in str(error.value)


@pytest.mark.parametrize(
    "row", ["100", "100,", ",left.png", "100,left.png,extra", '100,"unterminated']
)
def test_malformed_csv_rows(sequence_root: Path, row: str) -> None:
    _write_csv(sequence_root, "cam0", row)
    _write_csv(sequence_root, "cam1", "100,right.png")

    with pytest.raises(ValueError, match="Malformed CSV"):
        EurocSequence(sequence_root)


@pytest.mark.parametrize("timestamp", ["not-an-integer", "100.5"])
def test_noninteger_timestamp(sequence_root: Path, timestamp: str) -> None:
    _write_csv(sequence_root, "cam0", f"{timestamp},left.png")
    _write_csv(sequence_root, "cam1", "100,right.png")

    with pytest.raises(ValueError, match="Noninteger timestamp") as error:
        EurocSequence(sequence_root)
    assert timestamp in str(error.value)
    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize("camera", ["cam0", "cam1"])
def test_duplicate_timestamps(sequence_root: Path, camera: str) -> None:
    _write_pair_csvs(sequence_root)
    _write_csv(sequence_root, camera, "100,first.png", "100,second.png")

    with pytest.raises(ValueError, match="Duplicate timestamp 100") as error:
        EurocSequence(sequence_root)
    assert camera in str(error.value)


@pytest.mark.parametrize("camera", ["cam0", "cam1"])
def test_empty_camera_sequence(sequence_root: Path, camera: str) -> None:
    _write_pair_csvs(sequence_root)
    _write_csv(sequence_root, camera, "", "# another comment", "  ")

    with pytest.raises(ValueError, match="Empty camera sequence") as error:
        EurocSequence(sequence_root)
    assert camera in str(error.value)


def test_no_synchronized_pairs(sequence_root: Path) -> None:
    _write_csv(sequence_root, "cam0", "100,left.png")
    _write_csv(sequence_root, "cam1", "101,right.png")

    with pytest.raises(ValueError, match="No synchronized stereo pairs"):
        EurocSequence(sequence_root)


def test_load_stereo_pair_preserves_dimensions(sequence_root: Path) -> None:
    _write_pair_csvs(sequence_root)
    _write_image(sequence_root, "cam0", "left.png", np.full((3, 5, 3), 31, np.uint8))
    _write_image(sequence_root, "cam1", "right.png", np.full((7, 2, 3), 167, np.uint8))

    left, right = EurocSequence(sequence_root).load_stereo_pair(0)

    assert isinstance(left, np.ndarray)
    assert isinstance(right, np.ndarray)
    assert left.shape == (3, 5)
    assert right.shape == (7, 2)
    assert left.dtype == np.uint8
    assert right.dtype == np.uint8
    np.testing.assert_array_equal(left, np.full((3, 5), 31, np.uint8))
    np.testing.assert_array_equal(right, np.full((7, 2), 167, np.uint8))


@pytest.mark.parametrize("missing_camera", ["cam0", "cam1"])
def test_missing_image_file(sequence_root: Path, missing_camera: str) -> None:
    _write_pair_csvs(sequence_root)
    other_camera = "cam1" if missing_camera == "cam0" else "cam0"
    filename = "left.png" if other_camera == "cam0" else "right.png"
    _write_image(sequence_root, other_camera, filename, np.zeros((2, 3), np.uint8))

    with pytest.raises(FileNotFoundError, match="Missing image file") as error:
        EurocSequence(sequence_root).load_stereo_pair(0)
    assert missing_camera in str(error.value)


@pytest.mark.parametrize("unreadable_camera", ["cam0", "cam1"])
def test_unreadable_image_file(sequence_root: Path, unreadable_camera: str) -> None:
    _write_pair_csvs(sequence_root)
    for camera, filename in (("cam0", "left.png"), ("cam1", "right.png")):
        if camera == unreadable_camera:
            (sequence_root / "mav0" / camera / "data" / filename).write_bytes(
                b"not an image"
            )
        else:
            _write_image(sequence_root, camera, filename, np.zeros((2, 3), np.uint8))

    with pytest.raises(ValueError, match="Unreadable image file") as error:
        EurocSequence(sequence_root).load_stereo_pair(0)
    assert unreadable_camera in str(error.value)


@pytest.mark.parametrize("index", [-1, 1, 99])
def test_invalid_frame_index(sequence_root: Path, index: int) -> None:
    _write_pair_csvs(sequence_root)

    with pytest.raises(IndexError, match=f"Frame index {index} out of range"):
        EurocSequence(sequence_root).load_stereo_pair(index)


def test_inspection_cli_reports_first_pair_only(sequence_root: Path) -> None:
    _write_csv(
        sequence_root, "cam0", "200,later.png", "100,left.png", "300,left_only.png"
    )
    _write_csv(
        sequence_root, "cam1", "100,right.png", "400,right_only.png", "200,later.png"
    )
    _write_image(sequence_root, "cam0", "left.png", np.zeros((3, 5, 3), np.uint8))
    _write_image(sequence_root, "cam1", "right.png", np.zeros((7, 2, 3), np.uint8))

    result = _inspect(sequence_root)

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert f"Sequence path: {sequence_root.resolve()}" in result.stdout
    assert "cam0 record count: 3" in result.stdout
    assert "cam1 record count: 3" in result.stdout
    assert "Synchronized stereo-pair count: 2" in result.stdout
    assert "Unmatched cam0 timestamp count: 1" in result.stdout
    assert "Unmatched cam1 timestamp count: 1" in result.stdout
    assert "First synchronized timestamp (ns): 100" in result.stdout
    assert "Last synchronized timestamp (ns): 200" in result.stdout
    assert "First left-image resolution: 5x3" in result.stdout
    assert "First right-image resolution: 2x7" in result.stdout
    assert "First left-image NumPy data type: uint8" in result.stdout
    assert "First right-image NumPy data type: uint8" in result.stdout


def test_inspection_cli_preserves_parse_error(sequence_root: Path) -> None:
    _write_csv(sequence_root, "cam0", "bad,left.png")
    _write_csv(sequence_root, "cam1", "100,right.png")

    result = _inspect(sequence_root)

    assert result.returncode != 0
    assert result.stdout == ""
    assert "ValueError: Noninteger timestamp 'bad'" in result.stderr
    assert "caused by ValueError:" in result.stderr