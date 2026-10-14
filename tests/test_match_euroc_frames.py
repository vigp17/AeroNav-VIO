"""Script-level tests for scripts/match_euroc_frames.py using synthetic sequences."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


def _make_textured_pair(
    height: int = 240, width: int = 360
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(99)
    canvas = np.full((height + 16, width + 16), 25, dtype=np.uint8)
    for row in range(32, height - 32, 32):
        for col in range(32, width - 32, 32):
            val = int(rng.integers(80, 250))
            cv2.rectangle(canvas, (col, row), (col + 14, row + 14), val, -1)
    ref = canvas[8 : 8 + height, 8 : 8 + width].copy()
    cur = canvas[6 : 6 + height, 5 : 5 + width].copy()
    return ref, cur


@pytest.fixture
def synthetic_euroc_sequence(tmp_path: Path) -> Path:
    sequence_root = tmp_path / "synthetic_sequence"
    for camera in ("cam0", "cam1"):
        (sequence_root / "mav0" / camera / "data").mkdir(parents=True)

    ref, cur = _make_textured_pair()
    ts0 = 1403636579763555584
    ts1 = 1403636579813555456

    for camera in ("cam0", "cam1"):
        (sequence_root / "mav0" / camera / "data.csv").write_text(
            f"#timestamp [ns],filename\n{ts0},{ts0}.png\n{ts1},{ts1}.png\n",
            encoding="utf-8",
        )
        assert cv2.imwrite(
            str(sequence_root / "mav0" / camera / "data" / f"{ts0}.png"), ref
        )
        assert cv2.imwrite(
            str(sequence_root / "mav0" / camera / "data" / f"{ts1}.png"), cur
        )
    return sequence_root


def _run_match_script(*args: str) -> subprocess.CompletedProcess[str]:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "match_euroc_frames.py"
    )
    return subprocess.run(
        [sys.executable, str(script_path), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_match_script_creates_output_and_nested_directory(
    synthetic_euroc_sequence: Path, tmp_path: Path
) -> None:
    output_path = tmp_path / "nested_outputs" / "sub" / "matches_000000.png"

    result = _run_match_script(
        "--sequence",
        str(synthetic_euroc_sequence),
        "--frame-index",
        "0",
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    saved_image = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert saved_image is not None
    assert saved_image.shape[1] == 720

    assert f"Sequence path: {synthetic_euroc_sequence.resolve()}" in result.stdout
    assert "Reference frame index: 0" in result.stdout
    assert "Current frame index: 1" in result.stdout
    assert "Reference timestamp (ns): 1403636579763555584" in result.stdout
    assert "Current timestamp (ns): 1403636579813555456" in result.stdout
    assert "Timestamp difference (ns): 49999872" in result.stdout
    assert f"Output image path: {output_path.resolve()}" in result.stdout


@pytest.mark.parametrize("bad_index", ["-1", "1", "5"])
def test_match_script_rejects_invalid_reference_frame_index(
    synthetic_euroc_sequence: Path, tmp_path: Path, bad_index: str
) -> None:
    output_path = tmp_path / "outputs" / "should_not_exist.png"

    result = _run_match_script(
        "--sequence",
        str(synthetic_euroc_sequence),
        "--frame-index",
        bad_index,
        "--output",
        str(output_path),
    )

    assert result.returncode != 0
    assert "IndexError" in result.stderr
    assert not output_path.exists()


def test_match_script_fails_on_missing_sequence_path(tmp_path: Path) -> None:
    missing_seq = tmp_path / "does_not_exist"
    output_path = tmp_path / "outputs" / "matches.png"

    result = _run_match_script(
        "--sequence",
        str(missing_seq),
        "--frame-index",
        "0",
        "--output",
        str(output_path),
    )

    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
    assert not output_path.exists()
