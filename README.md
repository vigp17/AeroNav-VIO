# AeroNav-VIO

AeroNav-VIO aims to build and evaluate GPS-denied autonomous drone navigation
with stereo-inertial SLAM, occupancy mapping, A* planning, and
localization-aware control. PX4 SITL, ROS 2, and Gazebo belong to later stages.

## Current milestone: EuRoC stereo loading

Milestone 1 reads one EuRoC sequence, validates camera CSV records, pairs
cam0/cam1 frames at exactly matching integer nanosecond timestamps, and loads
requested image pairs as grayscale arrays. No camera images are decoded while
the CSV metadata is indexed.

## Windows setup

Use Windows PowerShell and Python 3.11 or newer (the current development
environment uses Python 3.14). From the repository root, create and activate a
virtual environment, then install the package and development dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

If PowerShell prevents activation, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` in that
PowerShell session, then retry activation. Installation includes NumPy, OpenCV,
pytest, and Ruff; no dataset download or copy is performed.

## EuRoC sequence

Point the loader at the **sequence root**, not at `mav0` or a camera folder:

```text
MH_01_easy/
	mav0/
		cam0/
			data.csv
			data/
				<image filename>.png
		cam1/
			data.csv
			data/
				<image filename>.png
```

Existing EuRoC calibration files and other directories remain untouched. For
the real dataset in this Windows setup, the sequence root is
`$HOME\Desktop\datasets\euroc\MH_01_easy`. Inspect it without modifying it:

```powershell
python scripts/inspect_euroc.py --sequence "$HOME\Desktop\datasets\euroc\MH_01_easy"
```

The command prints the normalized absolute path, camera record and unmatched
timestamp counts, synchronized pair count, first and last timestamps (ns),
and the first pair's resolution (width x height) and NumPy data type. It exits
nonzero with a path-specific error if the sequence or first image is invalid.

For Python use, `EurocSequence(Path(...))` exposes sorted `frames`,
`cam0_record_count`, `cam1_record_count`, `synchronized_pair_count`, and
`unmatched_cam0_timestamps` / `unmatched_cam1_timestamps`. Call
`load_stereo_pair(index)` to read just one left/right image pair as grayscale
NumPy arrays.

## Checks

Unit tests use synthetic sequences created under pytest's `tmp_path`; they do
not read or write the real EuRoC dataset.

```powershell
python -m pytest -q
python -m ruff check .
```

## Assumptions and limitations

- Each camera has a nonempty `data.csv` with `timestamp_ns,filename` records;
	blank lines and lines beginning with `#` are ignored. Filenames refer to
	images in that camera's `data` directory.
- Timestamps are parsed as integers and paired by exact equality. Frames and
	unmatched timestamp tuples are sorted by timestamp; unmatched images are
	not interpolated or included in `frames`.
- CSV metadata and image paths are held in memory, but image pixels are loaded
	only for a requested pair. The inspection command decodes only the first
	synchronized pair. OpenCV must be able to decode the referenced images.
- Image dimensions are preserved. Calibration, IMU, ground truth, and any
	other EuRoC files are not parsed or changed. No visual odometry, SLAM,
	mapping, planning, or flight control is implemented in this milestone.

The planned next milestone is calibration and inertial-stream validation in
preparation for stereo-inertial estimation; it is not implemented here.
