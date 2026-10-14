# AeroNav-VIO

AeroNav-VIO is a milestone-driven robotics project for developing and
evaluating GPS-denied autonomous drone navigation.

The planned system combines:

- Stereo-inertial localization
- Visual SLAM
- Occupancy mapping
- A* path planning
- Localization-aware flight control
- PX4 SITL, ROS 2, and Gazebo integration in later milestones

The project currently focuses on building and validating the computer-vision
and localization foundation before introducing simulation and closed-loop
flight control.

## Project status

### Completed Milestone 1: EuRoC stereo dataset loading

Milestone 1 provides a validated EuRoC sequence loader that:

- Parses `cam0` and `cam1` CSV files
- Preserves timestamps as integer nanoseconds
- Pairs stereo frames using exact timestamps
- Reports unmatched camera timestamps
- Loads requested images on demand
- Returns images as grayscale NumPy arrays
- Avoids loading the complete dataset into memory
- Validates malformed, missing, duplicate, and unreadable data
- Includes automated tests and a real-dataset inspection script

The main interfaces are:

- `EurocSequence`
- `EurocSequence.load_stereo_pair`
- `EurocSequence.load_left_image`

### Completed Milestone 2: ORB temporal feature matching

Milestone 2 provides an ORB-based temporal feature-matching pipeline for
EuRoC `cam0` images.

The implementation:

- Detects ORB keypoints and binary descriptors
- Matches descriptors using Hamming-distance KNN matching
- Applies Lowe's ratio test
- Estimates a fundamental matrix using RANSAC
- Separates geometrically consistent inliers from outliers
- Produces inlier-only diagnostic visualizations
- Reports matching and timing statistics
- Includes synthetic unit and script-level tests
- Runs successfully on real EuRoC `MH_01_easy` images

### Completed Milestone 3: Calibrated relative camera pose estimation

Milestone 3 builds calibrated two-view relative pose estimation on top of
Milestone 1 and Milestone 2.

The implementation:

- Loads and validates EuRoC `mav0/cam0/sensor.yaml` in read-only mode
- Constructs the pinhole camera intrinsic matrix
- Parses radial-tangential distortion coefficients
- Validates image resolution against the calibration
- Extracts only Milestone 2 geometric-inlier correspondences
- Undistorts matched pixel coordinates into normalized coordinates
- Estimates essential-matrix candidates with RANSAC
- Handles stacked essential-matrix candidates deterministically
- Recovers relative camera rotation and unit translation direction
- Applies a positive-depth cheirality check
- Validates rotation orthonormality and determinant
- Evaluates pose quality using configurable cheirality thresholds
- Rejects or flags unsupported relative-pose estimates
- Writes standards-compliant JSON reports under `outputs/`

Milestone 3 does not recover metric translation scale and does not accumulate a
multi-frame trajectory.

### Current focus: Milestone 4 preparation

Milestone 4 will extend the validated two-view pipeline toward quality-gated
multi-frame visual odometry and ground-truth trajectory evaluation.

Milestone 4 has not yet been implemented.

## Repository structure

```text
AeroNav-VIO/
├── .github/
│   └── copilot-instructions.md
├── configs/
├── docs/
├── outputs/
├── scripts/
│   ├── estimate_relative_pose.py
│   ├── inspect_euroc.py
│   └── match_euroc_frames.py
├── src/
│   └── aeronav_vio/
│       ├── calibration/
│       │   ├── __init__.py
│       │   └── euroc.py
│       ├── data/
│       │   └── euroc.py
│       ├── evaluation/
│       ├── features/
│       │   └── orb.py
│       └── geometry/
│           ├── __init__.py
│           └── relative_pose.py
├── tests/
│   ├── test_estimate_relative_pose.py
│   ├── test_euroc_calibration.py
│   ├── test_euroc_loader.py
│   ├── test_match_euroc_frames.py
│   ├── test_orb_features.py
│   └── test_relative_pose.py
├── .gitignore
├── pyproject.toml
└── README.md
```

Generated visualizations and JSON reports are written under `outputs/` and are
excluded from version control. Run the documented commands to reproduce them
locally.

## Development environment

The project is currently developed natively on Windows using:

- Windows PowerShell
- Python 3.14
- OpenCV 5.0.0
- Visual Studio Code
- GitHub Copilot
- Git

Python 3.11 or newer is required by the project configuration.

## Installation

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/YOUR_USERNAME/AeroNav-VIO.git
cd AeroNav-VIO
```

Replace `YOUR_USERNAME` with the GitHub account that owns the repository.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents environment activation, apply a temporary execution
policy to the current PowerShell process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Upgrade the packaging tools and install the project:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

Verify the environment:

```powershell
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -m pytest --version
python -m ruff --version
```

## Dataset

The project currently uses the EuRoC MAV `MH_01_easy` sequence.

The dataset must remain outside the Git repository.

Example Windows layout:

```text
C:\Users\<username>\Desktop\
├── AeroNav-VIO\
└── datasets\
    └── euroc\
        └── MH_01_easy\
            └── mav0\
                ├── cam0\
                │   ├── data.csv
                │   ├── sensor.yaml
                │   └── data\
                ├── cam1\
                │   ├── data.csv
                │   ├── sensor.yaml
                │   └── data\
                ├── imu0\
                ├── leica0\
                ├── state_groundtruth_estimate0\
                └── body.yaml
```

Commands must receive the sequence root containing `mav0`.

Correct:

```text
C:\Users\<username>\Desktop\datasets\euroc\MH_01_easy
```

Incorrect:

```text
C:\Users\<username>\Desktop\datasets\euroc\MH_01_easy\mav0
```

The project treats the dataset as read-only. Generated files are saved only
under the repository's `outputs/` directory.

# Milestone 1: EuRoC sequence loading

## Inspecting a sequence

Run:

```powershell
python scripts/inspect_euroc.py `
    --sequence "$HOME\Desktop\datasets\euroc\MH_01_easy"
```

The script reports:

- Normalized sequence path
- `cam0` record count
- `cam1` record count
- Synchronized stereo-pair count
- Unmatched timestamp counts
- First and last synchronized timestamps
- Left and right image resolutions
- Left and right NumPy data types

## Measured `MH_01_easy` inspection result

```text
cam0 record count: 3682
cam1 record count: 3682
Synchronized stereo-pair count: 3682
Unmatched cam0 timestamp count: 0
Unmatched cam1 timestamp count: 0
First synchronized timestamp (ns): 1403636579763555584
Last synchronized timestamp (ns): 1403636763813555456
First left-image resolution: 752x480
First right-image resolution: 752x480
First left-image NumPy data type: uint8
First right-image NumPy data type: uint8
```

These results confirm exact timestamp synchronization between the two camera
streams for all 3,682 frames in the sequence.

# Milestone 2: ORB temporal feature matching

Milestone 2 matches visual features between two `cam0` images.

The processing pipeline is:

```text
Reference cam0 image
        +
Current cam0 image
        ↓
ORB feature extraction
        ↓
Hamming-distance KNN matching
        ↓
Lowe's ratio test
        ↓
Fundamental-matrix RANSAC
        ↓
Geometric inlier matches
        ↓
Diagnostic visualization
```

## ORB feature extraction

`extract_orb_features` validates that the input is a nonempty,
two-dimensional, `uint8` grayscale NumPy array.

The source image is not:

- Resized
- Cropped
- Normalized
- Converted silently
- Mutated

The default ORB configuration is:

```text
Maximum retained features: 1000
Scale factor: 1.2
Pyramid levels: 8
Edge threshold: 31
FAST threshold: 20
```

ORB produces 32-byte, or 256-bit, binary descriptors.

## Descriptor matching

`match_orb_descriptors` uses:

```python
cv2.BFMatcher(cv2.NORM_HAMMING)
```

KNN matching is performed with:

```text
k = 2
```

Hamming distance is used because ORB produces binary descriptors.

## Ratio-test filtering

`filter_knn_matches` applies Lowe's ratio test:

```text
d1 < r * d2
```

where:

- `d1` is the nearest descriptor distance
- `d2` is the second-nearest descriptor distance
- `r` is the configurable ratio threshold
- The default ratio threshold is `0.75`

Accepted matches are sorted deterministically using distance and descriptor
indices.

## Fundamental-matrix verification

`verify_matches_ransac` uses matched keypoint coordinates to estimate a
fundamental matrix with RANSAC.

Default parameters:

```text
RANSAC reprojection threshold: 1.0 pixel
RANSAC confidence: 0.99
Minimum correspondences: 8
```

The RANSAC mask partitions ratio-filtered matches into:

- Geometric inliers
- Geometric outliers

If too few correspondences exist or estimation fails, the implementation
returns a structured result with no geometric inliers instead of crashing.

## Match visualization

`render_inlier_matches` creates a side-by-side result containing:

- Reference `cam0` frame
- Current `cam0` frame
- Only geometrically verified RANSAC inliers
- Frame timestamps
- Keypoint counts
- Raw and ratio-filtered match counts
- Inlier and outlier counts
- Geometric inlier ratio

The helper returns a NumPy array and does not open an interactive GUI window.

## Running temporal feature matching

Run matching between frame `k` and frame `k + 1`:

```powershell
python scripts/match_euroc_frames.py `
    --sequence "$HOME\Desktop\datasets\euroc\MH_01_easy" `
    --frame-index 0 `
    --output "outputs\matches_000000.png"
```

Available matching arguments include:

```text
--max-features
--ratio-threshold
--ransac-threshold
--ransac-confidence
```

Example with explicit parameters:

```powershell
python scripts/match_euroc_frames.py `
    --sequence "$HOME\Desktop\datasets\euroc\MH_01_easy" `
    --frame-index 0 `
    --max-features 1000 `
    --ratio-threshold 0.75 `
    --ransac-threshold 1.0 `
    --ransac-confidence 0.99 `
    --output "outputs\matches_000000.png"
```

## Measured Milestone 2 results

The following results were measured using real consecutive `cam0` images from
EuRoC `MH_01_easy`.

| Frame pair | Ref KPs | Cur KPs | Raw KNN | Ratio-filtered | RANSAC inliers | RANSAC outliers | Inlier ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0 → 1` | 1000 | 1000 | 1000 | 704 | 499 | 205 | 0.7088 |
| `100 → 101` | 1000 | 1000 | 1000 | 660 | 442 | 218 | 0.6697 |
| `1000 → 1001` | 1000 | 1000 | 1000 | 662 | 486 | 176 | 0.7341 |

Timing and output information:

| Frame pair | Reference timestamp (ns) | Current timestamp (ns) | Interval (ns) | Output |
|---|---:|---:|---:|---|
| `0 → 1` | 1403636579763555584 | 1403636579813555456 | 49999872 | `outputs\matches_000000.png` |
| `100 → 101` | 1403636584763555584 | 1403636584813555456 | 49999872 | `outputs\matches_000100.png` |
| `1000 → 1001` | 1403636629763555584 | 1403636629813555456 | 49999872 | `outputs\matches_001000.png` |

Across the three evaluated pairs:

- The interval was `49,999,872` ns, approximately 50 ms and consistent with a
  20 Hz stream.
- ORB reached the configured limit of 1,000 retained keypoints in each image.
- Lowe's ratio test retained between 660 and 704 matches.
- Fundamental-matrix RANSAC retained between 442 and 499 geometric inliers.
- The geometric inlier ratio ranged from 66.97% to 73.41%.

Reaching the 1,000-feature limit means ORB retained the configured maximum. It
does not necessarily mean exactly 1,000 candidate corners were initially
detected.

These frame pairs are diagnostic samples, not a complete sequence benchmark.

## Reproducing the visualizations

```powershell
python scripts/match_euroc_frames.py `
    --sequence "$HOME\Desktop\datasets\euroc\MH_01_easy" `
    --frame-index 0 `
    --output "outputs\matches_000000.png"

python scripts/match_euroc_frames.py `
    --sequence "$HOME\Desktop\datasets\euroc\MH_01_easy" `
    --frame-index 100 `
    --output "outputs\matches_000100.png"

python scripts/match_euroc_frames.py `
    --sequence "$HOME\Desktop\datasets\euroc\MH_01_easy" `
    --frame-index 1000 `
    --output "outputs\matches_001000.png"
```

Open a generated visualization in Windows:

```powershell
Start-Process "outputs\matches_000000.png"
```

# Milestone 3: Calibrated relative camera pose

Milestone 3 estimates calibrated two-view relative camera pose between selected
`cam0` frames using Milestone 2 geometric inliers.

## Calibration file

Calibration is loaded in read-only mode from:

```text
<sequence_root>/mav0/cam0/sensor.yaml
```

`load_cam0_calibration` parses and validates:

- `sensor_type`
- Optional sensor comment or label
- `camera_model`
- `distortion_model`
- `resolution`
- `rate_hz`
- `intrinsics`
- `distortion_coefficients`
- `T_BS`

The current implementation supports:

```text
Camera model: pinhole
Distortion model: radial-tangential
```

Calibration values are preserved using double-precision floating-point
representation.

## Camera intrinsic matrix

EuRoC intrinsics are interpreted as:

```text
[fx, fy, cx, cy]
```

The intrinsic matrix is:

```text
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

Distortion coefficients use the order:

```text
[k1, k2, p1, p2]
```

where:

- `k1` and `k2` are radial-distortion coefficients
- `p1` and `p2` are tangential-distortion coefficients

## Parsed EuRoC `cam0` calibration

The real `MH_01_easy` values are:

```text
Camera model: pinhole
Distortion model: radial-tangential
Resolution: 752x480
Rate: 20.0 Hz

fx: 458.654
fy: 457.296
cx: 367.215
cy: 248.375

Distortion coefficients:
[-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]
```

The corresponding camera matrix is:

```text
[[458.654,   0.000, 367.215],
 [  0.000, 457.296, 248.375],
 [  0.000,   0.000,   1.000]]
```

## Point-undistortion convention

`extract_inlier_correspondences` obtains only the Milestone 2
fundamental-matrix RANSAC inliers.

The correspondence arrays use:

```text
Shape: (N, 2)
Coordinate order: (u, v)
Data type: float64
```

`undistort_matched_points` applies `cv2.undistortPoints` using the parsed
intrinsic matrix and distortion coefficients.

No replacement projection matrix is provided, so the result contains
normalized, undistorted coordinates.

Conceptually:

```text
x = (u - cx) / fx
y = (v - cy) / fy
```

after accounting for lens distortion.

Because the points are already normalized, essential-matrix estimation and
pose recovery use the identity camera matrix. The camera intrinsic matrix is
not applied a second time.

## Essential-matrix estimation

The essential matrix is estimated from normalized correspondences using:

```python
cv2.findEssentialMat(
    normalized_reference_points,
    normalized_current_points,
    cameraMatrix=np.eye(3, dtype=np.float64),
    method=cv2.RANSAC,
    prob=ransac_probability,
    threshold=ransac_threshold,
)
```

A conservative minimum of eight correspondences is required by default.

The Milestone 2 fundamental matrix is never reused as the essential matrix.

If OpenCV returns vertically stacked essential-matrix candidates, each `3x3`
candidate is evaluated with pose recovery. The valid candidate with the largest
cheirality-inlier count is selected. Ties are resolved deterministically by
choosing the earliest candidate.

## Relative-pose convention

The recovered pose follows:

```text
X_current =
    R_current_reference @ X_reference
    + t_current_reference
```

where:

- `X_reference` is a 3D point in the reference-camera frame
- `X_current` is the point in the current-camera frame
- `R_current_reference` is the recovered `3x3` rotation matrix
- `t_current_reference` is the unit translation direction

The implementation validates:

- Rotation shape
- Rotation finiteness
- Rotation orthonormality
- Rotation determinant near `+1`
- Translation-vector shape
- Translation-vector finiteness
- Translation norm near `1`

Rotation angle is computed from the clipped matrix trace.

## Translation scale

Monocular two-view geometry cannot recover absolute metric translation scale.

The translation output is therefore:

```text
A unit direction vector
```

It is not:

```text
A translation measured in meters
```

A translation norm of `1.0` only confirms normalization. It does not prove that
the translation direction is physically accurate or well constrained.

Every terminal result and JSON report explicitly states that translation scale
is unknown.

## Cheirality and pose-quality gating

`cv2.recoverPose` performs a positive-depth cheirality check. A supported
triangulated point must lie in front of both cameras.

The implementation reports:

- Calibrated input correspondences
- Essential-matrix inliers
- Cheirality inliers
- Essential-inlier ratio
- Cheirality-inlier ratio

The reported `cheirality_inlier_ratio` is defined as:

```text
cheirality_inlier_count
/
calibrated_input_correspondence_count
```

It is not the ratio of cheirality inliers to essential-matrix inliers.

For example, the `100 → 105` pair reports:

```text
43 / 160 = 0.26875
```

A relative pose is marked valid for downstream motion use only when both
quality thresholds pass:

```text
cheirality_inlier_count >= 30
cheirality_inlier_ratio >= 0.15
```

These are initial project-level engineering safeguards, not universal
pose-validity guarantees.

A mathematically returned pose that fails the thresholds remains available for
diagnostics, but:

```text
pose_quality_valid = False
```

and `pose_quality_reason` explains which threshold failed.

## Why a configurable frame gap is required

At 20 Hz, consecutive frames are separated by approximately 50 ms.

A small gap often provides:

- Strong image overlap
- Many feature correspondences
- Very little translational parallax
- Poorly constrained translation direction

Increasing the gap can improve translation observability by creating a larger
camera baseline.

However, a larger gap can also cause:

- Greater viewpoint change
- Lower image overlap
- Fewer persistent features
- Increased descriptor variation
- More motion blur
- Unfavorable two-view geometry

Correspondence counts generally decrease as viewpoint and appearance changes
become larger, but the relationship is not guaranteed to be monotonic. Local
texture, motion direction, image overlap, and feature-detection variability
also affect the results.

No single fixed frame gap is assumed to work everywhere.

## Running relative-pose estimation

Default consecutive-frame execution:

```powershell
python scripts/estimate_relative_pose.py `
    --sequence "$HOME\Desktop\datasets\euroc\MH_01_easy" `
    --frame-index 0 `
    --output "outputs\pose_000000.json"
```

Execution with a five-frame gap:

```powershell
python scripts/estimate_relative_pose.py `
    --sequence "$HOME\Desktop\datasets\euroc\MH_01_easy" `
    --frame-index 100 `
    --frame-gap 5 `
    --output "outputs\pose_000100_gap005.json"
```

Available options include:

```text
--frame-gap
--min-cheirality-count
--min-cheirality-ratio
--max-features
--ratio-threshold
--ransac-threshold
--ransac-confidence
--essential-probability
--essential-threshold
--min-correspondences
```

Default quality-gate values:

```text
Minimum cheirality count: 30
Minimum cheirality ratio: 0.15
```

## JSON report

Each successful JSON report includes:

### Frame metadata

- Sequence path
- Calibration-file path
- Reference frame index
- Current frame index
- Frame gap
- Reference timestamp
- Current timestamp
- Timestamp difference

### Calibration

- Camera model
- Distortion model
- Resolution
- Camera rate
- `fx`, `fy`, `cx`, and `cy`
- Camera matrix
- Distortion coefficients
- Body-to-sensor transform

### Correspondence diagnostics

- Reference keypoint count
- Current keypoint count
- Ratio-filtered match count
- Milestone 2 geometric-inlier count
- Calibrated input-correspondence count
- Essential-matrix inlier count
- Cheirality-inlier count
- Essential-inlier ratio
- Cheirality-inlier ratio

### Pose-quality information

- Minimum cheirality count
- Minimum cheirality ratio
- Pose-quality validity
- Pose-quality reason

### Pose output

- Essential matrix
- Relative rotation matrix
- Rotation determinant
- Unit translation direction
- Translation norm
- Rotation angle
- Transform convention
- Translation-scale status

All JSON values are finite and standards compliant.

# Measured Milestone 3 experiments

Experiments were run at reference frames `0`, `100`, and `1000` with multiple
frame gaps.

Quality thresholds:

```text
Minimum cheirality count: 30
Minimum cheirality ratio: 0.15
```

| Reference | Gap | Current | Ratio-filtered | Calibrated input | Essential inliers | Cheirality | Cheirality ratio | Rotation | Quality |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1 | 1 | 704 | 499 | 337 | 24 | 0.0481 | 0.161839° | Rejected |
| 0 | 2 | 2 | 573 | 381 | 265 | 22 | 0.0577 | 0.773289° | Rejected |
| 0 | 5 | 5 | 345 | 241 | 177 | 27 | 0.1120 | 2.545557° | Rejected |
| 0 | 10 | 10 | 174 | 131 | 83 | 83 | 0.6336 | 2.990605° | Accepted |
| 100 | 1 | 101 | 660 | 442 | 291 | 2 | 0.0045 | 1.071188° | Rejected |
| 100 | 2 | 102 | 475 | 322 | 217 | 1 | 0.0031 | 1.772359° | Rejected |
| 100 | 5 | 105 | 264 | 160 | 106 | 43 | 0.2687 | 2.685171° | Accepted |
| 100 | 10 | 110 | 488 | 350 | 239 | 0 | 0.0000 | Not recovered | Failed |
| 1000 | 1 | 1001 | 662 | 486 | 330 | 7 | 0.0144 | 1.600239° | Rejected |
| 1000 | 2 | 1002 | 598 | 411 | 266 | 22 | 0.0535 | 2.526919° | Rejected |
| 1000 | 5 | 1005 | 345 | 212 | 134 | 134 | 0.6321 | 5.381084° | Accepted |
| 1000 | 10 | 1010 | 202 | 124 | 86 | 85 | 0.6855 | 8.563743° | Accepted |

All successful pose recoveries produced:

```text
Rotation determinant: approximately 1.0
Translation norm: approximately 1.0
Translation scale: unknown
```

## Findings

### Small gaps

All tested gap-1 and gap-2 pairs failed the quality gate.

Although these pairs retained many feature correspondences and essential-matrix
inliers, only 1 to 24 correspondences passed cheirality.

This indicates that strong feature overlap does not guarantee sufficiently
observable translation.

### Moderate gaps

A gap of five frames produced accepted poses at references 100 and 1000:

```text
100 → 105:
43 cheirality inliers
0.2687 cheirality-to-input ratio

1000 → 1005:
134 cheirality inliers
0.6321 cheirality-to-input ratio
```

The `0 → 5` pair remained below the thresholds, showing that gap 5 is not
universally sufficient.

### Larger gaps

Gap 10 produced accepted poses for:

```text
0 → 10
1000 → 1010
```

The `100 → 110` pair was correctly rejected. This pair retained 350 calibrated
correspondences and 239 essential-matrix inliers, but no correspondence passed
the positive-depth cheirality check.

The script therefore returned a nonzero exit status rather than emitting an
unsupported relative pose.

Possible causes include:

- Insufficient or unfavorable translational parallax
- Dominant rotational motion
- Scene geometry
- Correspondence ambiguity
- An essential-matrix decomposition inconsistent with positive depth

Ground-truth motion analysis is required before attributing the failure to a
specific vehicle-motion pattern.

## Conclusion from the frame-gap experiment

There is no single best fixed frame gap for every part of the sequence.

The results demonstrate a tradeoff:

```text
Gap too small
    ↓
Many matches but weak parallax
    ↓
Translation poorly constrained

Moderate gap
    ↓
Enough overlap and enough parallax
    ↓
Supported relative pose

Gap too large or locally unfavorable
    ↓
Reduced overlap or degenerate geometry
    ↓
Pose recovery may fail
```

Future multi-frame processing should use pose-quality or parallax-driven frame
selection instead of assuming a globally fixed frame gap.

These selected pairs are diagnostic examples, not a complete sequence
benchmark.

# Testing and code quality

Run all automated tests:

```powershell
python -m pytest -q
```

Current validated result:

```text
106 passed
```

Run Ruff:

```powershell
python -m ruff check .
```

Current validated result:

```text
All checks passed!
```

Automated tests use:

- Synthetic images
- Synthetic EuRoC-style directories
- Synthetic calibration files
- Deterministic synthetic 3D point clouds
- Known camera rotations and translations
- Controlled noise and outliers

The tests do not:

- Depend on the real EuRoC dataset
- Modify the real dataset
- Write generated files into the dataset
- Depend on exact OpenCV RANSAC inlier counts
- Assume exact ORB keypoint counts

Tests validate:

- Calibration parsing and precision
- Intrinsic-matrix construction
- Distortion-array construction
- Calibration-to-image resolution matching
- Correspondence extraction
- Point undistortion
- Essential-matrix estimation
- Candidate splitting and selection
- Relative-pose recovery
- Rotation orthonormality
- Rotation determinant
- Translation normalization
- Cheirality processing
- Pose-quality evaluation
- JSON serialization
- Structured failure handling
- Input immutability

# Assumptions and limitations

## Calibration assumptions

The current loader supports:

```text
camera_model: pinhole
distortion_model: radial-tangential
```

Expected intrinsics:

```text
[fx, fy, cx, cy]
```

Expected distortion coefficients:

```text
[k1, k2, p1, p2]
```

Images must be:

- Nonempty
- Two-dimensional
- Grayscale
- `uint8`
- Resolution-compatible with the calibration

## Structured failures

Relative-pose estimation returns a structured failure instead of fabricating
matrices when:

- Too few calibrated correspondences exist
- Coordinate arrays are malformed
- Coordinates contain nonfinite values
- Correspondences have insufficient spatial spread
- Essential-matrix estimation fails
- Essential matrices are malformed
- Returned masks are inconsistent
- All pose candidates fail validation
- No points pass cheirality

## Geometric degeneracies

Two-view monocular geometry can become unreliable under:

- Pure or nearly pure rotation
- Very small translation
- Distant scene structure
- Insufficient parallax
- Planar or nearly planar scenes
- Collinear feature configurations
- Low texture
- Repetitive patterns
- Motion blur
- Dynamic objects
- Calibration inaccuracies
- Remaining correspondence outliers

## Current scope limitations

Milestone 3 does not implement:

- Metric translation-scale recovery
- Multi-frame trajectory accumulation
- Visual odometry
- Stereo triangulation
- Stereo depth
- Keyframe selection
- Bundle adjustment
- Loop closure
- IMU preintegration
- Visual-inertial fusion
- SLAM
- ROS 2
- PX4
- Gazebo
- Occupancy mapping
- Path planning
- Drone control

# Planned Milestone 4

Milestone 4 will extend the two-view pipeline toward quality-gated visual
odometry and trajectory evaluation.

Planned work:

1. Parse and interpolate EuRoC ground-truth poses.
2. Evaluate estimated relative rotation against ground truth.
3. Evaluate translation direction against ground truth.
4. Introduce quality-gated frame or keyframe selection.
5. Accumulate accepted relative rotations.
6. Address translation-scale ambiguity explicitly.
7. Compare trajectory segments with ground truth.
8. Measure relative pose error and trajectory drift.
9. Reject unsupported pose updates from accumulation.
10. Document failure cases across longer sequence segments.

Metric scale, stereo odometry, and IMU fusion must be designed carefully before
claiming a complete metric trajectory.

# Long-term roadmap

```text
Milestone 1
EuRoC stereo dataset loading
        ↓
Milestone 2
ORB temporal feature matching
        ↓
Milestone 3
Calibrated relative pose estimation
        ↓
Milestone 4
Trajectory accumulation and ground-truth evaluation
        ↓
Milestone 5
Stereo-inertial SLAM integration
        ↓
Milestone 6
Failure-injection testing
        ↓
Milestone 7
ROS 2 integration
        ↓
Milestone 8
PX4 and Gazebo simulation
        ↓
Milestone 9
Occupancy mapping and A* planning
        ↓
Milestone 10
Closed-loop autonomous drone navigation
```

## License

A project license will be added in a separate repository update.