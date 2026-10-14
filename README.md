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
- Avoids loading the full dataset into memory
- Validates malformed, missing, duplicate, and unreadable data
- Includes automated tests and a real-dataset inspection script

The main interfaces are:

- `EurocSequence`
- `EurocSequence.load_stereo_pair`
- `EurocSequence.load_left_image`

### Completed Milestone 2: ORB temporal feature matching

Milestone 2 provides an ORB-based temporal feature-matching pipeline for
consecutive EuRoC `cam0` images.

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

### Current Milestone 3: Relative camera pose estimation

Milestone 3 will:

- Load EuRoC camera calibration from `sensor.yaml`
- Validate camera intrinsics and distortion parameters
- Undistort matched feature coordinates
- Estimate the essential matrix
- Recover relative camera rotation
- Recover relative translation direction
- Evaluate relative pose on selected frame pairs

Milestone 3 will not initially accumulate a complete trajectory or estimate
metric scale.

## Repository structure

```text
AeroNav-VIO/
├── .github/
│   └── copilot-instructions.md
├── configs/
├── docs/
├── outputs/
├── scripts/
│   ├── inspect_euroc.py
│   └── match_euroc_frames.py
├── src/
│   └── aeronav_vio/
│       ├── data/
│       │   └── euroc.py
│       ├── evaluation/
│       └── features/
│           └── orb.py
├── tests/
├── .gitignore
├── pyproject.toml
└── README.md
```

Generated visualizations are written under `outputs/` and excluded from version
control. Use the documented commands to reproduce them locally.

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

From Windows PowerShell, clone the repository and enter the project directory:

```powershell
git clone https://github.com/YOUR_USERNAME/AeroNav-VIO.git
cd AeroNav-VIO
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents environment activation, apply a temporary policy to the
current PowerShell process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install the package and development dependencies:

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

Commands should receive the sequence root containing `mav0`.

Correct:

```text
C:\Users\<username>\Desktop\datasets\euroc\MH_01_easy
```

Incorrect:

```text
C:\Users\<username>\Desktop\datasets\euroc\MH_01_easy\mav0
```

The scripts treat the dataset as read-only. Generated files are saved only
inside the repository's `outputs/` directory.

## Milestone 1: Inspecting a EuRoC sequence

Run the sequence-inspection script:

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

### Measured `MH_01_easy` inspection result

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

## Milestone 2: ORB temporal feature matching

Milestone 2 matches visual features between consecutive images from the left
camera, `cam0`.

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

### ORB feature extraction

`extract_orb_features` validates that the input is a nonempty, two-dimensional,
`uint8` grayscale NumPy array.

The source image is not:

- Resized
- Cropped
- Normalized
- Converted silently
- Mutated

The current default ORB configuration is:

```text
Maximum retained features: 1000
Scale factor: 1.2
Pyramid levels: 8
Edge threshold: 31
FAST threshold: 20
```

ORB produces 32-byte, or 256-bit, binary descriptors.

### Descriptor matching

`match_orb_descriptors` uses:

```python
cv2.BFMatcher(cv2.NORM_HAMMING)
```

KNN matching is performed with:

```text
k = 2
```

Hamming distance is appropriate because ORB produces binary descriptors.

### Ratio-test filtering

`filter_knn_matches` applies Lowe's ratio test:

```text
d1 < r * d2
```

where:

- `d1` is the distance to the nearest descriptor
- `d2` is the distance to the second-nearest descriptor
- `r` is the ratio-test threshold
- The default ratio threshold is `0.75`

Accepted matches are sorted deterministically using distance and descriptor
indices.

### Geometric verification

`verify_matches_ransac` uses matched keypoint coordinates to estimate a
fundamental matrix with RANSAC.

Default parameters:

```text
RANSAC reprojection threshold: 1.0 pixel
RANSAC confidence: 0.99
Minimum correspondences: 8
```

The RANSAC mask divides ratio-filtered matches into:

- Geometric inliers
- Geometric outliers

If fewer than eight correspondences exist, or fundamental-matrix estimation
fails, the implementation returns a valid result with no geometric inliers
rather than crashing.

### Visualization

`render_inlier_matches` creates a side-by-side image containing:

- Reference `cam0` frame on the left
- Current `cam0` frame on the right
- Only geometrically verified RANSAC inliers
- Reference and current timestamps
- Keypoint counts
- Raw and ratio-filtered match counts
- Inlier and outlier counts
- Geometric inlier ratio

The visualization helper returns a NumPy array and does not open an interactive
window.

## Running temporal feature matching

Run matching between frame `k` and frame `k + 1`:

```powershell
python scripts/match_euroc_frames.py `
    --sequence "$HOME\Desktop\datasets\euroc\MH_01_easy" `
    --frame-index 0 `
    --output "outputs\matches_000000.png"
```

Additional configurable arguments include:

```text
--max-features
--ratio-threshold
--ransac-threshold
--ransac-confidence
```

Example using explicit matching parameters:

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

The resulting visualization combines two `752x480` EuRoC `cam0` images with a
statistics header and geometrically verified inlier matches.

## Printed matching statistics

The temporal-matching script prints:

- Normalized sequence path
- Reference frame index
- Current frame index
- Reference timestamp in nanoseconds
- Current timestamp in nanoseconds
- Timestamp difference in nanoseconds
- Reference keypoint count
- Current keypoint count
- Raw KNN match count
- Ratio-filtered match count
- Geometric inlier count
- Geometric outlier count
- Geometric inlier ratio
- Output image path
- Output image resolution

The geometric inlier ratio is calculated as:

```text
geometric inlier count / ratio-filtered match count
```

If no ratio-filtered matches exist, the ratio is reported safely as zero.

## Measured Milestone 2 results

The following results were measured using real consecutive `cam0` images from
EuRoC `MH_01_easy`.

Default parameters:

```text
Maximum features: 1000
Ratio threshold: 0.75
RANSAC threshold: 1.0 pixel
RANSAC confidence: 0.99
```

| Frame pair | Ref KPs | Cur KPs | Raw KNN | Ratio-filtered | RANSAC inliers | RANSAC outliers | Inlier ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0 → 1` | 1000 | 1000 | 1000 | 704 | 499 | 205 | 0.7088 |
| `100 → 101` | 1000 | 1000 | 1000 | 660 | 442 | 218 | 0.6697 |
| `1000 → 1001` | 1000 | 1000 | 1000 | 662 | 486 | 176 | 0.7341 |

Detailed timing and output information:

| Frame pair | Reference timestamp (ns) | Current timestamp (ns) | Interval (ns) | Output |
|---|---:|---:|---:|---|
| `0 → 1` | 1403636579763555584 | 1403636579813555456 | 49999872 | `outputs\matches_000000.png` |
| `100 → 101` | 1403636584763555584 | 1403636584813555456 | 49999872 | `outputs\matches_000100.png` |
| `1000 → 1001` | 1403636629763555584 | 1403636629813555456 | 49999872 | `outputs\matches_001000.png` |

Across the three evaluated frame pairs:

- The observed interval was `49,999,872` ns, approximately 50 ms and
  consistent with a 20 Hz camera stream.
- ORB reached the configured limit of 1,000 retained keypoints in each
  evaluated image.
- Lowe's ratio test retained between 660 and 704 matches.
- Fundamental-matrix RANSAC verified between 442 and 499 geometric inliers.
- The geometric inlier ratio ranged from 66.97% to 73.41%.

Reaching the 1,000-feature limit does not necessarily mean exactly 1,000
candidate corners were initially detected. It means ORB retained up to the
configured maximum after internal feature selection.

These three frame pairs are diagnostic samples, not a complete sequence
benchmark. A broader evaluation should report matching and inlier
distributions across the complete sequence, including low-texture, blurred,
repetitive, and high-motion segments.

## Reproducing all measured visualizations

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

Generated visualizations are intentionally excluded from version control.
Running the commands above reproduces them locally.

## Testing and code quality

Run the complete automated test suite:

```powershell
python -m pytest -q
```

Current validated result:

```text
50 passed
```

Run Ruff:

```powershell
python -m ruff check .
```

Current validated result:

```text
All checks passed!
```

The automated tests use synthetic images and synthetic EuRoC-style directories
created under pytest's temporary directory.

Automated tests do not:

- Depend on the real EuRoC sequence
- Modify the real dataset
- Write generated files into the dataset
- Assume exact ORB keypoint counts when OpenCV-version-dependent behavior could
  vary

Tests use invariant-based assertions for descriptor dimensions, match-index
bounds, inlier counts, visualization dimensions, file creation, input
immutability, and error handling.

## Assumptions and limitations

### Image input

ORB extraction currently requires:

- A nonempty NumPy array
- Two-dimensional grayscale input
- `uint8` pixel values

Color images, empty arrays, and unsupported data types are rejected with
informative errors.

### Temporal matching only

Milestone 2 matches consecutive images from `cam0`.

The implementation does not yet use:

- `cam1` for stereo feature matching
- Stereo triangulation
- Stereo depth
- IMU measurements
- Ground-truth pose

### Lens distortion

Fundamental-matrix RANSAC currently operates on feature coordinates from the
original camera images.

Milestone 2 does not yet:

- Load `sensor.yaml`
- Apply camera intrinsics
- Undistort keypoint coordinates
- Compensate for radial-tangential lens distortion

These capabilities belong to Milestone 3.

### Geometric degeneracy

Feature matching and fundamental-matrix estimation may degrade under:

- Pure or nearly pure camera rotation
- Very small motion between frames
- Motion blur
- Low-texture surfaces
- Repetitive visual patterns
- Strong lighting variation
- Dynamic objects
- Residual lens distortion

A high RANSAC inlier count indicates epipolar consistency. It does not by
itself prove that the estimated geometry is physically correct or sufficient
for stable pose recovery.

### No pose estimation yet

Milestone 2 does not implement:

- Essential-matrix estimation
- Relative camera rotation
- Relative translation direction
- Translation magnitude
- Metric scale
- Trajectory accumulation
- Visual odometry
- SLAM
- Bundle adjustment
- Loop closure

## Planned Milestone 3

Milestone 3 will build relative camera pose estimation on top of the verified
Milestone 2 correspondences.

Planned work:

1. Parse the EuRoC `cam0/sensor.yaml` calibration file.
2. Validate the camera model, resolution, distortion coefficients, and
   projection parameters.
3. Build the camera intrinsic matrix.
4. Undistort matched feature coordinates.
5. Estimate the essential matrix using calibrated correspondences.
6. Recover relative camera rotation and translation direction.
7. Check cheirality and valid triangulation conditions.
8. Compare selected relative-pose estimates with EuRoC ground truth where
   coordinate-frame alignment permits meaningful evaluation.
9. Add synthetic and real-data tests.
10. Document failure cases and limitations.

A complete trajectory, metric scale, visual-inertial fusion, and SLAM remain
outside the initial Milestone 3 scope.

## Long-term roadmap

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
Trajectory accumulation and evaluation
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

The original AeroNav-VIO source code in this repository is licensed under the
Apache License 2.0. See [LICENSE](LICENSE) for the complete terms.

Third-party libraries, datasets, and future SLAM components remain subject to
their own licenses. The EuRoC dataset is not distributed in this repository.

Future ORB-SLAM3 integration will remain externally managed and will be
reviewed separately because ORB-SLAM3 is distributed under GPLv3.