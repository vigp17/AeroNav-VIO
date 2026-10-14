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

The project currently focuses on building and validating the perception,
calibration, relative-pose, and evaluation foundations before introducing
multi-frame odometry, simulation, planning, and closed-loop flight control.

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

Primary interfaces:

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

Milestone 3 provides calibrated two-view relative-pose estimation.

The implementation:

- Loads and validates EuRoC `cam0/sensor.yaml`
- Constructs the pinhole camera intrinsic matrix
- Parses radial-tangential distortion coefficients
- Validates image resolution against calibration
- Extracts Milestone 2 geometric-inlier correspondences
- Undistorts pixel coordinates into normalized coordinates
- Estimates essential-matrix candidates using RANSAC
- Handles stacked essential-matrix candidates deterministically
- Recovers relative camera rotation
- Recovers unit translation direction with unknown metric scale
- Applies positive-depth cheirality filtering
- Supports configurable frame gaps
- Applies cheirality-based pose-quality gating
- Writes structured JSON diagnostic reports
- Includes synthetic calibration, geometry, and script-level tests

### Completed Milestone 4A: Ground-truth relative-pose evaluation

Milestone 4A evaluates Milestone 3 estimates against EuRoC ground truth.

The implementation:

- Parses EuRoC ground-truth positions and `wxyz` quaternions
- Preserves integer nanosecond timestamps
- Performs exact timestamp lookup when possible
- Linearly interpolates position
- Interpolates orientation using shortest-path quaternion SLERP
- Rejects timestamps outside ground-truth coverage
- Converts body poses into `cam0` camera poses
- Computes ground-truth relative camera motion
- Measures relative rotation error
- Measures translation-direction angular error
- Reports metric ground-truth displacement
- Preserves the unknown-scale status of monocular estimated translation
- Evaluates accepted and rejected Milestone 3 poses separately
- Produces selected-pair JSON reports
- Produces per-pair benchmark CSV output
- Produces aggregate benchmark JSON output
- Includes synthetic ground-truth, transformation, interpolation, evaluation,
  and script-level tests

### Current focus: Milestone 4B preparation

Milestone 4B will investigate candidate-frame selection and quality-gated
multi-frame pose accumulation.

Milestone 4B has not yet been implemented.

A full metric trajectory, stereo odometry, IMU fusion, SLAM, ROS 2, PX4,
Gazebo, mapping, and planning remain future work.

## Repository structure

```text
AeroNav-VIO/
├── .github/
│   └── copilot-instructions.md
├── configs/
├── docs/
├── outputs/
├── scripts/
│   ├── benchmark_relative_pose.py
│   ├── estimate_relative_pose.py
│   ├── evaluate_relative_pose.py
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
│       │   ├── __init__.py
│       │   └── relative_pose.py
│       ├── features/
│       │   └── orb.py
│       ├── geometry/
│       │   ├── __init__.py
│       │   └── relative_pose.py
│       └── ground_truth/
│           ├── __init__.py
│           └── euroc.py
├── tests/
│   ├── test_estimate_relative_pose.py
│   ├── test_euroc_calibration.py
│   ├── test_euroc_ground_truth.py
│   ├── test_euroc_loader.py
│   ├── test_evaluate_relative_pose.py
│   ├── test_match_euroc_frames.py
│   ├── test_orb_features.py
│   ├── test_relative_pose.py
│   └── test_relative_pose_evaluation.py
├── .gitignore
├── pyproject.toml
└── README.md
```

Generated visualizations, pose reports, evaluation reports, and benchmark
outputs are written under `outputs/` and excluded from version control.

Use the documented commands to reproduce the results locally.

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

Clone the repository:

```powershell
git clone https://github.com/vigp17/AeroNav-VIO.git
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

If PowerShell prevents activation, apply a temporary policy to the current
PowerShell process:

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
                │   ├── data.csv
                │   └── sensor.yaml
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

## Running the sequence inspector

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

## Measured `MH_01_easy` result

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

The feature-matching pipeline is:

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

## ORB configuration

Default parameters:

```text
Maximum retained features: 1000
Scale factor: 1.2
Pyramid levels: 8
Edge threshold: 31
FAST threshold: 20
Ratio threshold: 0.75
Fundamental-matrix RANSAC threshold: 1.0 pixel
Fundamental-matrix RANSAC confidence: 0.99
Minimum correspondences: 8
```

ORB produces 32-byte, or 256-bit, binary descriptors.

Descriptor matching uses:

```python
cv2.BFMatcher(cv2.NORM_HAMMING)
```

with KNN matching:

```text
k = 2
```

## Running temporal feature matching

```powershell
python scripts/match_euroc_frames.py `
    --sequence "$HOME\Desktop\datasets\euroc\MH_01_easy" `
    --frame-index 0 `
    --output "outputs\matches_000000.png"
```

Available arguments include:

```text
--max-features
--ratio-threshold
--ransac-threshold
--ransac-confidence
```

## Measured feature-matching results

| Frame pair | Ref KPs | Cur KPs | Raw KNN | Ratio-filtered | RANSAC inliers | RANSAC outliers | Inlier ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0 → 1` | 1000 | 1000 | 1000 | 704 | 499 | 205 | 0.7088 |
| `100 → 101` | 1000 | 1000 | 1000 | 660 | 442 | 218 | 0.6697 |
| `1000 → 1001` | 1000 | 1000 | 1000 | 662 | 486 | 176 | 0.7341 |

These selected pairs are diagnostic examples, not a full-sequence feature
benchmark.

# Milestone 3: Calibrated relative camera pose

Milestone 3 uses the geometrically verified Milestone 2 correspondences to
estimate calibrated two-view relative camera motion.

The pipeline is:

```text
Geometrically verified correspondences
        ↓
Load cam0 calibration
        ↓
Undistort matched coordinates
        ↓
Estimate essential matrix
        ↓
Recover relative rotation
        ↓
Recover unit translation direction
        ↓
Apply cheirality check
        ↓
Evaluate pose quality
```

## Camera calibration

Calibration is loaded from:

```text
<sequence_root>/mav0/cam0/sensor.yaml
```

EuRoC intrinsics are interpreted as:

```text
[fx, fy, cx, cy]
```

The camera matrix is:

```text
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

Radial-tangential distortion coefficients are ordered as:

```text
[k1, k2, p1, p2]
```

## Parsed `MH_01_easy` cam0 calibration

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

## Normalized-coordinate convention

Matched pixel points use:

```text
(u, v)
```

`cv2.undistortPoints` removes radial-tangential distortion and produces
normalized coordinates.

Conceptually:

```text
x = (u - cx) / fx
y = (v - cy) / fy
```

after distortion compensation.

The essential matrix is estimated from normalized coordinates using the
identity camera matrix.

## Relative-pose convention

The estimated relative pose follows:

```text
X_current =
    R_current_reference @ X_reference
    + t_current_reference
```

where:

- `R_current_reference` is a `3x3` rotation matrix
- `t_current_reference` is a three-element unit translation direction

The estimated translation magnitude is unknown.

The translation vector is not measured in meters.

## Cheirality quality gate

The pose-quality gate requires:

```text
cheirality_inlier_count >= 30
cheirality_inlier_ratio >= 0.15
```

The cheirality ratio is defined as:

```text
cheirality_inlier_count
/
calibrated_input_correspondence_count
```

These thresholds are initial project-level safeguards, not universal
pose-validity guarantees.

A pose may be numerically recoverable but marked:

```text
pose_quality_valid = False
```

when geometric support is insufficient.

## Running relative-pose estimation

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

## Frame-gap experiment

Selected real-data results showed:

- Small frame gaps often retained many matches but had insufficient parallax
- Moderate gaps sometimes improved translation observability
- Large gaps sometimes worked well but could also produce complete pose failure
- No single fixed frame gap succeeded at every tested sequence location

This motivated ground-truth evaluation in Milestone 4A.

# Milestone 4A: Ground-truth relative-pose evaluation

Milestone 4A measures whether Milestone 3 estimates agree with the true EuRoC
camera motion.

The evaluation pipeline is:

```text
EuRoC body ground truth
        +
cam0 body-to-sensor extrinsic
        ↓
Ground-truth cam0 poses
        ↓
Ground-truth relative cam0 motion
        +
Milestone 3 estimated motion
        ↓
Rotation angular error
Translation-direction angular error
Quality-gate evaluation
```

## Ground-truth CSV fields

Ground truth is loaded from:

```text
mav0/state_groundtruth_estimate0/data.csv
```

Required fields:

```text
timestamp
p_RS_R_x
p_RS_R_y
p_RS_R_z
q_RS_w
q_RS_x
q_RS_y
q_RS_z
```

Position is represented as:

```text
p_world_body =
    [p_RS_R_x, p_RS_R_y, p_RS_R_z]
```

Quaternion values use scalar-first order:

```text
q_world_body_wxyz =
    [q_RS_w, q_RS_x, q_RS_y, q_RS_z]
```

## Ground-truth coverage limitation

The first ground-truth timestamp is:

```text
1403636580838555648 ns
```

Camera frame `0` occurs at:

```text
1403636579763555584 ns
```

Camera frame `0` therefore precedes available ground truth by approximately:

```text
1.075 seconds
```

The evaluator does not extrapolate.

An attempt to evaluate frame `0` returns an informative error:

```text
Requested timestamp precedes first available ground-truth timestamp
and ground-truth extrapolation is not permitted
```

## Ground-truth interpolation

For a requested camera timestamp `t`, the evaluator finds:

```text
t0 <= t <= t1
```

Position is interpolated linearly:

```text
alpha = (t - t0) / (t1 - t0)

p(t) = (1 - alpha) * p0 + alpha * p1
```

Orientation is interpolated using shortest-path quaternion SLERP.

The implementation:

- Preserves `wxyz` ordering
- Normalizes input quaternions
- Handles `q` and `-q` equivalence
- Uses the shortest interpolation path
- Uses normalized linear interpolation when rotations are extremely close
- Normalizes the final quaternion
- Rejects zero-norm and nonfinite quaternions
- Never extrapolates beyond ground-truth coverage

## Ground-truth camera-pose conversion

EuRoC ground truth describes body motion, while Milestone 3 estimates `cam0`
camera motion.

The evaluator combines:

```text
Interpolated body pose
        +
cam0 T_BS extrinsic
        ↓
World cam0 pose
```

The resulting ground-truth relative camera pose uses the same convention as
Milestone 3:

```text
X_current =
    R_current_reference @ X_reference
    + t_current_reference
```

Synthetic transform tests verify:

- Transform inversion
- Transform composition order
- Nonidentity camera extrinsics
- Translation and rotation composition
- Reversed-composition detection
- Rotation orthonormality
- Rotation determinant

## Rotation error

The rotation-error matrix is computed consistently with the shared relative
transform convention.

Angular rotation error is calculated from the clipped matrix trace:

```text
rotation_error =
    degrees(
        arccos(
            clip(
                (trace(R_error) - 1) / 2,
                -1,
                1
            )
        )
    )
```

The result lies between `0°` and `180°`.

## Translation-direction error

The metric ground-truth translation vector is normalized when its magnitude is
greater than the configured near-zero threshold.

Translation-direction angular error is calculated between:

```text
Estimated unit translation direction
```

and:

```text
Ground-truth unit translation direction
```

Opposite directions are not treated as equivalent.

If the ground-truth displacement is too small, translation direction is marked
undefined instead of fabricating an angular error.

The ground-truth translation magnitude is reported in meters.

The monocular estimated translation magnitude remains unknown.

## Running selected-pair evaluation

Evaluate a quality-gate rejected pair:

```powershell
python scripts/evaluate_relative_pose.py `
    --sequence "$HOME\Desktop\datasets\euroc\MH_01_easy" `
    --frame-index 100 `
    --frame-gap 1 `
    --output "outputs\evaluation_000100_gap001.json"
```

Evaluate a quality-gate accepted pair:

```powershell
python scripts/evaluate_relative_pose.py `
    --sequence "$HOME\Desktop\datasets\euroc\MH_01_easy" `
    --frame-index 100 `
    --frame-gap 5 `
    --output "outputs\evaluation_000100_gap005.json"
```

## Selected-pair results

| Pair | Quality gate | Rotation error | Translation-direction error | Ground-truth displacement |
|---|---|---:|---:|---:|
| `100 → 101` | Rejected | 0.270753° | 20.969438° | 0.029378 m |
| `100 → 105` | Accepted | 0.217376° | 4.951136° | 0.091140 m |
| `1000 → 1001` | Rejected | 0.229284° | 12.951683° | 0.027340 m |
| `1000 → 1005` | Accepted | 0.821720° | 10.951009° | 0.142743 m |
| `1000 → 1010` | Accepted | 0.398195° | 9.633668° | 0.284979 m |

These examples suggested that the quality gate was particularly useful for
screening translation-direction estimates. The broader benchmark below was
used to evaluate that hypothesis.

# Milestone 4A benchmark

## Benchmark command

```powershell
python scripts/benchmark_relative_pose.py `
    --sequence "$HOME\Desktop\datasets\euroc\MH_01_easy" `
    --start-frame 100 `
    --end-frame 1100 `
    --stride 50 `
    --frame-gaps 1 2 5 10 `
    --csv-output "outputs\benchmark_relative_pose.csv" `
    --json-output "outputs\benchmark_relative_pose_summary.json"
```

The benchmark requested:

```text
21 reference frames
×
4 frame gaps
=
84 frame pairs
```

## Overall benchmark results

```text
Total requested pairs: 84
Evaluated pairs: 48
Estimation failures: 36
Ground-truth unavailable: 0
Quality-gate accepted: 22
Quality-gate rejected: 26
Translation direction defined: 48
```

Evaluation success rate:

```text
48 / 84 = 57.1%
```

Estimation failure rate:

```text
36 / 84 = 42.9%
```

## Accepted versus rejected errors

| Group | Pair count | Median rotation error | Mean rotation error | P90 rotation error | Median translation error | Mean translation error | P90 translation error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Quality accepted | 22 | 0.1498° | 0.3307° | 0.9521° | 3.6853° | 10.7049° | 9.8558° |
| Quality rejected | 26 | 0.2398° | 1.1180° | 3.4168° | 22.1269° | 41.4549° | 106.3688° |

Across this diagnostic benchmark, accepted poses had:

- Lower median rotation error
- Lower mean rotation error
- Lower 90th-percentile rotation error
- Lower median translation-direction error
- Lower mean translation-direction error
- Lower 90th-percentile translation-direction error

The accepted median translation-direction error was approximately six times
lower than the rejected median.

## Results by frame gap

| Gap | Evaluated | Failures | Accepted | Rejected | Accepted median rotation error | Rejected median rotation error | Accepted median translation error | Rejected median translation error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13 | 8 | 4 | 9 | 0.1082° | 0.1717° | 4.7244° | 23.2844° |
| 2 | 12 | 9 | 6 | 6 | 0.1498° | 0.2855° | 3.1377° | 21.3946° |
| 5 | 12 | 9 | 7 | 5 | 0.2174° | 0.6637° | 3.7695° | 15.1823° |
| 10 | 11 | 10 | 5 | 6 | 0.2956° | 2.9969° | 4.8136° | 23.2642° |

## Benchmark findings

### The quality gate is useful

For every frame gap, quality-gate accepted poses had lower median rotation error
and lower median translation-direction error than rejected poses.

The measured results provide evidence that cheirality support is a useful
indicator of relative-pose reliability.

### The quality gate is not sufficient by itself

The accepted group still contained large-error outliers.

For gap 10:

```text
Accepted median translation-direction error: 4.8136°
Accepted mean translation-direction error: 32.8592°
Accepted P90 translation-direction error: 91.2507°
```

The large mean and 90th-percentile error indicate that at least one accepted
gap-10 pose had a very large translation-direction error.

Cheirality support alone does not eliminate every inaccurate pose.

Future pose selection should consider additional indicators such as:

- Parallax magnitude
- Reprojection error
- Spatial feature distribution
- Essential-matrix residuals
- Candidate-gap consistency
- Rotation consistency
- Temporal motion consistency

### There is no universally optimal frame gap

Gap 1 produced the lowest accepted median rotation error.

Gap 2 produced the lowest accepted median translation-direction error.

Gap 5 accepted the greatest number of poses.

Gap 10 sometimes provided a useful baseline but also contained accepted
translation outliers and the largest number of estimation failures.

No single frame gap was optimal for every sampled sequence location.

### Estimation failures remain significant

The pipeline failed to produce an evaluable pose for:

```text
36 of 84 requested pairs
```

A future trajectory system must not assume that every requested frame pair
produces a usable motion estimate.

The future system should support:

```text
Candidate gap selection
        ↓
Pose estimation
        ↓
Quality validation
        ↓
Accept supported pose
        or
Try another candidate
        or
Skip the update safely
```

## Benchmark scope

The benchmark is diagnostic rather than exhaustive.

It evaluated:

```text
Reference frames: 100 through 1100
Sampling stride: 50
Frame gaps: 1, 2, 5, and 10
```

The benchmark does not cover every frame in `MH_01_easy` and does not establish
performance on other EuRoC sequences or real hardware.

# Testing and code quality

Run the full test suite:

```powershell
python -m pytest -q
```

Current validated result:

```text
131 passed
```

Run Ruff:

```powershell
python -m ruff check .
```

Current validated result:

```text
All checks passed!
```

The tests use:

- Synthetic EuRoC-style directories
- Synthetic images
- Synthetic camera calibration
- Synthetic ground-truth CSV files
- Deterministic 3D point clouds
- Known body and camera transforms
- Known camera rotations and translations
- Controlled noise and outliers

The tests do not:

- Depend on the real EuRoC sequence
- Modify the real dataset
- Write generated files into the dataset
- Depend on exact ORB keypoint counts
- Depend on exact OpenCV RANSAC inlier counts

Tests validate:

- Dataset loading
- Timestamp precision
- Camera synchronization
- ORB feature extraction
- Descriptor matching
- Fundamental-matrix verification
- Calibration parsing
- Point undistortion
- Essential-matrix estimation
- Relative-pose recovery
- Cheirality processing
- Pose-quality gating
- Quaternion normalization
- Quaternion sign equivalence
- Quaternion SLERP
- Exact timestamp lookup
- Ground-truth interpolation
- Body-to-camera transform composition
- Reversed transform-order detection
- Rotation-error calculation
- Translation-direction-error calculation
- Ground-truth coverage rejection
- Near-zero translation handling
- JSON serialization
- CSV generation
- Aggregate benchmark statistics
- Structured failure handling
- Input immutability

# Assumptions and limitations

## Camera model

The current calibration loader supports:

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

## Monocular scale

Milestone 3 estimates only translation direction.

It does not independently recover metric translation magnitude.

Ground-truth magnitude is used for evaluation only.

## Two-view geometry

Two-view monocular geometry can become unreliable under:

- Pure or nearly pure rotation
- Small translation
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

## Quality-gate limitations

The current gate uses:

```text
Minimum cheirality count: 30
Minimum cheirality ratio: 0.15
```

The benchmark shows that accepted poses are substantially better on average,
but some accepted outliers remain.

The current gate must not be interpreted as a guarantee of accuracy.

## Evaluation coverage

The diagnostic benchmark:

- Covers a subset of `MH_01_easy`
- Does not evaluate every camera frame
- Does not evaluate all EuRoC sequences
- Does not evaluate live data
- Does not evaluate simulation
- Does not evaluate actual hardware
- Does not accumulate a trajectory

## Current scope

The project does not yet implement:

- Multi-frame trajectory accumulation
- Metric monocular scale recovery
- Stereo odometry
- Stereo depth
- Adaptive keyframe selection
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

# Planned Milestone 4B

Milestone 4B will explore candidate-frame selection and quality-gated
multi-frame pose accumulation.

Planned work:

1. Evaluate several candidate frame gaps for each reference frame.
2. Reject failed pose estimates.
3. Reject low-quality poses.
4. Add additional outlier indicators beyond cheirality.
5. Select the best supported candidate pose.
6. Accumulate accepted relative rotations.
7. Define a safe approach to translation scale before accumulating position.
8. Skip updates when no candidate is reliable.
9. Measure accumulated rotation drift.
10. Evaluate trajectory segments against EuRoC ground truth.

Milestone 4B must not blindly accumulate every frame-to-frame translation
direction.

A metric translation strategy must be defined before claiming a metric
trajectory.

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
Milestone 4A
Ground-truth relative-pose evaluation
        ↓
Milestone 4B
Candidate-frame selection and quality-gated accumulation
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

# License

A project license will be added in a separate repository update.