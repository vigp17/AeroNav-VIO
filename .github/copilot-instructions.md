# AeroNav-VIO Copilot Instructions

## Development environment

The project is currently developed natively on Windows using:

- PowerShell
- Python 3.14
- Visual Studio Code
- GitHub Copilot
- OpenCV 5

Use cross-platform Python code.

Do not use Bash-only commands in Python scripts or Windows documentation.

Use `pathlib.Path` for all filesystem paths.

Do not hardcode:

- Drive letters
- Usernames
- Dataset locations
- Repository locations
- Other user-specific absolute paths

Production source code must receive paths through function arguments,
command-line arguments, configuration values, or test-specific environment
variables.

## Project objective

Build and evaluate a GPS-denied autonomous drone navigation system using:

- Stereo-inertial localization
- Visual SLAM
- Occupancy mapping
- A* path planning
- Localization-aware flight control
- PX4 SITL, ROS 2, and Gazebo in later milestones

The project must progress incrementally.

Each milestone must remain:

- Independently testable
- Reproducible
- Documented
- Quantitatively evaluated
- Restricted to its stated scope

Do not begin a future milestone unless explicitly requested.

## Completed milestones

### Milestone 1: EuRoC stereo dataset loader

Milestone 1 is complete.

The existing implementation:

- Parses EuRoC `cam0` and `cam1` CSV files
- Preserves timestamps as integer nanoseconds
- Pairs stereo images using exact timestamps
- Reports unmatched timestamps
- Loads requested images as grayscale NumPy arrays
- Loads requested left-camera images independently
- Does not load the complete dataset into memory
- Includes unit tests
- Includes a real-dataset inspection script

Reuse the existing `EurocSequence` implementation.

Do not duplicate, replace, or bypass the Milestone 1 loader unless a verified
defect requires the smallest possible correction.

### Milestone 2: ORB temporal feature matching

Milestone 2 is complete.

The existing implementation:

- Loads `cam0` images through `EurocSequence`
- Detects ORB keypoints and binary descriptors
- Matches descriptors using Hamming-distance KNN matching
- Applies a configurable ratio test
- Performs geometric verification using fundamental-matrix RANSAC
- Produces structured feature and matching results
- Creates inlier-only match visualizations
- Includes unit and script-level tests
- Has been validated on real EuRoC `MH_01_easy` frame pairs

Reuse the existing Milestone 2 implementation, including:

- ORB feature extraction
- Hamming-distance descriptor matching
- KNN matching
- Ratio-test filtering
- Fundamental-matrix RANSAC verification
- Temporal matching result structures
- Existing visualization helpers

Do not duplicate or replace Milestone 2 components unless a verified defect
requires the smallest possible correction.

### Milestone 3: Calibrated relative camera pose estimation

Milestone 3 is complete.

The existing implementation:

- Loads and validates EuRoC `cam0/sensor.yaml`
- Constructs the pinhole camera intrinsic matrix
- Parses radial-tangential distortion coefficients
- Validates calibration resolution against loaded images
- Extracts Milestone 2 geometric-inlier correspondences
- Undistorts pixel coordinates into normalized camera coordinates
- Estimates essential-matrix candidates using RANSAC
- Handles stacked essential-matrix candidates deterministically
- Recovers relative camera rotation
- Recovers unit translation direction with unknown metric scale
- Applies positive-depth cheirality filtering
- Supports configurable frame gaps
- Applies cheirality-based pose-quality gating
- Produces structured JSON diagnostic reports
- Includes synthetic calibration, geometry, and script-level tests
- Has been evaluated on real EuRoC `MH_01_easy` frame pairs

Reuse the completed Milestone 3 implementation.

Do not duplicate, replace, or bypass Milestone 3 components unless a verified
defect requires the smallest possible correction.

Preserve the Milestone 3 relative-pose convention:

```text
X_current =
    R_current_reference @ X_reference
    + t_current_reference
```

The recovered translation remains a unit direction vector with unknown metric
scale.

Never treat the Milestone 3 translation vector as a displacement in meters.

## Current milestone

### Milestone 4A: Ground-truth relative-pose evaluation

The current milestone is limited to:

1. Parsing EuRoC ground-truth poses from:
   `mav0/state_groundtruth_estimate0/data.csv`
2. Preserving ground-truth timestamps as integer nanoseconds
3. Parsing ground-truth body positions
4. Parsing ground-truth body quaternions in scalar-first `wxyz` order
5. Validating and normalizing quaternions
6. Interpolating ground-truth position at camera timestamps
7. Interpolating ground-truth orientation using shortest-path quaternion SLERP
8. Rejecting timestamps outside the available ground-truth range
9. Converting interpolated body poses into `cam0` camera poses
10. Computing ground-truth relative `cam0` motion
11. Comparing Milestone 3 estimated rotation with ground-truth rotation
12. Comparing estimated translation direction with ground-truth translation
    direction
13. Reporting ground-truth translation magnitude in meters
14. Preserving the unknown-scale status of the estimated monocular translation
15. Evaluating whether the Milestone 3 pose-quality gate correlates with actual
    pose accuracy
16. Producing selected-pair JSON reports
17. Producing a small per-pair benchmark CSV
18. Producing aggregate benchmark JSON results
19. Documenting only actual measured results

Do not accumulate a multi-frame trajectory in Milestone 4A.

## Known ground-truth coverage constraint

For EuRoC `MH_01_easy`, the first camera timestamp precedes the first available
ground-truth timestamp by approximately 1.075 seconds.

The implementation must:

- Reject camera timestamps before the first ground-truth record
- Reject camera timestamps after the final ground-truth record
- Never extrapolate ground truth
- Return an informative out-of-range error
- Distinguish ground-truth unavailability from pose-estimation failure
- Demonstrate that evaluation beginning at camera frame `0` is rejected

A pair outside ground-truth coverage must not receive fabricated interpolated
poses or fabricated error metrics.

## Milestone 4A boundaries

Do not implement:

- Multi-frame trajectory accumulation
- Monocular metric-scale recovery
- Stereo odometry
- Stereo triangulation
- Stereo depth
- IMU integration
- IMU preintegration
- Visual-inertial fusion
- Keyframe trajectory accumulation
- Bundle adjustment
- Loop closure
- SLAM
- ROS 2
- PX4
- Gazebo
- WSL
- Docker
- Occupancy mapping
- Path planning
- Drone control
- Deep-learning models
- Databases
- Interactive GUI applications

Do not begin Milestone 4B unless explicitly requested.

## Engineering rules

- Use Python 3.11 or newer.
- Use the existing `src` package layout.
- Use `pathlib.Path` for all filesystem paths.
- Never concatenate Windows paths manually.
- Do not hardcode drive letters or usernames.
- Do not hardcode EuRoC calibration or ground-truth values in production code.
- Add type hints to public functions.
- Add concise and useful docstrings.
- Prefer immutable dataclasses for structured results.
- Keep functions small and independently testable.
- Prefer clear functions over unnecessary class hierarchies.
- Raise informative exceptions for invalid inputs.
- Do not suppress exceptions without adding useful context.
- Do not suppress failing tests.
- Do not weaken tests merely to make them pass.
- Never invent dataset results or performance measurements.
- Do not modify files unrelated to the current milestone.
- Make the smallest reasonable implementation.
- Use only actual measured output when documenting results.
- Preserve all existing Milestone 1, Milestone 2, and Milestone 3 behavior.
- All existing tests from Milestones 1 through 3 must continue to pass.

## General data conventions

- Preserve timestamps as integer nanoseconds.
- Load EuRoC images as grayscale NumPy arrays.
- Pixel coordinates use `(u, v)`.
- Preserve original image dimensions.
- Do not silently resize, crop, normalize, or alter source images.
- Do not modify EuRoC calibration files.
- Do not modify ground-truth files.
- Do not modify any other file in the EuRoC dataset.
- Do not load image data for the complete dataset into memory.
- Treat the first image as the reference frame.
- Treat the second image as the current frame.
- Use frame index `i` as the reference frame.
- Use frame index `i + frame_gap` as the current frame.

## Real dataset

The real EuRoC sequence is stored outside the Git repository.

Example sequence path:

```text
C:\Users\<username>\Desktop\datasets\euroc\MH_01_easy
```

The sequence root contains:

```text
MH_01_easy/
└── mav0/
    ├── cam0/
    │   ├── data/
    │   ├── data.csv
    │   └── sensor.yaml
    ├── cam1/
    │   ├── data/
    │   ├── data.csv
    │   └── sensor.yaml
    ├── imu0/
    ├── leica0/
    └── state_groundtruth_estimate0/
        ├── data.csv
        └── sensor.yaml
```

The dataset path may be supplied only through:

- A command-line argument
- A test-specific environment variable
- A function parameter

Never hardcode the example path in production source code.

Never:

- Copy the dataset into the repository
- Write generated files into the dataset
- Rename dataset files
- Delete dataset files
- Modify dataset CSV files
- Modify dataset images
- Modify dataset calibration files
- Modify ground-truth files

All generated reports and visualizations must be written under the repository's
`outputs/` directory.

## Camera calibration conventions

Load `cam0` calibration from:

```text
mav0/cam0/sensor.yaml
```

The calibration loader must continue to:

- Validate that the YAML root is a mapping
- Validate required fields before using them
- Preserve full floating-point precision
- Validate the camera model
- Validate the distortion model
- Validate image resolution
- Validate camera rate
- Validate intrinsic parameters
- Validate distortion coefficients
- Validate the body-to-sensor transform
- Produce informative errors for missing or malformed fields
- Never modify the YAML file

Support the existing EuRoC `cam0` configuration:

- Pinhole camera model
- Radial-tangential distortion model

Interpret camera intrinsics in this order:

```text
[fx, fy, cx, cy]
```

Build the intrinsic matrix as:

```text
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

Interpret radial-tangential distortion coefficients in this order:

```text
[k1, k2, p1, p2]
```

Verify that loaded image dimensions match calibration resolution whenever
calibration and images are used together.

## Image-coordinate conventions

Pixel coordinates use:

```text
(u, v)
```

where:

- `u` is the horizontal image coordinate
- `v` is the vertical image coordinate

Matched points begin as distorted pixel coordinates.

Use `cv2.undistortPoints` with:

- The parsed camera intrinsic matrix
- The parsed distortion coefficients
- No replacement projection matrix

This produces normalized, undistorted image coordinates.

When estimating an essential matrix from normalized coordinates, use the
identity camera matrix.

Do not apply the camera intrinsic matrix a second time after points have been
converted to normalized coordinates.

## Milestone 3 relative-pose convention

Preserve this convention:

```text
X_current =
    R_current_reference @ X_reference
    + t_current_reference
```

where:

- `X_reference` is a 3D point expressed in the reference-camera frame
- `X_current` is the same point expressed in the current-camera frame
- `R_current_reference` is a `3x3` rotation matrix
- `t_current_reference` is a three-element unit translation direction

The estimated translation-vector magnitude remains unknown.

Never report the Milestone 3 translation direction in meters.

Never compare the magnitude of the unit estimated translation vector directly
against ground-truth metric displacement.

## EuRoC ground-truth conventions

Load ground truth from:

```text
mav0/state_groundtruth_estimate0/data.csv
```

The required CSV fields are:

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

Interpret position as:

```text
p_world_body =
    [p_RS_R_x, p_RS_R_y, p_RS_R_z]
```

Interpret quaternion components in scalar-first order:

```text
q_world_body_wxyz =
    [q_RS_w, q_RS_x, q_RS_y, q_RS_z]
```

Ground-truth handling must:

- Preserve timestamps as integer nanoseconds
- Validate strictly increasing timestamp order
- Reject duplicate timestamps
- Reject unordered timestamps
- Reject nonfinite position values
- Reject nonfinite quaternion values
- Reject zero-norm quaternions
- Normalize quaternions before use
- Explicitly preserve the `wxyz` component convention
- Document the meaning and direction of every pose transform
- Never infer transform direction from variable names alone
- Never modify the ground-truth CSV or metadata

Velocity and bias columns are outside the Milestone 4A evaluation scope.

However, CSV rows must contain the expected number of fields before required
columns are accessed.

## Quaternion conventions

Quaternions must always be explicitly named and stored in scalar-first order:

```text
[w, x, y, z]
```

Quaternion utilities must:

- Accept only finite arrays with exactly four components
- Reject zero-norm quaternions
- Normalize input quaternions
- Normalize output quaternions
- Convert `wxyz` quaternions to valid `3x3` rotation matrices
- Validate rotation orthonormality
- Validate rotation determinant near `+1`
- Handle the equivalence of `q` and `-q`
- Use shortest-path interpolation
- Avoid numerical instability near identical quaternions

Do not silently treat `xyzw` input as `wxyz`.

## Ground-truth interpolation rules

For a requested camera timestamp `t`, find adjacent ground-truth timestamps:

```text
t0 <= t <= t1
```

If `t` exactly matches an available ground-truth timestamp, return the exact
sample.

Otherwise compute:

```text
alpha = (t - t0) / (t1 - t0)
```

Interpolate position linearly:

```text
p(t) = (1 - alpha) * p0 + alpha * p1
```

Interpolate orientation using shortest-path quaternion SLERP.

Before SLERP:

- Normalize both quaternions
- Compute their dot product
- If the dot product is negative, negate one quaternion
- Clamp numerical values before inverse trigonometric operations
- Use normalized linear interpolation when the quaternions are extremely close
- Normalize the final interpolated quaternion

Do not linearly interpolate quaternion components without normalization.

Never extrapolate before the first ground-truth timestamp or after the final
ground-truth timestamp.

Interpolation results must report:

- Requested timestamp
- Lower sample timestamp
- Upper sample timestamp
- Interpolation fraction
- Whether the result is an exact sample

## Ground-truth camera-pose conversion

EuRoC ground truth describes body motion, while Milestone 3 estimates `cam0`
camera motion.

Use the parsed `cam0` `T_BS` extrinsic to convert each interpolated body pose
into a `cam0` pose.

The implementation must:

- Explicitly document the EuRoC meaning of `T_BS`
- Derive every inversion and composition step
- Avoid relying on variable names alone
- Validate all body and camera transformations
- Verify homogeneous final rows
- Verify rotation orthonormality
- Verify rotation determinant near `+1`
- Use synthetic transforms to detect reversed multiplication order
- Produce ground-truth camera-relative motion in the same convention as
  Milestone 3

The final ground-truth relative pose must satisfy:

```text
X_current =
    R_current_reference @ X_reference
    + t_current_reference
```

This must match the Milestone 3 estimated relative-pose convention exactly
before errors are calculated.

## OpenCV matching conventions

Preserve the existing Milestone 2 conventions:

- Accept only nonempty, two-dimensional grayscale NumPy images unless a
  function explicitly documents otherwise.
- Use ORB descriptors.
- Use `cv2.NORM_HAMMING` for ORB descriptor matching.
- Use KNN matching with `k=2`.
- Apply a configurable ratio test.
- Use fundamental-matrix RANSAC for Milestone 2 geometric verification.
- Require at least eight correspondences before fundamental-matrix estimation.
- Handle missing descriptors and insufficient matches without crashing.
- Never call `cv2.imshow`.
- Visualization helpers must return NumPy arrays.
- Command-line scripts may save visualizations only under `outputs/`.

## Calibrated geometry conventions

Preserve the existing Milestone 3 conventions:

- Use `cv2.undistortPoints` with parsed camera calibration.
- Use normalized image coordinates for calibrated geometry.
- Use `cv2.findEssentialMat` for essential-matrix estimation.
- Use a configurable RANSAC probability.
- Use a configurable normalized-coordinate RANSAC threshold.
- Use the identity camera matrix with normalized coordinates.
- Use `cv2.recoverPose` for relative-pose recovery.
- Restrict pose recovery to essential-matrix RANSAC inliers.
- Require a conservative minimum of eight correspondences by default.
- Handle missing or malformed essential matrices.
- Handle stacked essential-matrix candidates deterministically.
- Validate all OpenCV-generated matrices and masks before using them.
- Ensure mask lengths are consistent with correspondence counts.
- Apply and report the cheirality mask.
- Never use the Milestone 2 fundamental matrix as the essential matrix.
- Never call `cv2.imshow`.
- Never write files from core geometry functions.

## Milestone 4A evaluation conventions

### Ground-truth relative pose

Ground-truth relative `cam0` pose must use the same convention as the estimated
Milestone 3 relative pose:

```text
X_current =
    R_current_reference @ X_reference
    + t_current_reference
```

Ground-truth translation may contain metric magnitude in meters.

Estimated Milestone 3 translation remains scale-free.

### Rotation error

Compare estimated and ground-truth relative rotations using a relative
rotation-error matrix consistent with the documented transform convention.

A valid formulation is:

```text
R_error =
    R_estimated @ R_ground_truth.T
```

or its mathematically equivalent form when derived consistently.

Compute rotation angular error from the clipped matrix trace:

```text
angle =
    arccos(
        clip(
            (trace(R_error) - 1) / 2,
            -1,
            1
        )
    )
```

Rotation error must be:

- Finite
- Reported in degrees
- Between `0` and `180` degrees

### Translation-direction error

Normalize ground-truth relative translation only when its metric magnitude is
greater than a configurable near-zero epsilon.

Compute angular error between:

```text
estimated unit translation direction
```

and:

```text
ground-truth unit translation direction
```

Do not silently apply an absolute value to the direction dot product.

Opposite translation directions are not equivalent unless the documented
relative-pose convention explicitly proves a sign ambiguity.

If ground-truth translation magnitude is smaller than the configured epsilon:

- Mark ground-truth translation direction as undefined
- Store translation-direction error as `None`
- Provide a clear reason
- Do not fabricate an error of zero degrees
- Do not fabricate an error of 180 degrees

Ground-truth translation magnitude may be reported in meters.

The Milestone 3 estimated translation magnitude remains unknown and must never
be compared directly with ground-truth translation magnitude.

### Quality-gate analysis

Preserve and report:

- Milestone 3 estimation success
- Milestone 3 pose-quality status
- Milestone 3 pose-quality reason
- Calibrated correspondence count
- Essential-inlier count
- Cheirality-inlier count
- Cheirality-inlier ratio
- Rotation error
- Translation-direction error when defined
- Ground-truth translation magnitude

Milestone 4A must measure whether quality-gate accepted poses are more accurate
than rejected poses.

Do not assume that the existing Milestone 3 quality gate is effective before
measuring its errors.

## Structured-result conventions

Prefer immutable dataclasses for:

- Ground-truth samples
- Interpolated poses
- Camera poses
- Ground-truth relative poses
- Relative-pose errors
- Relative-pose evaluations
- Benchmark pair results
- Aggregate benchmark results
- Failure information

A failed result must not contain fabricated transforms, directions, errors, or
statistics.

Use `None`, empty arrays, or another explicit representation documented by the
public interface.

## Testing rules

- Unit tests must use synthetic data.
- Unit tests must never depend on the real EuRoC dataset.
- Script-level tests should use synthetic EuRoC-style directories.
- OpenCV results may vary across versions.
- Tests must not make brittle assertions about exact OpenCV RANSAC counts.
- Prefer invariant-based assertions.
- Do not mutate input arrays in tests or production functions.
- All existing tests from Milestones 1 through 3 must continue to pass.

Useful invariant-based assertions include:

- Timestamps preserve integer precision
- Interpolation fractions remain between `0` and `1`
- Quaternion norms are close to `1`
- Rotation matrices have shape `3x3`
- Rotation matrices are finite
- Rotation matrices are orthonormal
- Rotation determinants are close to `+1`
- Homogeneous transforms have shape `4x4`
- Homogeneous final rows are `[0, 0, 0, 1]`
- Rotation errors are between `0` and `180` degrees
- Translation-direction errors are between `0` and `180` degrees when defined
- Ground-truth translation magnitudes are nonnegative
- JSON output contains only finite serializable values
- Expected output files exist and have nonzero size
- Input arrays are not mutated

## Milestone 4A ground-truth testing rules

Ground-truth tests must use synthetic CSV files under `pytest`'s `tmp_path`.

Tests must cover:

- Integer timestamp precision
- Successful position parsing
- Scalar-first `wxyz` quaternion parsing
- Quaternion normalization
- Quaternion sign equivalence
- Quaternion-to-rotation conversion
- Shortest-path SLERP
- Near-identical quaternion interpolation
- Exact timestamp lookup
- Midpoint position interpolation
- Midpoint orientation interpolation
- First valid timestamp
- Last valid timestamp
- Timestamp before ground-truth range
- Timestamp after ground-truth range
- Duplicate timestamps
- Unordered timestamps
- Missing file
- Empty file
- Malformed rows
- Insufficient row columns
- Nonfinite position
- Nonfinite quaternion
- Zero-norm quaternion

## Milestone 4A transform testing rules

Synthetic transform tests must cover:

- Identity body pose with nonidentity camera extrinsic
- Translated body pose
- Rotated body pose
- Combined body motion and camera extrinsic
- Transform inversion
- Transform composition order
- Camera-relative transform in the Milestone 3 convention
- Reversed-composition detection
- Homogeneous-transform validation
- Rotation orthonormality
- Rotation determinant

Synthetic transforms must be selected so that reversing multiplication order
produces a clearly different and test-detectable result.

## Milestone 4A error testing rules

Evaluation tests must cover:

- Zero rotation error for identical rotations
- Known rotation angular error
- Zero translation-direction error for matching directions
- Known translation-direction angular error
- Opposite translation directions
- Near-zero ground-truth translation
- Undefined translation direction
- Nonfinite estimated pose
- Convention mismatch protection
- Accepted Milestone 3 quality state
- Rejected Milestone 3 quality state
- No mutation of input arrays

## Command-line evaluation rules

The selected-pair evaluation script must:

- Use `argparse`
- Accept the EuRoC sequence root through `--sequence`
- Accept the reference frame through `--frame-index`
- Accept a positive `--frame-gap`
- Use `frame-index + frame-gap` as the current frame
- Load images and timestamps through `EurocSequence`
- Reuse Milestone 3 relative-pose estimation
- Load and interpolate EuRoC ground truth
- Convert body poses to `cam0` poses
- Compute ground-truth relative `cam0` pose
- Compute rotation error
- Compute translation-direction error when defined
- Expose the near-zero ground-truth translation epsilon
- Reject unavailable ground truth
- Never extrapolate ground truth
- Create the output parent directory
- Write JSON reports only under the requested output path
- Never write into the dataset
- Never open an interactive GUI window
- Return exit code `0` on successful evaluation
- Return a nonzero exit code for estimation or ground-truth failure
- Print concise errors with useful context
- Convert NumPy values safely for JSON
- Reject nonfinite JSON values

## Selected-pair JSON reporting rules

The evaluation JSON report must include:

- Sequence path
- Ground-truth CSV path
- Camera calibration path
- Reference frame index
- Current frame index
- Frame gap
- Reference camera timestamp
- Current camera timestamp
- Reference ground-truth lower timestamp
- Reference ground-truth upper timestamp
- Reference interpolation fraction
- Current ground-truth lower timestamp
- Current ground-truth upper timestamp
- Current interpolation fraction
- Estimated-pose success
- Milestone 3 pose-quality validity
- Milestone 3 pose-quality reason
- Calibrated correspondence count
- Essential-inlier count
- Cheirality-inlier count
- Cheirality-inlier ratio
- Estimated relative rotation matrix
- Ground-truth relative rotation matrix
- Estimated rotation angle
- Ground-truth rotation angle
- Rotation angular error
- Estimated unit translation direction
- Ground-truth translation vector
- Ground-truth unit translation direction when defined
- Ground-truth translation magnitude in meters
- Translation-direction angular error when defined
- Translation-direction-defined status
- Translation-direction undefined reason when applicable
- Transform convention
- Explicit statement that estimated monocular translation scale is unknown

Do not include NaN, infinity, or non-serializable NumPy values.

## Benchmark rules

The optional benchmark script must:

- Accept a sequence root
- Accept one or more frame gaps
- Accept a start frame
- Accept an end frame
- Accept a positive sampling stride
- Skip or explicitly report timestamps outside ground-truth coverage
- Run Milestone 3 estimation and Milestone 4A evaluation
- Distinguish ground-truth unavailability from estimation failure
- Distinguish estimation failure from quality-gate rejection
- Write per-pair CSV results
- Write aggregate JSON results
- Never treat skipped pairs as successful
- Never invent missing error values
- Avoid loading the complete image sequence into memory

Per-pair CSV columns must include:

- Reference index
- Current index
- Frame gap
- Reference timestamp
- Current timestamp
- Ground-truth availability
- Estimation success
- Pose-quality validity
- Pose-quality reason
- Calibrated correspondence count
- Essential-inlier count
- Cheirality-inlier count
- Cheirality-inlier ratio
- Estimated rotation angle
- Ground-truth rotation angle
- Rotation error
- Ground-truth translation magnitude
- Translation-direction error
- Translation-direction-defined status
- Failure category
- Failure reason

Aggregate benchmark output must report per frame gap:

- Total requested pairs
- Evaluated pairs
- Ground-truth unavailable count
- Estimation failure count
- Quality-gate accepted count
- Quality-gate rejected count
- Translation-direction-defined count
- Median rotation error
- Mean rotation error
- 90th-percentile rotation error
- Median translation-direction error when defined
- Mean translation-direction error when defined
- 90th-percentile translation-direction error when defined

Also report error metrics separately for:

- Quality-gate accepted poses
- Quality-gate rejected poses

Do not build a confusion matrix until explicit error thresholds for defining an
accurate pose have been selected and documented.

## Documentation rules

Update `README.md` with:

- Milestone 4A objective
- Status of Milestones 1 through 3
- Ground-truth CSV field definitions
- Ground-truth timestamp coverage limitation
- Position interpolation method
- Quaternion `wxyz` convention
- Shortest-path SLERP method
- Body-pose convention
- Camera-extrinsic convention
- Body-to-`cam0` conversion derivation
- Ground-truth relative-camera transform
- Milestone 3 relative-pose convention
- Rotation-error definition
- Translation-direction-error definition
- Near-zero translation handling
- Selected-pair commands
- Benchmark command
- JSON report description
- CSV report description
- Actual measured results
- Quality-gate accepted versus rejected results
- Assumptions
- Limitations
- Clear statement that trajectory accumulation remains unimplemented
- Planned Milestone 4B scope

Never invent real-data values.

Record only values produced by successful script execution.

State clearly that a small benchmark is diagnostic and not necessarily a full
sequence evaluation.

## Validation commands

Run the complete test suite:

```powershell
python -m pytest -q
```

Run linting:

```powershell
python -m ruff check .
```

Run selected real-data evaluation commands using the sequence path supplied
through the command line.

Generated JSON and CSV files must remain under `outputs/` and remain excluded
from version control.

## Definition of done

Milestone 4A is complete only when:

- Milestone 1 remains functional.
- Milestone 2 remains functional.
- Milestone 3 remains functional.
- All existing tests continue to pass.
- EuRoC ground-truth CSV files are parsed correctly.
- Ground-truth timestamps remain integer nanoseconds.
- Ground-truth positions are validated.
- Ground-truth `wxyz` quaternions are validated.
- Exact timestamp lookup works.
- Position interpolation works.
- Quaternion SLERP works.
- Timestamp extrapolation is rejected.
- Body poses are converted to `cam0` poses using validated transforms.
- Ground-truth relative `cam0` transforms match the Milestone 3 convention.
- Rotation error is calculated correctly.
- Translation-direction error is calculated correctly when defined.
- Near-zero translation is handled without fabricated direction errors.
- Camera frame `0` is rejected as outside ground-truth coverage.
- Selected real-data pairs are evaluated.
- A small benchmark is completed.
- Quality-gate accepted and rejected errors are reported separately.
- JSON and CSV reports contain finite, standards-compliant values.
- Unit tests pass.
- Script-level tests pass.
- Ruff passes.
- README instructions are updated.
- Actual measured results are documented.
- Assumptions and limitations are documented.
- The EuRoC dataset remains unchanged.
- Trajectory accumulation has not been implemented.
- Milestone 4B has not been started.