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

Do not hardcode drive letters, usernames, dataset locations, or other
user-specific absolute paths in production source code.

## Project objective

Build and evaluate a GPS-denied autonomous drone navigation system using:

- Stereo-inertial localization
- Visual SLAM
- Occupancy mapping
- A* path planning
- Localization-aware flight control
- PX4 SITL, ROS 2, and Gazebo in later milestones

The project must progress incrementally. Each milestone must remain independently
testable and documented.

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

- Loads consecutive `cam0` images through `EurocSequence`
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

## Current milestone

### Milestone 3: Calibrated relative camera pose estimation

The current milestone is limited to:

1. Parsing EuRoC `cam0/sensor.yaml`
2. Validating the camera calibration
3. Constructing the camera intrinsic matrix
4. Parsing radial-tangential distortion coefficients
5. Extracting Milestone 2 geometric-inlier coordinates
6. Undistorting matched feature coordinates
7. Estimating an essential matrix from calibrated correspondences
8. Recovering relative camera rotation
9. Recovering unit translation direction
10. Applying and reporting cheirality filtering
11. Generating structured relative-pose diagnostics
12. Writing JSON pose reports under `outputs/`
13. Adding synthetic calibration and geometry tests
14. Running selected real EuRoC frame pairs
15. Documenting actual measured results

Use the existing geometrically verified ORB correspondences from Milestone 2.

Use only `cam0` temporal correspondences for Milestone 3.

Do not use the Milestone 2 fundamental matrix as the essential matrix.

Do not describe recovered monocular translation as metric displacement. The
recovered translation represents direction only and has unknown scale.

## Milestone 3 boundaries

Do not implement:

- Metric translation scale
- Full trajectory accumulation
- Keyframe selection
- Bundle adjustment
- Loop closure
- Stereo triangulation
- Stereo depth
- IMU integration
- Visual-inertial fusion
- Full-sequence visual odometry
- SLAM
- ROS 2
- Gazebo
- PX4
- WSL
- Docker
- Occupancy mapping
- Path planning
- Drone control
- Deep-learning models
- Databases
- Interactive GUI windows

Do not begin Milestone 4 unless explicitly requested.

## Engineering rules

- Use Python 3.11 or newer.
- Use the existing `src` package layout.
- Use `pathlib.Path` for all filesystem paths.
- Never concatenate Windows paths manually.
- Do not hardcode drive letters or usernames.
- Do not hardcode EuRoC calibration values in production source code.
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
- Preserve all existing Milestone 1 and Milestone 2 behavior.
- Existing tests must continue to pass.

## Data conventions

- Preserve timestamps as integer nanoseconds.
- Load EuRoC images as grayscale NumPy arrays.
- Pixel coordinates use `(u, v)`.
- Preserve original image dimensions.
- Do not silently resize, crop, normalize, or alter source images.
- Do not modify EuRoC calibration files.
- Do not modify any other file in the EuRoC dataset.
- Do not load the complete dataset into memory.
- Treat the first image as the reference frame.
- Treat the second image as the current frame.
- Use frame index `i` as the reference frame.
- Use frame index `i + 1` as the current frame.

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

All generated reports and visualizations must be written under the repository's
`outputs/` directory.

## Calibration conventions

Load camera calibration from:

```text
mav0/cam0/sensor.yaml
```

The calibration loader must:

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

Support the EuRoC `cam0` calibration configuration used by this project:

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

Verify that loaded image dimensions match the calibration resolution whenever
calibration and images are used together.

## Coordinate conventions

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

This must produce normalized, undistorted image coordinates.

Normalized coordinates follow:

```text
x = (u - cx) / fx
y = (v - cy) / fy
```

after accounting for lens distortion through OpenCV's undistortion operation.

When estimating an essential matrix from normalized coordinates, use the
identity camera matrix.

Do not apply the camera intrinsic matrix a second time after points have been
converted to normalized coordinates.

## Relative-pose convention

Use this transform convention:

```text
X_current = R_current_reference @ X_reference + t_current_reference
```

where:

- `X_reference` is a 3D point expressed in the reference-camera frame
- `X_current` is the same 3D point expressed in the current-camera frame
- `R_current_reference` is a 3x3 rotation matrix
- `t_current_reference` is a three-element unit translation direction

The translation-vector magnitude is unknown.

Never report `t_current_reference` in meters.

Never describe the recovered translation direction as metric displacement.

The relative-pose implementation must:

- Return a finite 3x3 rotation matrix
- Return a finite three-element translation direction
- Validate rotation-matrix orthonormality
- Validate that the rotation determinant is close to `+1`
- Validate that the translation norm is close to `1`
- Calculate the relative rotation angle in degrees
- Report the essential-matrix inlier count
- Report the cheirality-inlier count
- Report inlier ratios safely
- Return structured failure information for insufficient geometry
- Return structured failure information for degenerate geometry

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

## Milestone 3 OpenCV conventions

- Use `cv2.undistortPoints` with parsed camera calibration.
- Use normalized image coordinates for calibrated geometry.
- Use `cv2.findEssentialMat` for essential-matrix estimation.
- Use a configurable RANSAC probability.
- Use a configurable normalized-coordinate RANSAC threshold.
- Use the identity camera matrix with normalized coordinates.
- Use `cv2.recoverPose` for relative-pose recovery.
- Restrict pose recovery to essential-matrix RANSAC inliers.
- Require a conservative minimum of eight correspondences by default.
- Handle missing essential matrices.
- Handle malformed essential matrices.
- Handle stacked essential-matrix candidates deterministically.
- Validate all OpenCV-generated matrices and masks before using them.
- Ensure mask lengths are consistent with correspondence counts.
- Apply and report the cheirality mask.
- Never use the Milestone 2 fundamental matrix as the essential matrix.
- Never call `cv2.imshow`.
- Never write files from core geometry functions.

## Structured-result conventions

Prefer immutable dataclasses for calibration, matching, statistics, pose
results, and failure information.

Relative-pose results should include:

- Success status
- Failure reason when unsuccessful
- Essential matrix
- Relative rotation matrix
- Unit translation direction
- Essential-matrix inlier mask
- Cheirality-inlier mask
- Normalized reference points
- Normalized current points
- Input correspondence count
- Essential-matrix inlier count
- Cheirality-inlier count
- Essential inlier ratio
- Cheirality inlier ratio
- Rotation angle in degrees
- Translation norm

A failed relative-pose result must not contain fabricated matrices or
statistics.

Use `None`, empty arrays, or another explicit representation documented by the
public interface.

## Testing rules

- Unit tests must use synthetic data.
- Unit tests must never depend on the real EuRoC dataset.
- Script-level tests should use synthetic EuRoC-style directories.
- Tests must not make brittle assertions about exact OpenCV RANSAC inlier
  counts.
- OpenCV results may vary across versions.
- Prefer invariant-based assertions.
- Do not mutate input arrays in tests or production functions.
- Existing Milestone 1 and Milestone 2 tests must continue to pass.

Useful invariant-based assertions include:

- Keypoint count is nonnegative
- Descriptor rows match keypoint count
- Descriptor columns match the ORB descriptor size
- Match indices remain within valid bounds
- Filtered-match count does not exceed raw-match count
- Geometric-inlier count does not exceed filtered-match count
- Essential-inlier count does not exceed input correspondence count
- Cheirality-inlier count does not exceed essential-inlier count
- Rotation matrix has shape `3x3`
- Rotation matrix is finite
- Rotation matrix is orthonormal within tolerance
- Rotation determinant is close to `+1`
- Translation direction contains three finite values
- Translation norm is close to `1`
- Masks have lengths consistent with their correspondence sets
- JSON output contains only finite, serializable values
- Expected output files exist and have nonzero size
- Input arrays are not mutated

## Milestone 3 calibration testing rules

Calibration tests must use synthetic YAML files under `pytest`'s `tmp_path`.

Calibration tests must cover:

- Successful calibration parsing
- Intrinsic-matrix construction
- Distortion-array construction
- Floating-point precision preservation
- Transform parsing
- Missing calibration file
- Invalid YAML root
- Missing required keys
- Incorrect intrinsic count
- Nonfinite intrinsic values
- Nonpositive focal lengths
- Invalid resolution
- Unsupported camera model
- Unsupported distortion model
- Incorrect distortion-coefficient count
- Invalid transform shape
- Invalid homogeneous-transform last row
- Nonfinite transform values
- Nonpositive camera rate

## Milestone 3 relative-pose testing rules

Relative-pose tests must use synthetic 3D points and known camera motion.

Synthetic pose tests must:

- Define a valid camera intrinsic matrix
- Generate 3D points in front of the reference camera
- Apply a known relative rotation and translation
- Ensure points are visible in both cameras
- Project points into both images
- Optionally add small, controlled image noise
- Optionally add controlled correspondence outliers
- Use deterministic random seeds
- Compare recovered rotation with the known rotation
- Compare recovered translation direction using angular error
- Verify rotation orthonormality
- Verify the rotation determinant
- Verify unit translation norm
- Verify correspondence and mask consistency
- Avoid exact assertions on RANSAC inlier counts

Tests must cover:

- Valid point undistortion
- Input arrays not mutated
- Invalid coordinate shape
- Nonfinite coordinates
- Too few correspondences
- Successful essential-matrix estimation
- Essential-matrix mask validation
- Stacked essential-matrix candidate handling
- Successful pose recovery
- Rotation-angle calculation
- Cheirality-mask processing
- Degenerate identical points
- Pure or nearly pure rotational motion
- Clear failure results instead of uncontrolled exceptions

## Command-line script rules

The Milestone 3 script must:

- Use `argparse`
- Accept the sequence root through `--sequence`
- Accept the reference frame through `--frame-index`
- Use `frame-index + 1` as the current frame
- Load images through `EurocSequence`
- Reuse Milestone 2 matching
- Load calibration from `cam0/sensor.yaml`
- Expose relevant ORB, matching, and essential-matrix parameters
- Create the output parent directory when necessary
- Write JSON reports only under the requested output path
- Never write into the dataset
- Never open an interactive GUI window
- Return exit code `0` on success
- Return a nonzero exit code on failure
- Print concise errors with useful context
- Convert NumPy values safely for JSON serialization
- Reject nonfinite values before writing JSON

## JSON reporting rules

The relative-pose JSON report must contain:

- Sequence path
- Calibration-file path
- Reference frame index
- Current frame index
- Reference timestamp in nanoseconds
- Current timestamp in nanoseconds
- Timestamp difference in nanoseconds
- Camera model
- Distortion model
- Calibration resolution
- `fx`
- `fy`
- `cx`
- `cy`
- Distortion coefficients
- Milestone 2 reference keypoint count
- Milestone 2 current keypoint count
- Milestone 2 ratio-filtered match count
- Milestone 2 geometric-inlier count
- Calibrated input-correspondence count
- Essential-matrix inlier count
- Cheirality-inlier count
- Essential inlier ratio
- Cheirality inlier ratio
- Essential matrix
- Relative rotation matrix
- Unit translation direction
- Translation norm
- Rotation angle in degrees
- Transform convention
- An explicit statement that translation scale is unknown

Do not include NaN, infinity, or non-serializable NumPy values in JSON output.

## Documentation rules

Update `README.md` with:

- Milestone 3 objective
- Calibration fields used
- Intrinsic-matrix definition
- Distortion model and coefficient order
- Normalized-coordinate convention
- Essential-matrix estimation pipeline
- Relative-pose transform convention
- Explanation of unknown translation scale
- Windows PowerShell commands
- JSON output description
- Actual measured real-data results
- Failure states and degeneracy cases
- Planned Milestone 4 scope

Never invent real-data values.

Record only values produced by successful script execution.

State clearly that selected frame pairs are diagnostic examples rather than a
complete sequence benchmark.

## Validation commands

Run the complete test suite:

```powershell
python -m pytest -q
```

Run linting:

```powershell
python -m ruff check .
```

Run selected real-data pose-estimation commands using the sequence path provided
through the command line.

Generated JSON files must remain under `outputs/` and remain excluded from
version control.

## Definition of done

Milestone 3 is complete only when:

- Milestone 1 remains functional.
- Milestone 2 remains functional.
- All existing tests continue to pass.
- EuRoC `cam0/sensor.yaml` is parsed and validated.
- The camera intrinsic matrix is constructed correctly.
- Distortion coefficients are parsed correctly.
- Matched pixel coordinates are undistorted.
- The essential matrix is estimated from normalized coordinates.
- Relative camera rotation is recovered.
- Unit translation direction is recovered.
- Translation scale is explicitly reported as unknown.
- Rotation orthonormality is validated.
- Rotation determinant is close to `+1`.
- Translation norm is close to `1`.
- Essential-matrix inliers are reported.
- Cheirality inliers are reported.
- Synthetic calibration tests pass.
- Synthetic geometry tests pass.
- Script-level tests pass.
- Ruff passes.
- Real EuRoC frame pairs run successfully.
- JSON reports are valid and reproducible.
- README usage instructions are updated.
- Actual measured results are documented.
- Assumptions and limitations are documented.
- The EuRoC dataset remains unchanged.
- Milestone 4 has not been started.