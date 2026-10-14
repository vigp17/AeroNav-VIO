# AeroNav-VIO Copilot Instructions

## Development environment

The project is currently developed natively on Windows using:

- PowerShell
- Python 3.14
- Visual Studio Code
- GitHub Copilot
- OpenCV 5

Use cross-platform Python code.

Do not use Bash-only commands in Python scripts or in Windows documentation.
Use `pathlib.Path` for filesystem paths.
Do not hardcode drive letters or user-specific absolute paths in source code.

## Project objective

Build and evaluate a GPS-denied autonomous drone navigation system using:

- Stereo-inertial SLAM
- Occupancy mapping
- A* path planning
- Localization-aware flight control
- PX4 SITL, ROS 2, and Gazebo in later milestones

## Completed milestones

### Milestone 1: EuRoC stereo dataset loader

Milestone 1 is complete.

The existing implementation:

- Parses EuRoC cam0 and cam1 CSV files
- Preserves timestamps as integer nanoseconds
- Pairs stereo images using exact timestamps
- Reports unmatched timestamps
- Loads requested images as grayscale NumPy arrays
- Does not load the complete dataset into memory
- Includes unit tests
- Includes a real-dataset inspection script

Reuse the existing `EurocSequence` implementation.

Do not duplicate, replace, or bypass the Milestone 1 loader unless a verified
bug requires the smallest possible correction.

## Current milestone

### Milestone 2: ORB temporal feature matching

The current milestone is limited to:

1. Loading two consecutive cam0 images through `EurocSequence`
2. Detecting ORB keypoints and descriptors
3. Matching binary descriptors using Hamming distance and KNN matching
4. Applying a configurable ratio test
5. Performing geometric verification using a fundamental matrix and RANSAC
6. Saving a match visualization under `outputs/`
7. Reporting matching statistics
8. Writing unit and script-level tests
9. Running the implementation on selected real EuRoC frame pairs

Use only the left-camera images from cam0 for temporal matching in this
milestone.

Do not use cam1 for stereo matching yet.

## Milestone 2 boundaries

Do not implement:

- Essential-matrix estimation
- Pose recovery
- Relative translation or rotation
- Trajectory accumulation
- Visual odometry
- Stereo depth
- SLAM
- Bundle adjustment
- Loop closure
- Camera calibration
- IMU processing
- Sensor fusion
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

Do not begin Milestone 3 unless explicitly requested.

## Engineering rules

- Use Python 3.11 or newer.
- Use the existing src package layout.
- Use `pathlib.Path` for all filesystem paths.
- Never concatenate Windows paths manually.
- Do not hardcode drive letters or user-specific absolute paths.
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
- Use real measured output when documenting results.

## OpenCV conventions

- Accept only nonempty, two-dimensional grayscale NumPy images unless a
  function explicitly documents otherwise.
- Do not silently resize, crop, normalize, or alter source images.
- Use ORB descriptors for this milestone.
- Use `cv2.NORM_HAMMING` for ORB descriptor matching.
