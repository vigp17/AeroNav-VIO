# AeroNav-VIO Copilot Instructions

## Development environment

The project is currently developed natively on Windows using:

- PowerShell
- Python 3.14
- Visual Studio Code
- GitHub Copilot
- OpenCV

Use cross-platform Python code. Do not use Bash-only commands in Python
scripts or in Windows documentation.

## Project objective

Build and evaluate a GPS-denied autonomous drone navigation system using:

- Stereo-inertial SLAM
- Occupancy mapping
- A* path planning
- Localization-aware flight control
- PX4 SITL, ROS 2, and Gazebo in later milestones

## Current milestone

The current milestone is limited to:

1. Loading one EuRoC sequence
2. Validating stereo-camera timestamps
3. Loading synchronized left and right images
4. Writing unit tests for the dataset loader

Do not add feature matching, visual odometry, ROS 2, Gazebo, PX4,
ORB-SLAM3, WSL, Docker, or deep-learning dependencies yet.

## Engineering rules

- Use Python 3.11 or newer.
- Use the src package layout.
- Use pathlib for all filesystem paths.
- Never concatenate Windows paths manually.
- Do not hardcode drive letters or absolute paths.
- Add type hints to public functions.
- Add concise docstrings.
- Keep functions small and independently testable.
- Raise informative exceptions for missing or malformed data.
- Write tests for normal behavior and error conditions.
- Run pytest and Ruff before declaring a task complete.
- Never suppress failing tests.
- Never invent dataset results or performance measurements.
- Do not modify files unrelated to the current milestone.

## Data conventions

- Preserve timestamps as integer nanoseconds.
- Load images as grayscale NumPy arrays.
- Pixel coordinates use `(u, v)`.
- Preserve EuRoC calibration files.
- Do not silently resize or crop images.
- Do not load the complete dataset into memory.

## Definition of done

A milestone is complete only when:

- The implementation runs using a documented PowerShell command.
- Unit tests pass.
- Ruff passes.
- README usage instructions are updated.
- Assumptions and limitations are documented.