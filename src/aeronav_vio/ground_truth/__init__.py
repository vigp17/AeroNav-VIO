"""EuRoC ground-truth loading, quaternion SLERP interpolation, and camera poses."""

from aeronav_vio.ground_truth.euroc import (
    CameraPose,
    EuRoCGroundTruthSequence,
    GroundTruthPose,
    InterpolatedGroundTruthPose,
    body_pose_to_camera_pose,
    invert_rigid_transform,
    normalize_quaternion_wxyz,
    quaternion_wxyz_to_rotation_matrix,
    slerp_quaternion_wxyz,
    validate_homogeneous_transform,
)

__all__ = [
    "CameraPose",
    "EuRoCGroundTruthSequence",
    "GroundTruthPose",
    "InterpolatedGroundTruthPose",
    "body_pose_to_camera_pose",
    "invert_rigid_transform",
    "normalize_quaternion_wxyz",
    "quaternion_wxyz_to_rotation_matrix",
    "slerp_quaternion_wxyz",
    "validate_homogeneous_transform",
]
