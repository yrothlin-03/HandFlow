import torch
import torch.nn.functional as F


MANO_JOINT_COUNT = 16
MANO_POSE_DIM = MANO_JOINT_COUNT * 3
MANO_ROTATION_6D_DIM = MANO_JOINT_COUNT * 6
MANO_HAND_DIM = 3 + MANO_ROTATION_6D_DIM
MANO_BIMANUAL_DIM = 2 * MANO_HAND_DIM


def axis_angle_to_matrix(axis_angle):
    if axis_angle.shape[-1] != 3:
        raise ValueError("axis_angle must have a final dimension of size 3")

    x, y, z = axis_angle.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        (
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ),
        dim=-1,
    ).reshape(*axis_angle.shape[:-1], 3, 3)

    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    first = torch.sinc(angle / torch.pi)[..., None]
    second = 0.5 * torch.sinc(angle / (2 * torch.pi)).square()[..., None]
    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return identity + first * skew + second * (skew @ skew)


def matrix_to_rotation_6d(matrix):
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("matrix must have final dimensions [3, 3]")
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(rotation):
    if rotation.shape[-1] != 6:
        raise ValueError("rotation must have a final dimension of size 6")

    first = rotation[..., :3]
    second = rotation[..., 3:]
    first = F.normalize(first, dim=-1, eps=1e-8)
    second = second - (first * second).sum(dim=-1, keepdim=True) * first
    second = F.normalize(second, dim=-1, eps=1e-8)
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-2)


def pose_axis_angle_to_rotation_6d(pose):
    if pose.shape[-1] != MANO_POSE_DIM:
        raise ValueError("pose must have a final dimension of size 48")
    rotations = pose.reshape(*pose.shape[:-1], MANO_JOINT_COUNT, 3)
    matrices = axis_angle_to_matrix(rotations)
    return matrix_to_rotation_6d(matrices).reshape(
        *pose.shape[:-1],
        MANO_ROTATION_6D_DIM,
    )


def handx_to_mano_6d(left_pose, left_trans, right_pose, right_trans):
    if left_pose.shape[-1] != MANO_POSE_DIM:
        raise ValueError("left_pose must have a final dimension of size 48")
    if right_pose.shape[-1] != MANO_POSE_DIM:
        raise ValueError("right_pose must have a final dimension of size 48")
    if (
        left_trans.shape[:-1] != left_pose.shape[:-1]
        or left_trans.shape[-1] != 3
    ):
        raise ValueError("left_trans is incompatible with left_pose")
    if (
        right_trans.shape[:-1] != right_pose.shape[:-1]
        or right_trans.shape[-1] != 3
    ):
        raise ValueError("right_trans is incompatible with right_pose")
    if left_pose.shape[:-1] != right_pose.shape[:-1]:
        raise ValueError("left_pose and right_pose are incompatible")

    left = torch.cat(
        (left_trans, pose_axis_angle_to_rotation_6d(left_pose)),
        dim=-1,
    )
    right = torch.cat(
        (right_trans, pose_axis_angle_to_rotation_6d(right_pose)),
        dim=-1,
    )
    return torch.cat((left, right), dim=-1)


def split_mano_6d(parameters):
    if parameters.shape[-1] != MANO_BIMANUAL_DIM:
        raise ValueError("parameters must have a final dimension of size 198")

    left = parameters[..., :MANO_HAND_DIM]
    right = parameters[..., MANO_HAND_DIM:]
    return {
        "left_parameters": left,
        "right_parameters": right,
        "left_trans": left[..., :3],
        "right_trans": right[..., :3],
        "left_rotation_6d": left[..., 3:].reshape(
            *left.shape[:-1], MANO_JOINT_COUNT, 6
        ),
        "right_rotation_6d": right[..., 3:].reshape(
            *right.shape[:-1], MANO_JOINT_COUNT, 6
        ),
    }
