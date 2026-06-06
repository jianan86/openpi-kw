"""Rotation utility functions: 6D rotation <-> rotation matrix, geodesic loss.

Based on "On the Continuity of Rotation Representations in Neural Networks" (Zhou et al., CVPR 2019).
6D rotation = first two columns of rotation matrix flattened = 6 numbers.
"""

import numpy as np
import torch
import torch.nn.functional as F


def rot6d_to_mat_np(d6: np.ndarray) -> np.ndarray:
    """Convert 6D rotation to 3x3 rotation matrix via Gram-Schmidt (NumPy).

    Args:
        d6: (..., 6) array, first 3 elements = column 0, next 3 = column 1

    Returns:
        (..., 3, 3) rotation matrix (orthonormal)
    """
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-7)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-7)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-1)


def rot6d_to_mat_torch(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation to 3x3 rotation matrix via Gram-Schmidt (PyTorch).

    Args:
        d6: (..., 6) tensor

    Returns:
        (..., 3, 3) rotation matrix
    """
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def mat_to_rot6d_torch(mat: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to 6D representation (PyTorch).

    Args:
        mat: (..., 3, 3) rotation matrix

    Returns:
        (..., 6) tensor
    """
    col0 = mat[..., :, 0]
    col1 = mat[..., :, 1]
    return torch.cat((col0, col1), dim=-1)


def geodesic_loss(
    R_pred: torch.Tensor,
    R_target: torch.Tensor,
    reduce: bool = True,
    eps: float = 1e-7,
    return_degrees: bool = False,
) -> torch.Tensor:
    """Geodesic (angular) distance between two rotation matrices.

    Computes: theta = arccos((trace(R_pred^T @ R_target) - 1) / 2)

    Args:
        R_pred:  (..., 3, 3) predicted rotation matrices
        R_target: (..., 3, 3) ground truth rotation matrices
        reduce:   if True, return mean across all dims
        eps:      numerical stability
        return_degrees: if True, convert radians to degrees (range [0, 180])

    Returns:
        If reduce=True: scalar tensor
        If reduce=False: (...,) tensor of per-sample geodesic distances
    """
    R_err = torch.matmul(torch.transpose(R_pred, -2, -1), R_target)
    trace = torch.diagonal(R_err, dim1=-2, dim2=-1).sum(-1)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    loss = torch.acos(cos_theta)  # (...,)

    if return_degrees:
        loss = loss / torch.pi * 180.0

    if reduce:
        loss = loss.mean()
    return loss


def compute_rotation_error_degrees(rot6d_pred: torch.Tensor, rot6d_gt: torch.Tensor) -> float:
    """Evaluate: convert 6D->matrix, compute geodesic in degrees.

    Args:
        rot6d_pred: (B, ..., 6) predicted 6D rotation
        rot6d_gt:   (B, ..., 6) ground truth 6D rotation

    Returns:
        mean angular error in degrees (scalar)
    """
    R_pred = rot6d_to_mat_torch(rot6d_pred)
    R_gt = rot6d_to_mat_torch(rot6d_gt)
    return geodesic_loss(R_pred, R_gt, reduce=True, return_degrees=True).item()
