import os
import cv2
import csv
import torch
import random
import numpy as np
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn.neighbors import NearestNeighbors
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryAUROC,
)

def save_stats(train_metrics: dict, test_metrics: dict, output_dir: str = "./logs"):
    # Overall statistics
    overall_stats = {
        split: {
            metric: calculate_stats([
                v for lens_metrics in metrics.values() 
                for m, values in lens_metrics.items() 
                if m == metric
                for v in values
            ])
            for metric in ["psnr", "ssim", "msssim", "lpips", "bbox_psnr", "bbox_ssim", "bbox_lpips"]
        }
        for split, metrics in [("train", train_metrics), ("test", test_metrics)]
    }

    # Save overall statistics (TXT)
    overall_path = os.path.join(output_dir, "overall_stats.txt")
    with open(overall_path, "w", encoding='utf-8') as f:
        for split, split_stats in overall_stats.items():
            f.write(f"\n{split.upper()} Set Statistics:\n")
            for metric, stats in split_stats.items():
                f.write(f"{metric}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})\n")
    
    # Save overall statistics (CSV)
    csv_path = os.path.join(output_dir, "overall_stats.csv")
    save_stats_to_csv(overall_stats, csv_path)

    # Per-lens statistics
    lens_stats = {
        split: {
            f"lens{int(cam_id):02d}": {
                metric: calculate_stats(values)
                for metric, values in lens_metrics.items()
            }
            for cam_id, lens_metrics in metrics.items()
        }
        for split, metrics in [("train", train_metrics), ("test", test_metrics)]
    }

    # Save per-camera statistics (TXT)
    lens_path = os.path.join(output_dir, "lens_stats.txt")
    with open(lens_path, "w") as f:
        for split, lenses in lens_stats.items():
            f.write(f"\n{split.upper()} Set Per-Lens Statistics:\n")
            for lens, metrics in lenses.items():
                f.write(f"\n{lens}:\n")
                for metric, stats in metrics.items():
                    f.write(f"  {metric}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})\n")

def save_stats_to_csv(overall_stats: dict, csv_path: str):
    """
    Save overall statistics to CSV format with metrics grouped by Train/Test columns.
    
    Format:
    psnr,psnr,ssim,ssim,...
    Train Views,Test Views,Train Views,Test Views,...
    34.877 ± 0.917,32.212 ± 2.041,0.889 ± 0.007,0.868 ± 0.015,...
    """
    # Define metric order
    metrics_order = ["psnr", "ssim", "msssim", "lpips", "bbox_psnr", "bbox_ssim", "bbox_lpips"]
    
    # Build the three rows
    header1 = []
    header2 = []
    values = []
    
    for metric in metrics_order:
        header1.append(metric)
        header1.append(metric)  # for test column
        
        header2.append("Train Views")
        header2.append("Test Views")
        
        # Train statistics
        if "train" in overall_stats and metric in overall_stats["train"]:
            train_stats = overall_stats["train"][metric]
            train_val = f"{train_stats['mean']:.3f} ± {train_stats['std']:.3f}"
        else:
            train_val = "N/A"
        
        # Test statistics
        if "test" in overall_stats and metric in overall_stats["test"]:
            test_stats = overall_stats["test"][metric]
            test_val = f"{test_stats['mean']:.3f} ± {test_stats['std']:.3f}"
        else:
            test_val = "N/A"
        
        values.append(train_val)
        values.append(test_val)
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header1)
        writer.writerow(header2)
        writer.writerow(values)
    
    print(f"CSV statistics saved to: {csv_path}")

def save_depth_stats_to_csv(overall_stats: dict, csv_path: str):
    """
    Save depth statistics to CSV format with metrics grouped by Train/Test columns.
    
    Format:
    mae,mae,abs_rel,abs_rel,...
    Train Views,Test Views,Train Views,Test Views,...
    0.1234 ± 0.0567,0.1456 ± 0.0678,0.0789 ± 0.0123,0.0812 ± 0.0145,...
    """
    # Define depth metric order
    metrics_order = ["mae", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    
    # Build the three rows
    header1 = []
    header2 = []
    values = []
    
    for metric in metrics_order:
        header1.append(metric)
        header1.append(metric)  # for test column
        
        header2.append("Train Views")
        header2.append("Test Views")
        
        # Train statistics
        if "train" in overall_stats and metric in overall_stats["train"]:
            train_stats = overall_stats["train"][metric]
            train_val = f"{train_stats['mean']:.3f} ± {train_stats['std']:.3f}"
        else:
            train_val = "N/A"
        
        # Test statistics
        if "test" in overall_stats and metric in overall_stats["test"]:
            test_stats = overall_stats["test"][metric]
            test_val = f"{test_stats['mean']:.3f} ± {test_stats['std']:.3f}"
        else:
            test_val = "N/A"
        
        values.append(train_val)
        values.append(test_val)
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header1)
        writer.writerow(header2)
        writer.writerow(values)
    
    print(f"Depth CSV statistics saved to: {csv_path}")


def calculate_stats(values):
    values = [v.cpu().numpy() if torch.is_tensor(v) else v for v in values]
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'count': len(values)
    }

def masked_l1_loss(pred, target, mask=None):
    if mask is None:
        return F.l1_loss(pred, target)
    
    valid_mask = mask > 0.5
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    return F.l1_loss(pred[valid_mask], target[valid_mask])

def masked_psnr(pred, target, mask):
    """Calculate PSNR only on masked regions"""
    # Handle dimension mismatch: expand mask to match pred/target shape
    if mask.dim() == 3 and pred.dim() == 4:
        # [B, H, W] -> [B, 1, H, W] -> [B, C, H, W]
        mask = mask.unsqueeze(1).expand_as(pred)
    
    valid_mask = mask > 0.5
    if valid_mask.sum() == 0:
        return 0.0
        
    # Calculate MSE only on valid pixels
    mse = torch.mean((pred[valid_mask] - target[valid_mask]) ** 2)
    return 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)
    
def slerp_quat(q1, q2, t):
    """Spherical linear interpolation between two quaternions (torch tensors)."""
    # Ensure shortest path interpolation
    dot = (q1 * q2).sum()
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    dot = torch.clamp(dot, -1.0, 1.0)
    theta = torch.acos(dot)
    
    if theta < 1e-6:
        return q1
    
    return (torch.sin((1-t)*theta) * q1 + torch.sin(t*theta) * q2) / torch.sin(theta)

def quat_multiply_batched(q1: Tensor, q2: Tensor, left_multi=False) -> Tensor:
    # Ensure proper broadcasting
    if q2.dim() < q1.dim():
        q2 = q2.view(*([1]*(q1.dim()-q2.dim())), * q2.shape)
    if left_multi:
        # Left multiplication q1 * q2
        w = q1[..., 0]*q2[..., 0] - q1[..., 1]*q2[..., 1] - q1[..., 2]*q2[..., 2] - q1[..., 3]*q2[..., 3]
        x = q1[..., 0]*q2[..., 1] + q1[..., 1]*q2[..., 0] + q1[..., 2]*q2[..., 3] - q1[..., 3]*q2[..., 2]
        y = q1[..., 0]*q2[..., 2] - q1[..., 1]*q2[..., 3] + q1[..., 2]*q2[..., 0] + q1[..., 3]*q2[..., 1]
        z = q1[..., 0]*q2[..., 3] + q1[..., 1]*q2[..., 2] - q1[..., 2]*q2[..., 1] + q1[..., 3]*q2[..., 0]
    else:
        # Right multiplication q2 * q1
        w = q2[..., 0]*q1[..., 0] - q2[..., 1]*q1[..., 1] - q2[..., 2]*q1[..., 2] - q2[..., 3]*q1[..., 3]
        x = q2[..., 0]*q1[..., 1] + q2[..., 1]*q1[..., 0] + q2[..., 2]*q1[..., 3] - q2[..., 3]*q1[..., 2]
        y = q2[..., 0]*q1[..., 2] - q2[..., 1]*q1[..., 3] + q2[..., 2]*q1[..., 0] + q2[..., 3]*q1[..., 1]
        z = q2[..., 0]*q1[..., 3] + q2[..., 1]*q1[..., 2] - q2[..., 2]*q1[..., 1] + q2[..., 3]*q1[..., 0]

    return torch.stack([w, x, y, z], dim=-1)

def quat_multiply(q1: Tensor, q2: Tensor, left_multi=False) -> Tensor:
    if left_multi:
        # Left multiplication q1 * q2
        w = q1[:, 0]*q2[0] - q1[:, 1]*q2[1] - q1[:, 2]*q2[2] - q1[:, 3]*q2[3]
        x = q1[:, 0]*q2[1] + q1[:, 1]*q2[0] + q1[:, 2]*q2[3] - q1[:, 3]*q2[2]
        y = q1[:, 0]*q2[2] - q1[:, 1]*q2[3] + q1[:, 2]*q2[0] + q1[:, 3]*q2[1]
        z = q1[:, 0]*q2[3] + q1[:, 1]*q2[2] - q1[:, 2]*q2[1] + q1[:, 3]*q2[0]
    else:
        # Right multiplication q2 * q1
        w = q2[0]*q1[:, 0] - q2[1]*q1[:, 1] - q2[2]*q1[:, 2] - q2[3]*q1[:, 3]
        x = q2[0]*q1[:, 1] + q2[1]*q1[:, 0] + q2[2]*q1[:, 3] - q2[3]*q1[:, 2]
        y = q2[0]*q1[:, 2] - q2[1]*q1[:, 3] + q2[2]*q1[:, 0] + q2[3]*q1[:, 1]
        z = q2[0]*q1[:, 3] + q2[1]*q1[:, 2] - q2[2]*q1[:, 1] + q2[3]*q1[:, 0]

    return torch.stack([w, x, y, z], dim=1)

def batched_rotmat_to_quat(R: Tensor) -> Tensor:
    # Validate input shape
    if R.ndim != 4 or R.shape[-2:] != (3, 3):
        raise ValueError(f"Input must have shape [C, N, 3, 3], got {R.shape}")

    # Compute trace of each rotation matrix: tr = R00 + R11 + R22
    tr = R[:, :, 0, 0] + R[:, :, 1, 1] + R[:, :, 2, 2]  # Shape: [C, N]

    # Define masks for each case
    case0 = tr > 0
    case1 = (~case0) & (R[:, :, 0, 0] > R[:, :, 1, 1]) & (R[:, :, 0, 0] > R[:, :, 2, 2])
    case2 = (~case0) & (~case1) & (R[:, :, 1, 1] > R[:, :, 2, 2])

    # Compute S for each case
    S0 = torch.sqrt(tr + 1.0) * 2  # [C, N]
    S1 = torch.sqrt(1.0 + R[:, :, 0, 0] - R[:, :, 1, 1] - R[:, :, 2, 2]) * 2  # [C, N]
    S2 = torch.sqrt(1.0 + R[:, :, 1, 1] - R[:, :, 0, 0] - R[:, :, 2, 2]) * 2  # [C, N]
    S3 = torch.sqrt(1.0 + R[:, :, 2, 2] - R[:, :, 0, 0] - R[:, :, 1, 1]) * 2  # [C, N]
    
    # Initialize quaternion components
    q0 = torch.where(
        case0, 
        0.25 * S0, 
        torch.where(
            case1, 
            (R[:, :, 2, 1] - R[:, :, 1, 2]) / S1, 
            torch.where(
                case2, 
                (R[:, :, 0, 2] - R[:, :, 2, 0]) / S2, 
                (R[:, :, 1, 0] - R[:, :, 0, 1]) / S3
            )
        )
    ) 
    
    q1 = torch.where(
        case0,
        (R[:, :, 0, 1] + R[:, :, 1, 0]) / S0,
        torch.where(
            case1,
            0.25 * S1,
            torch.where(
                case2,
                (R[:, :, 0, 1] + R[:, :, 1, 0]) / S2,
                (R[:, :, 0, 2] + R[:, :, 2, 0]) / S3
            )
        )
    )

    q2 = torch.where(
        case0,
        (R[:, :, 0, 2] - R[:, :, 2, 0]) / S0,
        torch.where(
            case1,
            (R[:, :, 0, 2] + R[:, :, 2, 0]) / S1,
            torch.where(
                case2,
                0.25 * S2,
                (R[:, :, 1, 2] + R[:, :, 2, 1]) / S3
            )
        )
    )

    q3 = torch.where(
        case0,
        (R[:, :, 1, 0] - R[:, :, 0, 1]) / S0,
        torch.where(
            case1,
            (R[:, :, 1, 2] + R[:, :, 2, 1]) / S1,
            torch.where(
                case2,
                (R[:, :, 1, 2] + R[:, :, 2, 1]) / S2,
                0.25 * S3
            )
        )
    )
    
    # Stack quaternion components into a single tensor
    q = torch.stack([q0, q1, q2, q3], dim=-1)  # Shape: [C, N, 4]

    return q

def rotmat_to_quat(R: Tensor) -> Tensor:
    # Ensure R is a proper rotation matrix of shape (3, 3)
    if R.shape != (3, 3):
        raise ValueError("Input must be a 3x3 matrix.")

    # Allocate space for the quaternion
    q = torch.empty(4, device=R.device, dtype=R.dtype)

    # Compute the trace of the matrix
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0:
        S = torch.sqrt(tr + 1.0) * 2  # S=4*qw
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S

    return q

def quat_to_rotmat_batched(qs):
    assert qs.shape[-1] == 4, f"Last dimension must be 4, got {qs.shape}"
    # Split quaternion components
    w, x, y, z = qs[..., 0], qs[..., 1], qs[..., 2], qs[..., 3]

    # Initialize output tensor with correct batch dimensions
    batch_shape = qs.shape[:-1]
    rotms = torch.zeros((*batch_shape, 3, 3), device=qs.device, dtype=qs.dtype)

    # Compute the rotation matrices
    rotms[..., 0, 0] = 1 - 2*y*y - 2*z*z
    rotms[..., 0, 1] = 2*x*y - 2*z*w
    rotms[..., 0, 2] = 2*x*z + 2*y*w
    rotms[..., 1, 0] = 2*x*y + 2*z*w
    rotms[..., 1, 1] = 1 - 2*x*x - 2*z*z
    rotms[..., 1, 2] = 2*y*z - 2*x*w
    rotms[..., 2, 0] = 2*x*z - 2*y*w
    rotms[..., 2, 1] = 2*y*z + 2*x*w
    rotms[..., 2, 2] = 1 - 2*x*x - 2*y*y

    return rotms

def quat_to_rotmat(qs):
    w, x, y, z = qs[:, 0], qs[:, 1], qs[:, 2], qs[:, 3]
    N = qs.shape[0]

    # Compute the rotation matrices
    rotms = torch.zeros((N, 3, 3), device=qs.device, dtype=qs.dtype)
    rotms[:, 0, 0] = 1 - 2*y*y - 2*z*z
    rotms[:, 0, 1] = 2*x*y - 2*z*w
    rotms[:, 0, 2] = 2*x*z + 2*y*w
    rotms[:, 1, 0] = 2*x*y + 2*z*w
    rotms[:, 1, 1] = 1 - 2*x*x - 2*z*z
    rotms[:, 1, 2] = 2*y*z - 2*x*w
    rotms[:, 2, 0] = 2*x*z - 2*y*w
    rotms[:, 2, 1] = 2*y*z + 2*x*w
    rotms[:, 2, 2] = 1 - 2*x*x - 2*y*y

    return rotms

def rotmat_to_quat_batched(rotation_matrices: Tensor) -> Tensor:
    """
    Converts a batch of 3x3 rotation matrices to quaternions.

    :param rotation_matrices: Tensor of shape (N, 3, 3) representing N 3x3 rotation matrices.
    :return: Tensor of shape (N, 4) representing N quaternions (w, x, y, z).
    """
    assert rotation_matrices.shape[-2:] == (3, 3), "Input should be a batch of 3x3 matrices."
    
    # Ensure all calculations happen on the same device as the input tensor
    device = rotation_matrices.device
    dtype = rotation_matrices.dtype
    
    # Pre-allocate quaternion tensor on the correct device and dtype
    N = rotation_matrices.shape[0]
    quaternions = torch.zeros((N, 4), device=device, dtype=dtype)

    # Extract rotation matrix elements
    R = rotation_matrices
    t = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    # Compute the trace-based branch
    cond = t > 0
    S = torch.sqrt(t[cond] + 1.0).to(device) * 2  # S = 4 * qw
    quaternions[cond, 0] = 0.25 * S
    quaternions[cond, 1] = (R[cond, 2, 1] - R[cond, 1, 2]) / S
    quaternions[cond, 2] = (R[cond, 0, 2] - R[cond, 2, 0]) / S
    quaternions[cond, 3] = (R[cond, 1, 0] - R[cond, 0, 1]) / S

    # Compute the largest diagonal element branch
    cond = ~cond
    r_max = torch.argmax(R[cond].diagonal(dim1=-2, dim2=-1), dim=-1)
    S_max = torch.sqrt(1.0 + 2.0 * R[cond, r_max, r_max] - t[cond]).to(device) * 2
    idx = torch.arange(N, device=device)[cond]
    
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        is_i = (r_max == i)
        quaternions[idx[is_i], i+1] = 0.25 * S_max[is_i]
        quaternions[idx[is_i], 0] = (R[idx[is_i], k, j] - R[idx[is_i], j, k]) / S_max[is_i]
        quaternions[idx[is_i], j+1] = (R[idx[is_i], j, i] + R[idx[is_i], i, j]) / S_max[is_i]
        quaternions[idx[is_i], k+1] = (R[idx[is_i], k, i] + R[idx[is_i], i, k]) / S_max[is_i]

    return quaternions

def batch_quat_mul(q1: Tensor, q2: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Hamilton product, scalar-first (w,x,y,z). Supports (...,4).
    """
    # Ensure the last dimension is 4 for both quaternions
    assert q1.shape[-1] == 4 and q2.shape[-1] == 4

    # Ensure proper broadcasting
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    # Compute the Hamilton product
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    # Stack the results into a quaternion tensor
    q3 = torch.stack((w, x, y, z), dim=-1)
    
    # Branchless normalization
    s = (q3 * q3).sum(dim=-1, keepdim=True).clamp_min(eps)
    q3 = q3 * torch.rsqrt(s)
    
    return q3

def get_positional_encodings(height: int, width: int, dim: int, device: str = "cuda") -> Tensor:
    
    # Generate grid of (x, y) coordinates
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )

    # Normalize coordinates to the range of [0, 1]
    y = y / (height - 1)
    x = x / (width - 1)

    # Create frequence range [1, 2, 4, ..., 2^(dim-1)]
    frequencies = torch.pow(2, torch.arange(dim, device=device)).float() * torch.pi

    # Compute sine and cosine of the frequencies multiplied by the coordinates
    y_encodings = torch.cat([torch.cos(y.unsqueeze(-1) * frequencies), 
                             torch.cos(y.unsqueeze(-1) * frequencies)], dim=-1)
    x_encodings = torch.cat([torch.sin(x.unsqueeze(-1) * frequencies),  
                             torch.cos(x.unsqueeze(-1) * frequencies)], dim=-1)
    
    # Concatenate the encodings along the channel dimension
    pos_encodings = torch.cat([y_encodings, x_encodings], dim=-1)

    return pos_encodings

def normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
    """
    Converts normalized quaternion to rotation matrix.
    Args:
        quat: normalized quaternion of size (*, 4)

    Returns:
        batch of rotation matrices of size (*, 3, 3)
    """
    assert quat.shape[-1] == 4, quat.shape
    # q = q / torch.norm(q, dim=-1, keepdim=True)
    w, x, y, z = torch.unbind(quat, dim=-1)
    rotmat = torch.stack(
        [
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)],
        ],
        dim=-1,
    )
    return rotmat

def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)

def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.
    - Data loading
    - Model initialization
    - Training process
    - Multi-GPU support"""
    # Basic seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # GPU settings
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Multi-process seeds
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Return generator for torch dataloader
    return torch.Generator().manual_seed(seed)

# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/general_utils.py#L163
def colormap(img, cmap="jet"):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H / dpi, W / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data).float().permute(2, 0, 1)
    plt.close()
    return img

def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]

def apply_depth_colormap(
    depth: torch.Tensor,
    acc: torch.Tensor = None,
    near_plane: float = None,
    far_plane: float = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth, colormap="turbo")
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img


def anneal_linear(step: int, total=4000, t_start=0.1, t_end=0.01):
    frac = min(step / total, 1.0)
    return t_start + (t_end - t_start) * frac

def anneal_exp(step: int, total=4000, t_start=0.1, t_end=0.01):
    frac = min(step / total, 1.0)
    return t_start * (t_end / t_start) ** frac

def piecewise_tau(step: int, t0: float, t1: float, t2: float, A: int, B: int) -> float:
    """ Piecewise linear interpolation for a value that changes over time.
    - t0: value at step A
    - t1: value at step B
    - t2: value after step B
    - A: step at which t0 is valid
    - B: step at which t1 is valid
    """
    assert A < B, f"Invalid A={A} and B={B} values"
    assert step >= 0, f"Step must be non-negative, got {step}"
    if step <= A: return t0
    if step <= B:
        a = (step - A) / max(1.0, (B - A))
        return t0 + a * (t1 - t0)
    a2 = min(1.0, (step - B) / max(1.0, (B - A)))
    return t1 + a2 * (t2 - t1)

def piecewise_linear(step: int, v0: float, v1: float, A: int, B: int) -> float:
    """ Piecewise linear interpolation for a value that changes over time.
    - v0: value at step A
    - v1: value at step B
    - A: step at which v0 is valid
    - B: step at which v1 is valid
    """
    assert A < B, f"Invalid A={A} and B={B} values"
    assert step >= 0, f"Step must be non-negative, got {step}"
    if step <= A: return v0
    if step <= B:
        a = (step - A) / max(1.0, (B - A))
        return v0 + a * (v1 - v0)
    return v1


def save_hist(data, title, xlabel, filename, save_dir, color='orange'):
    plt.figure()
    plt.hist(data, bins=100, range=(0,1), color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def save_scatter(x, y, xlabel, ylabel, filename, save_dir, step=None):
    plt.figure()
    plt.scatter(x, y, alpha=0.3, s=2)
    if step is None:
        plt.title(f"{ylabel} vs. {xlabel}")
    else:
        plt.title(f"{ylabel} vs. {xlabel} @ step {step}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def scatter_map(H, W, u, v, val):
    img = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.int32)
    # Create mask for valid (non-NaN) values
    valid_mask = ~(np.isnan(u) | np.isnan(v))
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    val_valid = val[valid_mask]
    for i in range(len(u_valid)):
        x, y = int(u_valid[i]), int(v_valid[i])
        if 0 <= x < W and 0 <= y < H:
            img[y, x] += val_valid[i]
            count[y, x] += 1
    # for i in range(len(u)):
    #     x, y = int(u[i]), int(v[i])
    #     if 0 <= x < W and 0 <= y < H:
    #         img[y, x] += val[i]
    #         count[y, x] += 1
    count[count == 0] = 1
    return img / count


@torch.no_grad()
def calculate_bbox_iou(bbox1, bbox2):
    """Calculate IoU between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return float(intersection / union) if union > 0 else 0.0

@torch.no_grad()
def merge_close_bboxes(bboxes, iou_threshold=0.3, max_passes=4):
    if len(bboxes) <= 1:
        return bboxes
    bxs = list(bboxes)
    changed = True
    passes = 0
    while changed and passes < max_passes:
        changed = False
        kept = []
        used = [False]*len(bxs)
        for i in range(len(bxs)):
            if used[i]:
                continue
            cur = list(bxs[i])
            for j in range(i+1, len(bxs)):
                if used[j]:
                    continue
                if calculate_bbox_iou(cur, bxs[j]) > iou_threshold:
                    # merge
                    cur[0] = min(cur[0], bxs[j][0])
                    cur[1] = min(cur[1], bxs[j][1])
                    cur[2] = max(cur[2], bxs[j][2])
                    cur[3] = max(cur[3], bxs[j][3])
                    used[j] = True
                    changed = True
            used[i] = True
            kept.append(tuple(cur))
        bxs = kept
        passes += 1
    return bxs

@torch.no_grad()
def extract_dyn_bboxes(
    dyn_mask_2d: torch.Tensor, 
    min_area: int = 100, 
    merge_close: bool = True,
    iou_threshold: float = 0.3,
    min_side: int = 16,
    morphology: bool = False,
    bbox_dilat: int = 8
):
    """
    Enhanced bbox extraction for fisheye dynamic scenes.
    
    Args:
        dyn_mask_2d: [H, W] binary mask (already excludes rig_range)
        min_area: Minimum area threshold
        merge_close: Merge nearby bboxes (IoU > 0.3)
        iou_threshold: IoU threshold for merging.
        min_side: Minimum side length for bboxes.
        morphology: apply small open+close to clean noise.
        bbox_dilat: Dilation size for bbox extraction.

    Returns:
        List of bboxes [(x1, y1, x2, y2), ...]
    """
    if dyn_mask_2d is None or not dyn_mask_2d.any():
        return []
        
    mask_np = dyn_mask_2d.cpu().numpy().astype(np.uint8)
    H, W = mask_np.shape

    if morphology:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, k)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, k)
    
    # Find connected components
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < min_side or h < min_side:
            continue
        # Apply fixed bbox dilation
        x1_dilated = max(0, x - bbox_dilat)
        y1_dilated = max(0, y - bbox_dilat)  
        x2_dilated = min(W, x + w + bbox_dilat)
        y2_dilated = min(H, y + h + bbox_dilat)
        bboxes.append((int(x1_dilated), int(y1_dilated), int(x2_dilated), int(y2_dilated)))

    if merge_close and len(bboxes) > 1:
        bboxes = merge_close_bboxes(bboxes, iou_threshold=iou_threshold)
    
    return bboxes

@torch.no_grad()
def eval_splat_dyn(
    splats_dynamicness: torch.Tensor, # [N] canonical dynamicness logits
    deform_params,                    # deformation parameters (can be None)
    info: dict,                       # rasterization info with means2d, radii
    dynamic_masks_gt: torch.Tensor,   # [B, H, W] ground truth masks
    device: torch.device
) -> dict:
    """
    Evaluate dynamicness classification performance against ground truth masks.
    
    Args:
        splats_dynamicness: [N] canonical dynamicness logits from splats
        deform_params: deformation parameters (has dynamic_offset if available)
        info: rasterization info containing means2d [B,N,2] and radii [B,N] or [B,N,2]
        dynamic_masks_gt: [B,H,W] ground truth binary dynamic masks
        device: computation device
        
    Returns:
        dict with classification metrics: accuracy, precision, recall, f1, auroc
    """
    # Initialize metrics objects
    # dyn_accuracy = BinaryAccuracy().to(device)
    # dyn_precision = BinaryPrecision().to(device)
    # dyn_recall = BinaryRecall().to(device)
    dyn_f1 = BinaryF1Score().to(device)
    dyn_auroc = BinaryAUROC().to(device)
    
    means2d = info.get("means2d")  # [B, N, 2]
    radii = info.get("radii")      # [B, N] or [B, N, 2]
    
    if means2d is None or radii is None:
        return {}
    
    B = means2d.shape[0]
    _, H, W = dynamic_masks_gt.shape
        
    # Get predicted dynamicness scores
    if deform_params is not None and hasattr(deform_params, 'dynamic_offset') and deform_params.dynamic_offset is not None:
        # Use the final combined dynamicness for evaluation
        pred_dyn_logits = splats_dynamicness + deform_params.dynamic_offset
    else:
        # Fallback to canonical if deform is not active
        pred_dyn_logits = splats_dynamicness
    
    pred_dyn_scores = torch.sigmoid(pred_dyn_logits)  # [N]
    
    # Visibility mask for this batch
    if radii.ndim == 3 and radii.shape[-1] == 2: # Anisotropic Gaussians
        visible_mask = torch.sqrt((radii ** 2).sum(dim=-1)) > 0  # [B, N]
    elif radii.ndim == 2: # Isotropic Gaussians, batched
        visible_mask = radii > 0  # [B, N]
    else:  # Isotropic Gaussians, not batched (fallback)
        visible_mask = (radii > 0).expand(B, -1)  # [N] -> [B, N]

    # Loop over each item in the batch
    for b in range(B):
        # Get GT labels by sampling the mask at projected 2D centers
        visible_mask_b = visible_mask[b]
        u = means2d[b, :, 0].long().clamp(0, W - 1)
        v = means2d[b, :, 1].long().clamp(0, H - 1)
        gt_labels = dynamic_masks_gt[b, v, u].long()

        # Filter to only visible Gaussians for this view
        visible_preds = pred_dyn_scores[visible_mask_b]
        visible_gt = gt_labels[visible_mask_b]
        
        # Update and compute metrics
        if len(visible_preds) > 0:  # Ensure we have visible Gaussians
            # dyn_accuracy.update(visible_preds, visible_gt)
            # dyn_precision.update(visible_preds, visible_gt)
            # dyn_recall.update(visible_preds, visible_gt)
            dyn_f1.update(visible_preds, visible_gt)
            dyn_auroc.update(visible_preds, visible_gt)
        
    # Compute final values
    dyn_metrics = {
        # 'dyn_accuracy': dyn_accuracy.compute().item(),
        # 'dyn_precision': dyn_precision.compute().item(),
        # 'dyn_recall': dyn_recall.compute().item(),
        'dyn_f1': dyn_f1.compute().item(),
        'dyn_auroc': dyn_auroc.compute().item(),
    }
    return dyn_metrics


def batched_bilinear_sample(image, coords):
    """
    Args:
        image: [H, W, 3]
        coords: [N, 2] - (u, v) coordinates
    Returns:
        colors: [N, 3] - sampled colors
    """
    if coords.shape[0] == 0:
        return torch.zeros(0, 3, device=image.device)
    
    H, W = image.shape[:2]
    
    # Clamp coordinates to valid range
    u = torch.clamp(coords[:, 0], 0, W - 1)  # [N]
    v = torch.clamp(coords[:, 1], 0, H - 1)  # [N]
    
    # Get integer coordinates
    u0 = torch.floor(u).long()
    v0 = torch.floor(v).long()
    u1 = torch.clamp(u0 + 1, 0, W - 1)
    v1 = torch.clamp(v0 + 1, 0, H - 1)
    
    # Fractional parts for interpolation weights
    du = u - u0.float()  # [N]
    dv = v - v0.float()  # [N]
    
    # Bilinear interpolation weights
    w00 = (1 - du) * (1 - dv)  # [N]
    w01 = (1 - du) * dv        # [N] 
    w10 = du * (1 - dv)        # [N]
    w11 = du * dv              # [N]
    
    # Sample and interpolate colors
    colors = (image[v0, u0] * w00.unsqueeze(-1) +   # [N, 3]
              image[v0, u1] * w10.unsqueeze(-1) +   # [N, 3]
              image[v1, u0] * w01.unsqueeze(-1) +   # [N, 3]  
              image[v1, u1] * w11.unsqueeze(-1))    # [N, 3]
    
    return colors