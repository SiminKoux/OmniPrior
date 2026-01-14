import os
import cv2
import csv
import json
import torch
import tqdm
import argparse
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict

from torchmetrics.image import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure, 
    MultiScaleStructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def create_circular_mask(w, h):
    """Creates circular mask for fisheye images"""
    y, x = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    radius = np.sqrt((w/4)**2 + (h/2)**2)
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    mask = (dist_from_center <= radius).astype(np.float32)
    return torch.from_numpy(mask).cuda()

def masked_psnr(pred, target, mask):
    """Calculate PSNR only on masked regions"""
    # Mask prediction and target
    valid_mask = mask > 0.5  # [1,H,W]
    if valid_mask.sum() == 0:
        return 0.0
        
    # Calculate MSE only on valid pixels
    mse = torch.mean((pred[valid_mask] - target[valid_mask]) ** 2)
    return 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)

def masked_ssim(pred, target, mask, window_size=11, reduction='mean'):
    """
    Calculate SSIM only on masked regions using sliding window approach.
    
    Args:
        pred: [1, 3, H, W] predicted image
        target: [1, 3, H, W] target image
        mask: [H, W] binary mask (1 = dynamic, 0 = static)
        window_size: Size of Gaussian window (default: 11)
        reduction: 'mean' or 'sum' for aggregation
    
    Returns:
        SSIM value for masked region
    """
    import torch.nn.functional as F
    
    # Ensure inputs are on same device
    device = pred.device
    mask = mask.to(device)
    
    # Get original spatial dimensions
    _, _, H, W = pred.shape
    
    # Expand mask to match image channels [1, 1, H, W]
    mask_expanded = mask.unsqueeze(0).unsqueeze(0).float()
    
    # Create Gaussian window
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)
    
    window = gaussian_window(window_size)
    window = window.expand(3, 1, window_size, window_size)  # [3, 1, k, k]
    
    # SSIM constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Pad images and mask
    padding = window_size // 2
    pred_padded = F.pad(pred, [padding] * 4, mode='reflect')
    target_padded = F.pad(target, [padding] * 4, mode='reflect')
    mask_padded = F.pad(mask_expanded, [padding] * 4, mode='constant', value=0)
    
    # Calculate local statistics with Gaussian weighting
    mu_pred = F.conv2d(pred_padded, window, groups=3)
    mu_target = F.conv2d(target_padded, window, groups=3)
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    sigma_pred_sq = F.conv2d(pred_padded ** 2, window, groups=3) - mu_pred_sq
    sigma_target_sq = F.conv2d(target_padded ** 2, window, groups=3) - mu_target_sq
    sigma_pred_target = F.conv2d(pred_padded * target_padded, window, groups=3) - mu_pred_target
    
    # SSIM formula
    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    
    # Resize mask to match ssim_map dimensions [1, 3, H, W]
    # ssim_map has the same spatial size as input after padding/conv
    if ssim_map.shape[2] != H or ssim_map.shape[3] != W:
        mask_valid = F.interpolate(mask_expanded, size=(ssim_map.shape[2], ssim_map.shape[3]), 
                                   mode='nearest')
    else:
        mask_valid = mask_expanded
    
    # Expand to 3 channels [1, 3, H, W]
    mask_valid = mask_valid.expand(-1, 3, -1, -1)
    mask_valid = (mask_valid > 0.5).float()
    
    # Apply mask to ssim_map [1, 3, H, W]
    ssim_masked = ssim_map * mask_valid
    
    # Count valid pixels (where mask is True)
    valid_pixels = mask_valid.sum()
    
    if valid_pixels == 0:
        return 0.0
    
    # Average over valid masked regions
    if reduction == 'mean':
        return (ssim_masked.sum() / valid_pixels).item()
    else:
        return ssim_masked.sum().item()

def calculate_stats(values):
    values = [v.cpu.numpy() if torch.is_tensor(v) else v for v in values]
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'count': len(values)
    }

def save_stats_to_csv(overall_stats: dict, csv_path: str):
    """
    Save overall statistics to CSV format with metrics grouped by Train/Test columns.
    
    Format:
    psnr,psnr,ssim,ssim,...
    Train Views,Test Views,Train Views,Test Views,...
    34.877 ± 0.917,32.212 ± 2.041,0.889 ± 0.007,0.868 ± 0.015,...
    """
    # Define metric order
    metrics_order = ["psnr", "msssim", "lpips", "dyn_psnr", "dyn_ssim", "bbox_lpips"]
    
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
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header1)
        writer.writerow(header2)
        writer.writerow(values)
    
    print(f"CSV statistics saved to: {csv_path}")

def save_stats(train_metrics: dict, test_metrics: dict, output_dir: str = "./logs"):
    # Overall statistics (added dyn_ metrics)
    overall_stats = {
        split: {
            metric: calculate_stats([
                v for lens_metrics in metrics.values() 
                for m, values in lens_metrics.items() 
                if m == metric
                for v in values
            ])
            for metric in ["psnr", "ssim", "msssim", "lpips", "dyn_psnr", "dyn_ssim", "bbox_lpips"]
        }
        for split, metrics in [("train", train_metrics), ("test", test_metrics)]
    }
    # Save overall statistics (TXT)
    overall_path = os.path.join(output_dir, "overall_stats.txt")
    with open(overall_path, "w") as f:
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

class ResultEvaluator:
    def __init__(
        self,
        data_dir: str,
        result_dir: str,
        device: str = "cuda",
        w: int = 576,
        h: int = 768,
        factor: int = 1
    ):
        """
        Initialize the evaluator
        
        Args:
            render_dirs: List of paths to gt and rendered results
            device: Device to run evaluation on
        """
        self.data_dir = data_dir
        self.result_dir= result_dir
        self.device = device
        circular_mask = create_circular_mask(w, h)
        self.circular_mask = circular_mask
        self.w = w
        self.h = h
        self.factor = factor

        # Initialize metrics
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.msssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", 
            normalize=False
        ).to(device)

        # Load rig range masks
        self.rig_masks = self._load_rig_masks()

        # Load dynamic masks following dataset.py pattern
        self.dynamic_masks_dict = self._load_dynamic_masks()

    def _load_rig_masks(self) -> Dict[int, torch.Tensor]:
        """Load camera-specific range masks"""
        rig_masks = {}
        mask_dir = Path(os.path.join(self.data_dir, "robot_range"))
        if mask_dir.exists():
            mask_files = sorted(
                mask_dir.glob("lens*.npy"),
                key=lambda x: int(x.stem[4:])
            )
            for mask_file in mask_files:
                cam_id = int(mask_file.stem[4:6])
                rig_masks[cam_id] = torch.from_numpy(
                    np.load(mask_file)
                ).float().unsqueeze(0).to(self.device)
        return rig_masks
    
    def _load_dynamic_masks(self) -> Dict[str, Dict[int, Dict[str, torch.Tensor]]]:
        """
        Load dynamic masks following dataset.py pattern.
        Returns: dict with 'train' and 'test' keys, each containing 
                 {lens_id: {frame_name: mask_tensor}}
        """
        mask_dir = os.path.join(self.data_dir, "masks")
        if not os.path.exists(mask_dir):
            print(f"Warning: Mask directory {mask_dir} does not exist.")
            return {'train': {}, 'test': {}}
        
        print(f"Loading dynamic masks from {mask_dir}...")
        
        # Load mask npz files for each lens
        mask_npz = {}
        for lens_id in range(1, 7):  # lens01 to lens06
            mask_file = os.path.join(mask_dir, f"lens{lens_id:02d}.npz")
            if os.path.exists(mask_file):
                mask_npz[lens_id] = dict(np.load(mask_file))
                print(f"  Loaded lens{lens_id:02d}.npz with {len(mask_npz[lens_id])} frames")
        
        # Apply train/test split pattern (7 train + 3 test)
        dynamic_masks_split = {'train': {}, 'test': {}}
        pattern_length = 10
        train_length = 7
        
        for lens_id, masks in mask_npz.items():
            dynamic_masks_split['train'][lens_id] = {}
            dynamic_masks_split['test'][lens_id] = {}
            
            # Sort frame names to ensure consistent ordering
            frame_names = sorted(masks.keys())
            
            for frame_name in frame_names:
                # Extract frame number from frame_name (e.g., "frame_0001" -> 1)
                # Assumes format like "frame_0001", "frame_0010", etc.
                try:
                    frame_num = int(frame_name.split('_')[-1])
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse frame number from {frame_name}, skipping")
                    continue
                
                mask = masks[frame_name]
                
                # Resize if factor > 1
                if self.factor > 1:
                    mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
                
                # Convert to float tensor
                mask = (mask > 0).astype(np.float32)
                mask_tensor = torch.from_numpy(mask).float().to(self.device)
                
                # Apply train/test split pattern based on frame number
                # Frame numbers are 1-indexed, so adjust: frame 1 -> index 0
                frame_idx = frame_num - 1
                if frame_idx % pattern_length < train_length:
                    dynamic_masks_split['train'][lens_id][frame_name] = mask_tensor
                else:
                    dynamic_masks_split['test'][lens_id][frame_name] = mask_tensor
        
        # Print statistics
        for split in ['train', 'test']:
            total_masks = sum(len(masks) for masks in dynamic_masks_split[split].values())
            print(f"  {split}: {total_masks} dynamic masks across {len(dynamic_masks_split[split])} lenses")
        
        return dynamic_masks_split
        
    def evaluate_images(self, save_dir: Optional[str] = None):
        """Evaluate all methods and save results"""
        results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        # results = defaultdict(lambda: defaultdict(list))
        frame_metrics = defaultdict(dict)

        # Process both train and test splits
        for split in ['train', 'test']:
            split_dir = Path(os.path.join(result_dir, split))
            if not split_dir.exists():
                print(f"Skipping {split_dir} - directory not found")
                continue
            
            gt_dir = Path(os.path.join(split_dir, "gt_final"))
            render_dir = Path(os.path.join(split_dir, "renders_final"))
            
            if not (gt_dir.exists() and render_dir.exists()):
                print(f"Skipping {gt_dir} and {render_dir} - missing gt or rendered directory")
                continue
            
            # Process each lens directory
            for lens_id in range(1, 7):
                lens_dir = f"lens{lens_id:02d}"
                print(f"\nProcessing {lens_dir}...")

                # Get ground truth and rendered images from same lens directory
                lens_gt_dir = Path(os.path.join(gt_dir, lens_dir)) # GT subfolder
                lens_render_dir = Path(os.path.join(render_dir, lens_dir)) # Rendered subfolder

                lens_gt_files = sorted(lens_gt_dir.glob("*.png"))
                lens_render_files = sorted(lens_render_dir.glob("*.png"))

                if len(lens_gt_files) != len(lens_render_files):
                    print(f"Warning: Mismatch in number of images for {split}/{lens_dir}")
                    continue

                # Get dynamic masks for this lens and split
                dyn_masks_for_lens = self.dynamic_masks_dict[split].get(lens_id, {})

                # Process each image
                for gt_path, render_path in tqdm.tqdm(
                    zip(lens_gt_files, lens_render_files), 
                    total=len(lens_gt_files),
                    desc=f"Evaluating {lens_dir}/{split}"
                ):
                    # Extract frame name from path and convert to mask format
                    # GT/render: "frame_000001.png" -> mask format: "frame_0001"
                    frame_name_full = gt_path.stem  # "frame_000001"
                    try:
                        frame_num = int(frame_name_full.split('_')[-1])  # Extract number: 1
                        frame_name_mask_format = f"frame_{frame_num:04d}"  # Convert to "frame_0001"
                    except (ValueError, IndexError):
                        print(f"Warning: Could not parse frame number from {frame_name_full}")
                        frame_name_mask_format = frame_name_full

                    # Load and process images
                    lens_gt_img = imageio.imread(gt_path)[..., :3] / 255.0
                    lens_render_img = imageio.imread(render_path)[..., :3] / 255.0
                    
                    lens_gt_tensor = torch.from_numpy(lens_gt_img).float().unsqueeze(0).to(self.device)
                    lens_render_tensor = torch.from_numpy(lens_render_img).float().unsqueeze(0).to(self.device)

                    # Apply rig_masks if available
                    mask = self.rig_masks.get(lens_id, None)
                    
                    # Apply rig range mask and fisheye circular mask
                    combined_mask = None
                    if mask is not None:
                        # print("Rig maks avaliable!")
                        combined_mask = self.circular_mask * mask
                    else:
                        # print("Rig mask is unavaliable!")
                        combined_mask = self.circular_mask
                    # print("combined_mask:", combined_mask.sum())
                    
                    # Calculate metrics
                    gt_p = lens_gt_tensor.permute(0, 3, 1, 2)
                    render_p = lens_render_tensor.permute(0, 3, 1, 2)
                    
                    # Overall PSNR
                    if mask is not None:
                        psnr_val = float(masked_psnr(lens_render_tensor, lens_gt_tensor, combined_mask))
                    else:
                        psnr_val = self.psnr(render_p, gt_p).item()
                    
                    # Overall SSIM
                    if combined_mask is not None:
                        # Compute SSIM only on valid masked region
                        ssim_val = float(masked_ssim(render_p, gt_p, combined_mask.squeeze(0)))
                    else:
                        ssim_val = self.ssim(render_p, gt_p).item()

                    curr_metrics = {
                        "psnr": psnr_val,
                        "ssim": ssim_val,
                        "msssim": self.msssim(render_p, gt_p).item(),
                        "lpips": self.lpips(render_p, gt_p).item()
                    }

                    # -------- Dynamic-focused metrics (EXCLUDE robot_range) --------
                    dyn_roi = None
                    # Dynamic-focused metrics using loaded masks
                    dyn_mask = dyn_masks_for_lens.get(frame_name_mask_format, None)
                    if dyn_mask is not None:
                        # Apply combined mask to dynamic mask
                        dyn_roi = dyn_mask.unsqueeze(0)  # [1, H, W]
                        if combined_mask is not None:
                            dyn_roi = dyn_roi * combined_mask

                        valid_px = int(dyn_roi.sum().item())
                        if valid_px > 0:
                            # Extract bounding boxes from GT dynamic mask (rig-excluded)
                            dyn_mask_2d = dyn_roi.squeeze(0)  # [H, W]
                            bboxes = extract_dyn_bboxes(dyn_mask_2d, min_area=200, merge_close=True, bbox_dilat=8)

                            if len(bboxes) > 0:
                                dyn_psnr_vals, dyn_ssim_vals = [], []
                                bbox_lpips_vals = []
                                valid_dyn_areas = []  # track dynamic pixel areas
                                valid_bbox_areas = []
                                patch_save_dir = os.path.join(split_dir, "bbox", f"lens0{lens_id}")
                                patch_path = os.path.join(patch_save_dir, "patches")
                                bbox_path = os.path.join(patch_save_dir, "bboxes")
                                os.makedirs(patch_save_dir, exist_ok=True)
                                os.makedirs(patch_path, exist_ok=True)
                                os.makedirs(bbox_path, exist_ok=True)
                                # Save bbox visualization on the original image
                                gt_with_bboxes = lens_gt_tensor[0].cpu().numpy().copy()  # [H, W, 3]
                                # Draw all bboxes
                                for j, (bx1, by1, bx2, by2) in enumerate(bboxes):
                                    bx1c = max(0, bx1); by1c = max(0, by1)
                                    bx2c = min(self.w, bx2); by2c = min(self.h, by2)
                                    # Draw bbox rectangle (red border)
                                    # Top and bottom edges
                                    gt_with_bboxes[by1c:by1c+2, bx1c:bx2c, 0] = 1.0  # top edge
                                    gt_with_bboxes[by2c-2:by2c, bx1c:bx2c, 0] = 1.0  # bottom edge
                                    # Left and right edges
                                    gt_with_bboxes[by1c:by2c, bx1c:bx1c+2, 0] = 1.0  # left edge
                                    gt_with_bboxes[by1c:by2c, bx2c-2:bx2c, 0] = 1.0  # right edge
                                    # Add bbox number text (simple approach)
                                    text_y = max(10, by1c - 5)
                                    text_x = max(10, bx1c + 5)
                                    # Simple number overlay (you could use cv2.putText for better text)
                                    if text_y < self.h - 10 and text_x < self.w - 10:
                                        gt_with_bboxes[text_y:text_y+8, text_x:text_x+8, :] = [1.0, 1.0, 0.0]  # yellow square as number indicator
                                # Save bbox overlay
                                bbox_overlay_path = os.path.join(bbox_path, f"{gt_path.stem}.png")
                                imageio.imwrite(bbox_overlay_path, (np.clip(gt_with_bboxes, 0, 1) * 255).astype(np.uint8))
                                for i, (x1, y1, x2, y2) in enumerate(bboxes):
                                    # Clamp (safety)
                                    x1c = max(0, x1); y1c = max(0, y1)
                                    x2c = min(self.w, x2); y2c = min(self.h, y2)
                                    if x2c - x1c <= 180 and y2c - y1c <= 180:
                                        continue  # too small for stable metrics
                                    
                                    # Extract patches [1, 3, h, w]
                                    gt_patch = lens_gt_tensor[0, y1c:y2c, x1c:x2c].permute(2, 0, 1).unsqueeze(0)
                                    pred_patch = lens_render_tensor[0, y1c:y2c, x1c:x2c].permute(2, 0, 1).unsqueeze(0)
                                    
                                    # Extract dynamic mask for this bbox region [H, W]
                                    bbox_dyn_mask = dyn_mask_2d[y1c:y2c, x1c:x2c]  # [h, w]
                                    dyn_pixel_count = int(bbox_dyn_mask.sum().item())
                                    
                                    # Save original patches
                                    # Convert to numpy for saving [H, W, 3]
                                    gt_patch_np = gt_patch[0].permute(1, 2, 0).cpu().numpy()
                                    pred_patch_np = pred_patch[0].permute(1, 2, 0).cpu().numpy()
                                    
                                    # Create masked versions (black out static pixels)
                                    # Mask is [h, w], expand to [h, w, 3] for RGB
                                    bbox_dyn_mask_3ch = bbox_dyn_mask.unsqueeze(-1).cpu().numpy()  # [h, w, 1]
                                    gt_patch_masked_np = gt_patch_np * bbox_dyn_mask_3ch  # Black out static
                                    pred_patch_masked_np = pred_patch_np * bbox_dyn_mask_3ch  # Black out static
                                    
                                    # Create side-by-side comparison with gap
                                    gap_width = 4  # pixels of gap between patches
                                    gap_color = [1.0, 1.0, 1.0]  # white gap
                                    
                                    patch_h, patch_w = gt_patch_np.shape[:2]
                                    gap_strip = np.full((patch_h, gap_width, 3), gap_color, dtype=gt_patch_np.dtype)
                                    
                                    # Concatenate: GT + gap + rendered
                                    patch_comparison = np.concatenate([gt_patch_np, gap_strip, pred_patch_np], axis=1)
                                    
                                    # Concatenate masked versions: GT_masked + gap + rendered_masked
                                    patch_comparison_masked = np.concatenate([gt_patch_masked_np, gap_strip, pred_patch_masked_np], axis=1)
                                    
                                    # Save individual patches and comparison
                                    base_name = f"{gt_path.stem}_bbox{(i+1):02d}"
                                    
                                    # Save side-by-side comparison (full bbox)
                                    comp_patch_path = os.path.join(patch_path, f"{base_name}.png")
                                    imageio.imwrite(comp_patch_path, (np.clip(patch_comparison, 0, 1) * 255).astype(np.uint8))
                                    
                                    # Save masked version (dynamic pixels only)
                                    comp_patch_masked_path = os.path.join(patch_path, f"{base_name}_dynonly.png")
                                    imageio.imwrite(comp_patch_masked_path, (np.clip(patch_comparison_masked, 0, 1) * 255).astype(np.uint8))

                                    try:
                                        # Bbox metrics (full rectangle including static background)
                                        lp = self.lpips(pred_patch, gt_patch).item()
                                        
                                        # Dynamic-only metrics (only on dynamic pixels)
                                        if dyn_pixel_count > 100:  # Require at least 100 dynamic pixels
                                            # Get flattened GT and pred values for dynamic pixels only
                                            # bbox_dyn_mask is [h, w], expand to [h, w, 3]
                                            dyn_mask_bool = bbox_dyn_mask > 0.5  # [h, w] boolean
                                            
                                            # Extract only dynamic pixels from patches [1, 3, h, w]
                                            gt_patch_hwc = gt_patch[0].permute(1, 2, 0)  # [h, w, 3]
                                            pred_patch_hwc = pred_patch[0].permute(1, 2, 0)  # [h, w, 3]
                                            
                                            # Apply mask to each channel
                                            gt_dyn_pixels = gt_patch_hwc[dyn_mask_bool]      # [N_dyn, 3]
                                            pred_dyn_pixels = pred_patch_hwc[dyn_mask_bool]  # [N_dyn, 3]
                                            
                                            # Compute PSNR only on dynamic pixels
                                            mse_dyn = torch.mean((pred_dyn_pixels - gt_dyn_pixels) ** 2)
                                            if mse_dyn > 0:
                                                dyn_ps = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse_dyn)
                                                dyn_ps = dyn_ps.item()
                                            else:
                                                dyn_ps = 100.0  # Perfect match
                                            
                                            # For SSIM, compute on dynamic pixels only using masked SSIM
                                            # This computes SSIM on regions where the mask is True
                                            dyn_sm = masked_ssim(pred_patch, gt_patch, bbox_dyn_mask)
                                            
                                            dyn_psnr_vals.append(dyn_ps)
                                            dyn_ssim_vals.append(dyn_sm)
                                            valid_dyn_areas.append(float(dyn_pixel_count))
                                        
                                    except Exception as e:
                                        print(f"[dyn bbox warn] metric fail bbox {i}: {e}")
                                        continue
                                    
                                    area = float((x2c - x1c) * (y2c - y1c))
                                    bbox_lpips_vals.append(lp)
                                    valid_bbox_areas.append(area)
                                
                                if valid_bbox_areas:
                                    total_area = sum(valid_bbox_areas)
                                    w_areas = [a / total_area for a in valid_bbox_areas]
                                    curr_metrics["bbox_lpips"] = sum(l * w for l, w in zip(bbox_lpips_vals, w_areas))
                                
                                # Aggregate dynamic-only metrics (weighted by dynamic pixel count)
                                if valid_dyn_areas:
                                    total_dyn_area = sum(valid_dyn_areas)
                                    w_dyn_areas = [a / total_dyn_area for a in valid_dyn_areas]
                                    curr_metrics["dyn_psnr"] = sum(p * w for p, w in zip(dyn_psnr_vals, w_dyn_areas))
                                    curr_metrics["dyn_ssim"] = sum(s * w for s, w in zip(dyn_ssim_vals, w_dyn_areas))

                    # Store metrics
                    for metric, value in curr_metrics.items():
                        results[split][lens_id][metric].append(value)
                    
                    frame_metrics[f"{split}_{lens_dir}_{gt_path.stem}"] = curr_metrics

        if save_dir:
            self._save_results(results, frame_metrics, save_dir)
        
        return results, frame_metrics

    def _save_results(self, results: Dict, frame_metrics: Dict, save_dir: str):
        """Save evaluation results"""
        os.makedirs(save_dir, exist_ok=True)
        print("saving statistics into:", save_dir)
        # print(results["test"])

        # Save per-frame metrics
        with open(os.path.join(save_dir, "frame_metrics.json"), "w") as f:
            json.dump({"per_frame": frame_metrics}, f, indent=2)

        # Calculate and save summarized results
        all_results = {
            "overall": defaultdict(dict),
            "per_lens": defaultdict(lambda: defaultdict(dict))
        }

        train_metrics = results["train"]
        test_metrics = results["test"]
        save_stats(train_metrics=train_metrics, test_metrics=test_metrics, output_dir=save_dir)

        for split in ['train', 'test']:
            # Calculate method-wide averages per split
            method_metrics = defaultdict(list)
            for lens_metrics in results[split].values():
                for metric, values in lens_metrics.items():
                    method_metrics[metric].extend(values)
            
            all_results["overall"][split] = {
                metric: float(np.mean(values))
                for metric, values in method_metrics.items()
            }

            # Calculate per-lens averages
            for lens_id, lens_metrics in results[split].items():
                all_results["per_lens"][split][f"lens{lens_id:02d}"] = {
                    metric: float(np.mean(values))
                    for metric, values in lens_metrics.items()
                }

        # Save summarized results
        with open(os.path.join(save_dir, "summarized_metrics.json"), "w") as f:
            json.dump(all_results, f, indent=2)

        # Print summary
        print("\nOverall Results:")
        for split in ['train', 'test']:
            metrics = all_results["overall"][split]
            print(f"{split} - ", " ".join(
                f"{k}: {v:.4f}" for k, v in metrics.items()
            ))

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate multi-fisheye rendering results')
    parser.add_argument('--scene', type=str, default='suite', help='Scene name to evaluate')
    args = parser.parse_args()

    data_root_dir = './data/OmniFisheye_plus'
    result_root_dir = './results'

    scene_name = args.scene

    data_dir = os.path.join(data_root_dir, scene_name)
    result_path = os.path.join(result_root_dir, scene_name)
    result_root = os.path.join(result_path, "init_colmap_metric")
    result_dir = os.path.join(result_root, "renders")
    output_dir = f"{result_root}/stats/multi-fisheye/omniprior"

    if scene_name == "lounge" or scene_name == "hall":
        print(f"Evaluate for scene {scene_name}")
        width = 720
        height = 960
    else:
        width = 576
        height = 768
    
    # Setup evaluator
    evaluator = ResultEvaluator(
        data_dir=data_dir,
        result_dir=result_dir,
        device="cuda",
        w = width,
        h = height
    )
    
    # Run evaluation
    results, frame_metrics = evaluator.evaluate_images(save_dir=output_dir)