import os
import json
import pickle
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio

import torch
import numpy as np
from pycolmap import SceneManager

from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
from scipy.ndimage import binary_erosion

def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths

class Parser:
    """COLMAP parser."""
    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        init_type: str = "metric",
        filter: bool = False
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize

        # COLMAP directory selection based on init_type
        # - "metric": Use metric-rescaled COLMAP output (sparse/rescaled)
        # - "original": Use original COLMAP output (sparse/0)
        if init_type == "metric":
            # Use rescaled COLMAP reconstruction (metric scale)
            colmap_dir = os.path.join(data_dir, "sparse/rescaled/")
            if not os.path.exists(colmap_dir):
                print(f"Warning: Rescaled COLMAP directory {colmap_dir} not found.")
                print("         Falling back to original COLMAP (sparse/0).")
                colmap_dir = os.path.join(data_dir, "sparse/0/")
        elif init_type == "original" or init_type == "random":
            # Use original COLMAP reconstruction (arbitrary scale)
            colmap_dir = os.path.join(data_dir, "sparse/0/")
        metric_depth_dir = os.path.join(data_dir, "metric_depths")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(metric_depth_dir), f"Metric depth directory {mask_dir} does not exist."
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."
        print(f"Using COLMAP directory: {colmap_dir} (init_type={init_type})")
        
        # Mask directory.
        mask_dir = os.path.join(data_dir, "masks")
        assert os.path.exists(mask_dir), f"Mask directory {mask_dir} does not exist."

        # Monocular depth directory.
        mono_depth_dir = os.path.join(data_dir, "mono_depths")
        assert os.path.exists(mono_depth_dir), f"Monocular depth directory {mask_dir} does not exist."

        # Sparse matches path (optional).
        sparse_matches_path = os.path.join(data_dir, "sparse_matches.pkl")
        if not os.path.exists(sparse_matches_path):
            sparse_matches_path = None
    
        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        dynamic_mask_list = []
        depth_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()       # width, height
        mask_npz = dict()          # Cache for dynamic binary masks
        mono_depth_dict = dict()   # Cache for monocular depths
        metric_depth_dict = dict() # Cache for metric depths
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([-cam.k1, -cam.k2, -cam.k3, -cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            
            # Load mask if available.
            if camera_id not in mask_npz:
                mask_file = os.path.join(mask_dir, f"lens{camera_id:02d}.npz")
                mask_npz[camera_id] = dict(np.load(mask_file))
            # Load monocular depth if available.
            if camera_id not in mono_depth_dict:
                depth_file = os.path.join(mono_depth_dir, f"lens{camera_id:02d}.pt")
                depth_data = torch.load(depth_file)
                # Store both data and mapping
                mono_depth_dict[camera_id] = {
                    'depth': depth_data['depth'],  # [num_frames, H, W] tensor, raw depth
                    'valid_min': float(depth_data['valid_min']),   # minimum valid depth
                    'valid_max': float(depth_data['valid_max']),   # maximum valid depth
                    'valid_mean': float(depth_data['valid_mean']), # mean valid depth
                }
            # Load metric depth if available.
            if camera_id not in metric_depth_dict:
                if os.path.exists(metric_depth_dir):
                    metric_depth_file = os.path.join(metric_depth_dir, f"lens{camera_id:02d}.npz")
                    if os.path.exists(metric_depth_file):
                        metric_data = np.load(metric_depth_file)
                        if 'depths' in metric_data:
                            metric_depth_dict[camera_id] = metric_data['depths']  # [N_frames, H, W] in meters
            frame_name = os.path.splitext(os.path.basename(im.name))[0]
            dynamic_mask = mask_npz[camera_id][frame_name]
            timestamp_idx = int(frame_name.split('_')[1]) - 1  # frame_0001 -> 0, frame_0002 -> 1
            if factor > 1:
                dynamic_mask = cv2.resize(dynamic_mask, (cam.width, cam.height), interpolation=cv2.INTER_NEAREST)
            dynamic_mask = (dynamic_mask > 0).astype(np.float32)
            dynamic_mask_list.append(dynamic_mask)
            depth_ids.append(timestamp_idx)

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")

        w2c_mats = np.stack(w2c_mats, axis=0)
        dynamic_masks = np.stack(dynamic_mask_list, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. 
        # No need for permuting the poses according to image names anymore.
        image_names = [imdata[k].name for k in imdata]
        inds = np.argsort(image_names)  # [0 1 2 ... (C*num_frames-1]
        image_names = [image_names[i] for i in inds]  # ['lens01/frame_0001.png', 'lens01/frame_0002.png', ...]
        camtoworlds = camtoworlds[inds]
        dynamic_masks = dynamic_masks[inds]
        camera_ids = [camera_ids[i] for i in inds] # [1, 1, ..., 1, 2, 2, ..., 2, ... 6, 6, ..., 6]
        depth_ids = [depth_ids[i] for i in inds]   # [0, 1, 2, ..., num_frames-1, 0, 1, 2, ..., num_frames-1, ...]

        print("-" * 10 + f" [Parser] " + "-" * 10)
        print(f"{len(imdata)} images, taken by {len(set(camera_ids))} lenses.")
        print(f"{len(mask_npz)} mask npz files.")
        print(f"{len(mono_depth_dict)} monocular depth files.")
        if len(metric_depth_dict) > 0:
            print(f"{len(metric_depth_dict)} metric depth files.")

        # Create a shared mask for each lens captured fisheye images
        valid_mask, eroded_mask = self._create_circular_mask(
            imsize_tuple=imsize_dict[camera_ids[0]],
            edge_margin=20
        ) # np.ndarray, (height, width), float32, 1.0 for valid region, 0.0 for invalid region
        self.shared_mask = valid_mask   # Use this for depth normalization and rendering masking
        self.eroded_mask = eroded_mask  # Use this for edge-filtered sampling

        self._normalize_mono_depths(
            mono_depth_dict=mono_depth_dict,
            camera_ids=camera_ids,
            mono_depth_indices=depth_ids,
            factor=factor
        )
        
        # Load times if possible (used in deformation module).
        metadata_file = os.path.join(data_dir, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file) as f:
                metadata = json.load(f)
            all_time = [metadata[i]['time_id'] for i in image_names]
            max_time = max(all_time)
            self.max_time_id = max_time
            self.times = [metadata[i]['time_id'] / max_time for i in image_names]
            self.time_ids = [metadata[i]['time_id'] for i in image_names]  # [1, 2, 3, ...]
            self.selected_times = set(self.times)
            print(f"{len(self.selected_times)} time ids.")

        # Load images.
        if factor > 1:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        points_err = manager.point3D_errors.astype(np.float32)
        
        point_indices = dict()
        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principal_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1

            # Fix for up side down. 
            # We assume more points towards the bottom of the scene 
            # which is true when ground floor is present in the images.
            if np.median(points[:, 2]) > np.mean(points[:, 2]):
                # rotate 180 degrees around x axis such that z is flipped
                T3 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                camtoworlds = transform_cameras(T3, camtoworlds)
                points = transform_points(T3, points)
                transform = T3 @ transform
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids  = camera_ids   # List[int], (num_images,)
        self.Ks_dict     = Ks_dict      # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.dynamic_masks = dynamic_masks  # np.ndarray, (num_images, height, width)
        self.mask_npz = mask_npz        # Dict of camera_id -> {frame_name: mask}
        self._assign_soft_masks()       # Assign soft masks for each lens
        
        if filter:
            # Store original points for reference
            self.points_original = points                # np.ndarray, (num_points, 3) - original unfiltered
            self.points_err_original = points_err        # np.ndarray, (num_points,) - original unfiltered
            self.points_rgb_original = points_rgb        # np.ndarray, (num_points, 3) - original unfiltered
            self.point_indices_original = point_indices  # Dict[str, np.ndarray] - original mapping
            self._filter_points_for_init()
        else:
            self.points = points                # np.ndarray, (num_points, 3)
            self.points_err = points_err        # np.ndarray, (num_points,)
            self.points_rgb = points_rgb        # np.ndarray, (num_points, 3)
            self.point_indices = point_indices  # Dict[str, np.ndarray]

        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))
        
        # Compute scene scale with outlier filtering
        self._compute_scene_scale()

        self._rig_center()

        self.sparse_matches = None
        if sparse_matches_path is not None:
            self._load_sparse_matches(sparse_matches_path)
        
        # Process metric depths if loaded
        if len(metric_depth_dict) > 0:
            self._process_metric_depths(metric_depth_dict, camera_ids, depth_ids, factor)
        else:
            self.metric_depths = None
    
    def _assign_soft_masks(self):
        self.soft_masks = {}
        for camera_id, mask_dict in self.mask_npz.items():
            all_masks = []
            for frame_name, mask in mask_dict.items():
                # flatten mask into 1D probability array
                mask = mask.astype(np.float32)
                # Normalize to [0, 1] range
                if mask.max() > 1.0:
                    mask = mask / 255.0
                all_masks.append(mask)
            # mean probability map over all frames
            soft_mask = np.mean(all_masks, axis=0)
            self.soft_masks[camera_id] = soft_mask

    def _create_circular_mask(self, imsize_tuple, edge_margin=10):
        """
        Creates circular mask for cutomized fisheye images with optional erosion.
        Args:
            imsize_tuple: (width, height)
            edge_margin: Pixels to erode from edges (0 = no erosion)
        
        Returns: 
            mask: torch.Tensor of shape (H, W) with dtype=float32
            eroded_mask: torch.Tensor of shape (H, W) for edge-filtered sampling
        """
        # Get image dimensions
        w, h = imsize_tuple
        
        # Create coordinate grid efficiently
        y, x = np.ogrid[:h, :w]
        
        # Center coordinates
        cx, cy = w // 2, h // 2
        
        # Calculate radius to include entire fisheye circle
        # Using diagonal distance from center to corner
        radius = np.sqrt((w/4)**2 + (h/2)**2)
        
        # Compute distances (vectorized)
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Create binary mask as np.ndarray of dtype=float32
        mask = (dist_from_center <= radius).astype(np.float32)

        # Create eroded mask for edge-filtered sampling
        if edge_margin > 0:
            # Morphological erosion (one-time cost)
            structure = np.ones((edge_margin * 2 + 1, edge_margin * 2 + 1))
            eroded = binary_erosion(mask, structure=structure, iterations=1)
            eroded_mask = eroded.astype(np.float32)
        else:
            eroded_mask = mask.copy()

        return mask, eroded_mask

    def _normalize_mono_depths(
        self,
        mono_depth_dict: Dict,
        camera_ids: List[int],
        mono_depth_indices: List[int],
        factor: int
    ):
        """
        Normalize monocular depths to [0, 1] for ranking-based supervision.
        Uses pre-computed global min/max per camera.
        """
        print("\n" + "="*50)
        print("Normalizing Monocular Depths (Per-Lens)")
        print("="*50)

        # Get shared fisheye mask
        mask = self.shared_mask  # [H, W] bool or float
        
        # Storage for normalized depths
        normalized_depths = []
        camera_depth_ranges = {}
        
        # Process each camera's depth stack
        unique_cameras = sorted(set(camera_ids))
        
        for camera_id in unique_cameras:
            print(f"\nProcessing Lens {camera_id}:")
            
            # Get raw depth stack [T, H, W]
            depth_stack = mono_depth_dict[camera_id]['depth'].numpy()
            
            # Use PRE-COMPUTED global min/max for this lens
            mono_min_global = mono_depth_dict[camera_id]['valid_min']
            mono_max_global = mono_depth_dict[camera_id]['valid_max']
            
            T, H, W = depth_stack.shape
            
            # Resize if needed
            if factor > 1:
                target_h = H // factor
                target_w = W // factor
                depth_stack_resized = np.zeros((T, target_h, target_w), dtype=np.float32)
                for t in range(T):
                    depth_stack_resized[t] = cv2.resize(
                        depth_stack[t], 
                        (target_w, target_h), 
                        interpolation=cv2.INTER_LINEAR
                    )
                depth_stack = depth_stack_resized
                print(f"  Resized from [{H}, {W}] to [{target_h}, {target_w}]")
            
            # Resize mask to match depth dimensions
            if mask.shape != (H, W):
                mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
                valid_mask = mask_resized > 0.5
            else:
                valid_mask = mask > 0.5
            
            # Count valid pixels
            num_valid = valid_mask.sum()
            num_total = H * W
            print(f"  Valid pixels: {num_valid:,} / {num_total:,} ({100*num_valid/num_total:.1f}%)")

            # Verify we have valid depth range
            if mono_max_global - mono_min_global < 1e-6:
                print(f"  WARNING: Constant depth for lens {camera_id}!")
                depth_stack[:] = 0.5
            else:
                # Normalize to [0, 1] using GLOBAL min/max
                # This preserves ranking across ALL timestamps for this lens
                depth_normalized = (depth_stack - mono_min_global) / (mono_max_global - mono_min_global)
                depth_stack = depth_normalized  # [T, H, W] in [0, 1]
            
            # Report statistics
            valid_depths_only = depth_stack[:, valid_mask]
            invalid_depths_only = depth_stack[:, ~valid_mask]
            print(f"  Original range: [{mono_min_global:.3f}, {mono_max_global:.3f}]")
            print(f"  After normalization:")
            print(f"    All pixels: [{depth_stack.min():.3f}, {depth_stack.max():.3f}]")
            print(f"    Valid region: [{valid_depths_only.min():.3f}, {valid_depths_only.max():.3f}]")
            if invalid_depths_only.size > 0:
                print(f"    Invalid region: [{invalid_depths_only.min():.3f}, {invalid_depths_only.max():.3f}] (will be masked out)")
 
            # Store original range for reference (useful for visualization/debugging)
            camera_depth_ranges[camera_id] = (mono_min_global, mono_max_global)
            
            # Store normalized depth stack
            mono_depth_dict[camera_id]['norm_depth'] = depth_stack  # [T, H, W] in [0, 1]
        
        # Construct final sorted depth array [N_images, H, W]
        print(f"\nConstructing final depth array...")
        for idx, (camera_id, timestamp_idx) in enumerate(zip(camera_ids, mono_depth_indices)):
            normalized_depth_frame = mono_depth_dict[camera_id]['norm_depth'][timestamp_idx]
            normalized_depths.append(normalized_depth_frame)
        
        self.mono_depths = np.stack(normalized_depths, axis=0)  # [N_images, H, W] in [0, 1]
        self.camera_depth_ranges = camera_depth_ranges          # Store for reference only
        
        print(f"\n  Depth normalization complete!")
        print(f"  Final depth array shape: {self.mono_depths.shape}")
        print(f"  Global range: [{self.mono_depths.min():.3f}, {self.mono_depths.max():.3f}]")
        print(f"  Note: Out-of-range values in invalid regions will be masked during training")
        print(f"  Original ranges (for reference):")
        for cam_id, (d_min, d_max) in sorted(camera_depth_ranges.items()):
            print(f"    Lens {cam_id}: [{d_min:.3f}, {d_max:.3f}]")
        print("="*50 + "\n")
    
    def _load_sparse_matches(self, matches_path: str):
        """
        Load sparse matches from .pkl file.
        
        Expected structure:
        {
            frame_idx (int): {  # e.g., 1, 2, 3, ... (corresponds to time_id)
                '0->1': {
                    'pts_i': torch.Tensor [M, 2],  # (y, x) format
                    'pts_j': torch.Tensor [M, 2],  # (y, x) format
                    'confidence': torch.Tensor [M]
                },
                '1->2': {...},
                ...
            },
            ...
        }
        """
        if not os.path.exists(matches_path):
            print(f"Warning: Sparse matches file not found: {matches_path}")
            return
        
        try:
            with open(matches_path, 'rb') as f:
                self.sparse_matches = pickle.load(f)
            
            print(f"\n{'='*50}")
            print(f"Loaded Sparse Matches")
            print(f"{'='*50}")
            print(f"  Path: {matches_path}")
            print(f"  Frames with matches: {len(self.sparse_matches)}")
            
            # Verify structure and print statistics
            if len(self.sparse_matches) > 0:
                # Get a sample frame
                sample_frame_idx = list(self.sparse_matches.keys())[0]
                sample_frame = self.sparse_matches[sample_frame_idx]
                
                print(f"  Lens pairs per frame: {len(sample_frame)}")
                print(f"  Example frame {sample_frame_idx}:")
                
                total_matches = 0
                all_confidences = []
                for pair_key, match_data in sample_frame.items():
                    n_matches = len(match_data['pts_i'])
                    avg_conf = match_data['confidence'].mean().item()
                    total_matches += n_matches
                    all_confidences.extend(match_data['confidence'].tolist())
                    print(f"    {pair_key}: {n_matches} matches (avg_conf={avg_conf:.3f})")
                
                print(f"  Total matches per frame: {total_matches}")
                print(f"  Global avg confidence: {np.mean(all_confidences):.3f}")
                print(f"{'='*50}\n")
                
        except Exception as e:
            print(f"Error loading sparse matches: {e}")
            self.sparse_matches = None
    
    def _process_metric_depths(
        self,
        metric_depth_dict: Dict[int, np.ndarray],
        camera_ids: List[int],
        depth_indices: List[int],
        factor: int = 1
    ):
        """
        Process pre-loaded metric depths from cached dictionary.
        
        This method is called after metric depth files have been loaded during
        the initialization loop (following the same pattern as mask_npz and depth_dict).
        
        Args:
            metric_depth_dict: Dict mapping camera_id -> depth_stack [N_frames, H, W]
            camera_ids: List of camera IDs for all images (1-indexed)
            depth_indices: List of frame indices for all images (0-indexed)
            factor: Downsampling factor
        """
        print(f"\n{'='*50}")
        print(f"Processing Metric Depths")
        print(f"{'='*50}")
        
        # Resize if needed and store processed depths
        processed_dict = {}
        
        for camera_id, depth_stack in metric_depth_dict.items():
            N, H, W = depth_stack.shape
            
            print(f"  Lens {camera_id:02d}: shape={depth_stack.shape}")
            
            # Resize if needed
            if factor > 1:
                target_h = H // factor
                target_w = W // factor
                depth_stack_resized = np.zeros((N, target_h, target_w), dtype=np.float32)
                for t in range(N):
                    depth_stack_resized[t] = cv2.resize(
                        depth_stack[t], 
                        (target_w, target_h), 
                        interpolation=cv2.INTER_LINEAR
                    )
                depth_stack = depth_stack_resized
                print(f"    Resized from [{H}, {W}] to [{target_h}, {target_w}]")
            
            # Store processed depths
            processed_dict[camera_id] = depth_stack
        
        # Construct final array [N_images, H, W] following the order in camera_ids
        metric_depths_list = []
        for camera_id, frame_idx in zip(camera_ids, depth_indices):
            metric_depth_frame = processed_dict[camera_id][frame_idx]  # [H, W]
            metric_depths_list.append(metric_depth_frame)
        
        self.metric_depths = np.stack(metric_depths_list, axis=0)  # [N_images, H, W]
        
        # Report statistics (excluding NaN values)
        valid_depths = self.metric_depths[~np.isnan(self.metric_depths)]
        
        print(f"\n  Metric depths loaded successfully!")
        print(f"  Shape: {self.metric_depths.shape}")
        print(f"  Depth range: [{valid_depths.min():.2f}m, {valid_depths.max():.2f}m]")
        print(f"  Mean depth: {valid_depths.mean():.2f}m ± {valid_depths.std():.2f}m")
        print(f"  Valid coverage: {100*len(valid_depths)/self.metric_depths.size:.1f}%")
        print(f"{'='*50}\n")

    def _rig_center(self):
        unique_cameras = list(set(self.camera_ids))
        if len(unique_cameras) < 2:
            print("Warning: Less than 2 lenses found")
            return
        
        # Group camera poses by time_id
        camera_poses_by_time = {}
        
        # for i, (camtoworld, camera_id, time) in enumerate(zip(self.camtoworlds, self.camera_ids, self.time_ids)): # for colmap_dense_pcd
        for i, (camtoworld, camera_id, time) in enumerate(zip(self.camtoworlds, self.camera_ids, self.times)):
            if time not in camera_poses_by_time:
                camera_poses_by_time[time] = {}
            camera_poses_by_time[time][camera_id] = camtoworld
        
        # Compute rig center for each timestamp
        self.rig_centers = {}
        
        for time in sorted(camera_poses_by_time.keys()):
            # Check if we have all cameras at this timestamp
            if len(camera_poses_by_time[time]) != len(unique_cameras):
                print(f"Warning: Time {time} has {len(camera_poses_by_time[time])} lenses, expected {len(unique_cameras)}")
                continue
            
            # Extract positions from all cameras at this timestamp
            positions = []
            for cam_id in sorted(camera_poses_by_time[time].keys()):
                positions.append(camera_poses_by_time[time][cam_id][:3, 3])
            positions = np.array(positions)  # [N_cameras, 3]
            
            # Compute rig center as mean of all camera positions
            self.rig_centers[time] = np.mean(positions, axis=0)  # [3]
    
        print(f"Computed rig centers for {len(self.rig_centers)} timestamps")

    def _compute_scene_scale(self):
        """
        Compute scene scale using the 3D points.
        This method is called after point filtering, so self.points contains the robust points.
        """
        points = self.points  # Use already points
        print(f"Computing scene scale from {len(points)} points...")
        
        # Calculate distances from origin (already filtered points)
        distances = np.linalg.norm(points, axis=1)
        
        # Compute scene statistics from filtered points
        points_extent_robust = np.max(distances)
        
        # Also compute axis-wise extents for better understanding
        xyz_min = points.min(axis=0)
        xyz_max = points.max(axis=0)
        xyz_range = xyz_max - xyz_min
        max_axis_range = np.max(xyz_range)
        
        # Use the more conservative estimate between max distance and max axis range
        scene_scale_distance = points_extent_robust
        scene_scale_axis = max_axis_range
        
        # Take the larger of the two measures, with a minimum threshold
        self.scene_scale = max(scene_scale_distance, scene_scale_axis, 10.0)
        
        # Store additional statistics for potential use
        self.scene_stats = {
            'total_points': len(points),
            'xyz_min': xyz_min,
            'xyz_max': xyz_max,
            'xyz_range': xyz_range,
            'max_distance': points_extent_robust,
            'max_axis_range': max_axis_range,
            'mean_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'std_distance': np.std(distances)
        }
        
        print(f"  XYZ range: X[{xyz_min[0]:.3f}, {xyz_max[0]:.3f}], Y[{xyz_min[1]:.3f}, {xyz_max[1]:.3f}], Z[{xyz_min[2]:.3f}, {xyz_max[2]:.3f}]")
        print(f"  Max axis range: {max_axis_range:.3f}")
        print(f"  Max distance from center: {points_extent_robust:.3f}")
        print(f"  Final scene scale: {self.scene_scale:.3f}")
    
    def _filter_points_for_init(self, percentile=95, error_threshold=None):
        """
        Filter points for robust Gaussian initialization 
        while maintaining point_indices consistency.
        
        Args:
            percentile: percentile threshold for distance-based filtering (default: 95)
            error_threshold: error threshold for filtering (default: None, no filtering)
        """ 
        # Start with original points
        points = self.points_original.copy()
        points_err = self.points_err_original.copy()
        points_rgb = self.points_rgb_original.copy()
        
        # Create filtering mask
        valid_mask = np.ones(len(points), dtype=bool)
        
        # Filter by reprojection error first
        if error_threshold is not None:
            error_mask = points_err <= error_threshold
            valid_mask &= error_mask
           
        # Filter by distance percentile
        if percentile is not None:
            distances = np.linalg.norm(points, axis=1)
            threshold = np.percentile(distances, percentile)
            distance_mask = distances <= threshold
            valid_mask &= distance_mask
            
        # Apply filtering
        filtered_points = points[valid_mask]
        filtered_points_err = points_err[valid_mask]
        filtered_points_rgb = points_rgb[valid_mask]
        
        # Create mapping from old indices to new indices
        old_to_new_idx = {}
        new_idx = 0
        for old_idx, is_valid in enumerate(valid_mask):
            if is_valid:
                old_to_new_idx[old_idx] = new_idx
                new_idx += 1
        
        # Update point_indices to use new indices, filtering out invalid points
        filtered_point_indices = {}
        total_original_refs = 0
        total_filtered_refs = 0
        
        for image_name, old_indices in self.point_indices_original.items():
            total_original_refs += len(old_indices)
            new_indices = []
            for old_idx in old_indices:
                if old_idx in old_to_new_idx:
                    new_indices.append(old_to_new_idx[old_idx])
            
            if len(new_indices) > 0:  # Only keep images that still have valid points
                filtered_point_indices[image_name] = np.array(new_indices, dtype=np.int32)
                total_filtered_refs += len(new_indices)
        
        # Store filtered data as the main data for Gaussian initialization
        self.points = filtered_points          # np.ndarray, (filtered_num_points, 3)
        self.points_err = filtered_points_err  # np.ndarray, (filtered_num_points,)
        self.points_rgb = filtered_points_rgb  # np.ndarray, (filtered_num_points, 3)
        self.point_indices = filtered_point_indices  # Dict[str, np.ndarray], image_name -> [filtered_M,]
        
        # Store filtering statistics
        self.filtering_stats = {
            'original_points': len(self.points_original),
            'filtered_points': len(self.points),
            'filtering_ratio': len(self.points) / len(self.points_original),
            'original_image_point_refs': total_original_refs,
            'filtered_image_point_refs': total_filtered_refs,
            'images_with_points': len(filtered_point_indices),
            'original_images_with_points': len(self.point_indices_original),
            'distance_percentile': percentile,
            'error_threshold': error_threshold
        }
        
        print(f"Final filtered points for initialization: {len(self.points)} / {len(self.points_original)} ({100*self.filtering_stats['filtering_ratio']:.1f}%)")
        print(f"Image-point references: {total_filtered_refs} / {total_original_refs} ({100*total_filtered_refs/total_original_refs:.1f}%)")
        print(f"Images with valid points: {len(filtered_point_indices)} / {len(self.point_indices_original)}")
    
    def vis_filtered_points(self, save_path="filtered_points_3d.html"):
        """
        Visualize only the filtered points from _filter_points_for_init.
        
        Args:
            save_path: path to save HTML file
        """
        print(f"Creating filtered points visualization...")
        
        # Check if filtering was applied
        if not hasattr(self, 'points_original'):
            print("No filtering was applied. Use filter=True in Parser initialization.")
            return
        
        # Get filtered points
        points_filtered = self.points
        colors_filtered = self.points_rgb
        
        # Convert colors to RGB strings for Plotly
        rgb_filtered = [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in colors_filtered]
        
        # Create simple 3D scatter plot
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter3d(
                x=points_filtered[:, 0],
                y=points_filtered[:, 1],
                z=points_filtered[:, 2],
                mode='markers',
                marker=dict(size=2, color=rgb_filtered, opacity=0.8),
                showlegend=False,
                hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            )
        )
        
        # Update layout - clean 3D view without any grid, axes, or background
        fig.update_layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(
                    showticklabels=False, 
                    showgrid=False, 
                    zeroline=False, 
                    showline=False,
                    showbackground=False,
                    title=''
                ),
                yaxis=dict(
                    showticklabels=False, 
                    showgrid=False, 
                    zeroline=False, 
                    showline=False,
                    showbackground=False,
                    title=''
                ),
                zaxis=dict(
                    showticklabels=False, 
                    showgrid=False, 
                    zeroline=False, 
                    showline=False,
                    showbackground=False,
                    title=''
                ),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                aspectmode='cube',
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        # Save HTML
        plot(fig, filename=save_path, auto_open=False)
        print(f"Filtered points visualization saved to: {save_path}")
        print(f"  Total points: {len(points_filtered):,}")
        
        return fig

    def vis_soft_masks(self, save_path="soft_masks_visualization.html"):
        """
        Create interactive visualization of soft masks (dynamicness probability) for each lens.

        Args:
            save_path: path to save HTML file
        """
        print(f"Creating soft masks visualization...")
        
        # Get unique camera IDs and sort them
        camera_ids = sorted(self.soft_masks.keys())
        num_cameras = len(camera_ids)
        
        if num_cameras == 0:
            print("No soft masks found to visualize.")
            return
        
        # Create subplot layout based on number of cameras
        if num_cameras <= 2:
            rows, cols = 1, num_cameras
        elif num_cameras <= 4:
            rows, cols = 2, 2
        elif num_cameras <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3  # Max 9 cameras
        
        # Create subplot titles
        subplot_titles = [f'Lens {cam_id}' for cam_id in camera_ids]

        # Create subplots - all 2D heatmaps
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{"type": "xy"} for _ in range(cols)] for _ in range(rows)],
            vertical_spacing=0.12,  # Increased to accommodate annotations
            horizontal_spacing=0.08 
        )
        
        # Add soft mask for each camera
        for idx, camera_id in enumerate(camera_ids):
            row = idx // cols + 1
            col = idx % cols + 1
            
            soft_mask = self.soft_masks[camera_id]

            zeros = (soft_mask == 0.0).sum()
            ones = (soft_mask == 1.0).sum()
            intermediate = ((soft_mask > 0) & (soft_mask < 1)).sum()
            total = soft_mask.size
            h, w = soft_mask.shape
            high_dynamic = np.sum(soft_mask > 0.5)
            
            # Create heatmap trace
            fig.add_trace(
                go.Heatmap(
                    z=soft_mask,
                    colorscale='Hot',  # Good for probability visualization
                    zmin=0.0,
                    zmax=1.0,
                    colorbar=dict(
                        title="Dynamic Probability",
                        x=1.02 if col == cols else None,  # Only show colorbar on rightmost column
                        len=1.0,
                        y=0.5
                    ) if col == cols else None,
                    showscale=(col == cols),  # Only show colorbar on rightmost column
                    hovertemplate='X: %{x}<br>Y: %{y}<br>Probability: %{z:.3f}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Add detailed statistics as annotation for each subplot
            stats_text = (
                f"Size: {h}*{w} ({total:,} pixels)<br>"
                f"Zeros: {zeros:,} ({zeros/total*100:.1f}%)<br>"
                f"Ones: {ones:,} ({ones/total*100:.1f}%)<br>"
                f"Intermediate: {intermediate:,} ({intermediate/total*100:.1f}%)<br>"
                f"Range: [{soft_mask.min():.3f}, {soft_mask.max():.3f}]<br>"
                f"Mean Probability: {soft_mask.mean():.3f}<br>"
                f"High Dynamic (>0.5): {high_dynamic:,} ({100*high_dynamic/total:.1f}%)"
            )
            
            # Calculate annotation position (top-left corner of each subplot)
            fig.add_annotation(
                text=stats_text,
                x=0.02, y=0.98,  # Top-left corner in normalized coordinates
                xref=f"x{idx+1} domain" if idx > 0 else "x domain",
                yref=f"y{idx+1} domain" if idx > 0 else "y domain",
                showarrow=False,
                align="left",
                valign="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=20, color="black")
            )
            
            # Update axis properties for this subplot
            fig.update_xaxes(title_text="X (pixels)", row=row, col=col, showticklabels=True)
            fig.update_yaxes(title_text="Y (pixels)", row=row, col=col, showticklabels=True, autorange="reversed")
        
        # Calculate statistics for title
        total_pixels = sum(mask.size for mask in self.soft_masks.values())
        avg_dynamic_prob = np.mean([np.mean(mask) for mask in self.soft_masks.values()])
        high_dynamic_pixels = sum(np.sum(mask > 0.5) for mask in self.soft_masks.values())
        
        # Update layout
        fig.update_layout(
            title_text=f"<b>Soft Masks - Canonical Dynamicness Probability</b> | "
            f"<span style='font-size:22px'>Avg probability: {avg_dynamic_prob:.3f} | "
            f"High dynamic pixels (>0.5): {high_dynamic_pixels:,}/{total_pixels:,} "
            f"({100*high_dynamic_pixels/total_pixels:.1f}%)</span>",
            title_x=0.5,
            title_font_size=24,
            height=max(768, 768 * rows),
            width=max(576, 576 * cols),
            showlegend=False
        )
        
        # Save HTML
        plot(fig, filename=save_path, auto_open=False)
        print(f"Soft masks visualization saved to: {save_path}")
        
        return fig

    def vis_point_filtering(self, save_path="points3d_filtering_comparison.html"):
        """
        Create interactive HTML visualization comparing original vs filtered points.
        
        Args:
            save_path: path to save HTML file
        """
        print(f"Creating point filtering comparison visualization...")
        
        # Get original and filtered points
        points_original = self.points_original
        points_filtered = self.points
        colors_filtered = self.points_rgb
        
        # Convert colors to RGB strings for Plotly
        rgb_filtered = [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in colors_filtered]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'Original vs Filtered Comparison ({len(points_filtered):,}/{len(points_original):,} points, {100*len(points_filtered)/len(points_original):.1f}%)',
                f'XY Projection: Original (coral) vs Filtered (colored by Z)',
                f'XZ Projection: Original (coral) vs Filtered (colored by Y)',
                f'YZ Projection: Original (coral) vs Filtered (colored by X)'
            ),
            specs=[[{"type": "scatter3d"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Comparison: Original (coral) vs Filtered (colored) in 3D
        # Original points 
        fig.add_trace(
            go.Scatter3d(
                x=points_original[:, 0],
                y=points_original[:, 1],
                z=points_original[:, 2],
                mode='markers',
                marker=dict(size=1, color='lightcoral', opacity=0.3),
                name='Original Points',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Filtered points overlay
        fig.add_trace(
            go.Scatter3d(
                x=points_filtered[:, 0],
                y=points_filtered[:, 1],
                z=points_filtered[:, 2],
                mode='markers',
                marker=dict(size=2, color=rgb_filtered, opacity=0.9),
                name='Filtered Overlay',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. XY projection comparison: Original vs Filtered
        # Original points 
        fig.add_trace(
            go.Scatter(
                x=points_original[:, 0],
                y=points_original[:, 1],
                mode='markers',
                marker=dict(
                    size=2,
                    color='lightcoral',
                    opacity=0.4
                ),
                name='Original (XY)',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Filtered points overlay with Z color coding
        fig.add_trace(
            go.Scatter(
                x=points_filtered[:, 0],
                y=points_filtered[:, 1],
                mode='markers',
                marker=dict(
                    size=3,
                    color=points_filtered[:, 2],
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(
                        title="Z", 
                        x=1.00,
                        len=0.5,
                        y=0.79,
                        xanchor="left"
                    )
                ),
                name='Filtered (XY)',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. XZ projection comparison: Original vs Filtered
        # Original points 
        fig.add_trace(
            go.Scatter(
                x=points_original[:, 0],
                y=points_original[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='lightcoral',
                    opacity=0.4
                ),
                name='Original (XZ)',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Filtered points overlay with Y color coding
        fig.add_trace(
            go.Scatter(
                x=points_filtered[:, 0],
                y=points_filtered[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=points_filtered[:, 1],
                    colorscale='Plasma',
                    opacity=0.8,
                    colorbar=dict(
                        title="Y", 
                        x=0.45,
                        len=0.5,
                        y=0.24,
                        xanchor="left"
                    )
                ),
                name='Filtered (XZ)',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. YZ projection comparison: Original vs Filtered
        # Original points 
        fig.add_trace(
            go.Scatter(
                x=points_original[:, 1],
                y=points_original[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='lightcoral',
                    opacity=0.4
                ),
                name='Original (YZ)',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Filtered points overlay with X color coding
        fig.add_trace(
            go.Scatter(
                x=points_filtered[:, 1],
                y=points_filtered[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=points_filtered[:, 0],
                    colorscale='Cividis',
                    opacity=0.8,
                    colorbar=dict(
                        title="X", 
                        x=1.00,
                        len=0.5,
                        y=0.24,
                        xanchor="left"
                    )
                ),
                name='Filtered (YZ)',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        stats = self.filtering_stats
        fig.update_layout(
            title_text=f"Point Filtering Comparison<br>",
            title_x=0.5,
            height=1000,
            width=1400
        )
        
        # Update 3D scene layout (only one 3D subplot now)
        fig.update_layout(scene1=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode='cube'
        ))
        
        # Update 2D subplot layouts
        fig.update_xaxes(title_text="X", row=1, col=2)
        fig.update_yaxes(title_text="Y", row=1, col=2)
        fig.update_xaxes(title_text="X", row=2, col=1)
        fig.update_yaxes(title_text="Z", row=2, col=1)
        fig.update_xaxes(title_text="Y", row=2, col=2)
        fig.update_yaxes(title_text="Z", row=2, col=2)
        
        # Save HTML
        plot(fig, filename=save_path, auto_open=False)
        print(f"Point filtering comparison HTML saved to: {save_path}")
        
        # Print filtering statistics
        print(f"\nFiltering Statistics:")
        print(f"  Original points: {stats['original_points']:,}")
        print(f"  Filtered points: {stats['filtered_points']:,}")
        print(f"  Filtering ratio: {100*stats['filtering_ratio']:.1f}%")
        print(f"  Distance percentile threshold: {stats['distance_percentile']}%")
        print(f"  Error threshold: {stats['error_threshold']}")
        print(f"  Original image-point references: {stats['original_image_point_refs']:,}")
        print(f"  Filtered image-point references: {stats['filtered_image_point_refs']:,}")
        print(f"  Images with points: {stats['images_with_points']} / {stats['original_images_with_points']}")
        
        return fig
    
    def vis_rig_centers(self, save_path=None):
        """Visualize rig centers and camera positions in 3D space"""
        if not hasattr(self, 'rig_centers') or len(self.rig_centers) == 0:
            print("No rig centers to visualize")
            return
        
        # Extract rig center positions
        times = sorted(self.rig_centers.keys())
        rig_positions = np.array([self.rig_centers[t] for t in times])
        
        # Create figure
        fig = go.Figure()
        
        # Plot rig centers trajectory
        fig.add_trace(go.Scatter3d(
            x=rig_positions[:, 0],
            y=rig_positions[:, 1],
            z=rig_positions[:, 2],
            mode='lines+markers',
            marker=dict(size=6, color='red'),
            line=dict(color='red', width=2),
            name='Rig Centers'
        ))
        
        # Optionally: Plot individual camera positions at first timestamp
        first_time = times[0]
        unique_cameras = list(set(self.camera_ids))
        camera_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta']
        
        for cam_idx, camera_id in enumerate(sorted(unique_cameras)):
            cam_positions = []
            for i, (camtoworld, cam_id, time) in enumerate(zip(self.camtoworlds, self.camera_ids, self.times)):
                if cam_id == camera_id:
                    cam_positions.append(camtoworld[:3, 3])
            
            if len(cam_positions) > 0:
                cam_positions = np.array(cam_positions)
                color = camera_colors[cam_idx % len(camera_colors)]
                fig.add_trace(go.Scatter3d(
                    x=cam_positions[:, 0],
                    y=cam_positions[:, 1],
                    z=cam_positions[:, 2],
                    mode='markers',
                    marker=dict(size=3, color=color, opacity=0.5),
                    name=f'Camera {camera_id}'
                ))
        
        # Update layout
        fig.update_layout(
            title='Rig Centers and Camera Trajectories',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=1000,
            height=800
        )
        
        # Save or show
        if save_path:
            plot(fig, filename=save_path, auto_open=False)
            print(f"Saved visualization to {save_path}")
        else:
            fig.show()
    
    def vis_points_by_error(self, error_threshold=1.0, save_path="points3d_error_analysis.html"):
        """
        Visualize point cloud filtered by reprojection error to assess reliability.
        
        Args:
            error_threshold: Maximum reprojection error (in pixels) for "reliable" points
            save_path: Path to save HTML file
        """
        print(f"\n{'='*60}")
        print(f"Analyzing Point Cloud by Reprojection Error")
        print(f"{'='*60}")
        
        # Get original points and errors
        points = self.points_original
        points_err = self.points_err_original
        points_rgb = self.points_rgb_original
        
        # Filter by error threshold
        reliable_mask = points_err <= error_threshold
        unreliable_mask = ~reliable_mask
        
        points_reliable = points[reliable_mask]
        points_unreliable = points[unreliable_mask]
        colors_reliable = points_rgb[reliable_mask]
        errors_reliable = points_err[reliable_mask]
        errors_unreliable = points_err[unreliable_mask]
        
        # Convert colors to RGB strings
        rgb_reliable = [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in colors_reliable]
        
        # Statistics
        total_points = len(points)
        num_reliable = len(points_reliable)
        num_unreliable = len(points_unreliable)
        reliable_ratio = num_reliable / total_points if total_points > 0 else 0
        
        print(f"\nError Analysis:")
        print(f"  Error threshold: {error_threshold:.2f} pixels")
        print(f"  Total points: {total_points:,}")
        print(f"  Reliable points (<={error_threshold:.2f}px): {num_reliable:,} ({100*reliable_ratio:.1f}%)")
        print(f"  Unreliable points (>{error_threshold:.2f}px): {num_unreliable:,} ({100*(1-reliable_ratio):.1f}%)")
        print(f"\nError Statistics:")
        print(f"  All points - mean: {points_err.mean():.3f}px, median: {np.median(points_err):.3f}px")
        print(f"  All points - min: {points_err.min():.3f}px, max: {points_err.max():.3f}px")
        if num_reliable > 0:
            print(f"  Reliable - mean: {errors_reliable.mean():.3f}px, median: {np.median(errors_reliable):.3f}px")
        if num_unreliable > 0:
            print(f"  Unreliable - mean: {errors_unreliable.mean():.3f}px, median: {np.median(errors_unreliable):.3f}px")
        
        # Spatial statistics
        if num_reliable > 0:
            reliable_extent = np.linalg.norm(points_reliable, axis=1)
            print(f"\nReliable Points Spatial Distribution:")
            print(f"  Distance from origin - mean: {reliable_extent.mean():.3f}m, median: {np.median(reliable_extent):.3f}m")
            print(f"  Distance from origin - min: {reliable_extent.min():.3f}m, max: {reliable_extent.max():.3f}m")
            print(f"  XYZ range: X[{points_reliable[:, 0].min():.2f}, {points_reliable[:, 0].max():.2f}], "
                f"Y[{points_reliable[:, 1].min():.2f}, {points_reliable[:, 1].max():.2f}], "
                f"Z[{points_reliable[:, 2].min():.2f}, {points_reliable[:, 2].max():.2f}]")
        
        # Create subplots: 2x3 layout
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                f'3D View: Reliable (colored by RGB) vs Unreliable (red)',
                f'3D View: All Points Colored by Error',
                f'Error Distribution Histogram',
                f'XY Projection: Reliable vs Unreliable',
                f'XZ Projection: Reliable vs Unreliable',
                f'YZ Projection: Reliable vs Unreliable'
            ),
            specs=[
                [{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        # 1. 3D: Reliable (RGB) vs Unreliable (red)
        if num_unreliable > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=points_unreliable[:, 0],
                    y=points_unreliable[:, 1],
                    z=points_unreliable[:, 2],
                    mode='markers',
                    marker=dict(size=1, color='red', opacity=0.3),
                    name='Unreliable',
                    hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        if num_reliable > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=points_reliable[:, 0],
                    y=points_reliable[:, 1],
                    z=points_reliable[:, 2],
                    mode='markers',
                    marker=dict(size=2, color=rgb_reliable, opacity=0.8),
                    name='Reliable',
                    hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # 2. 3D: All points colored by error
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=points_err,
                    colorscale='Turbo',
                    cmin=0,
                    cmax=np.percentile(points_err, 95),  # Cap at 95th percentile for better visualization
                    colorbar=dict(
                        title="Error (px)",
                        x=0.6,
                        len=0.4,
                        y=0.75
                    ),
                    opacity=0.7
                ),
                name='By Error',
                hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>Error: %{marker.color:.3f}px<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Error histogram
        fig.add_trace(
            go.Histogram(
                x=points_err,
                nbinsx=100,
                marker=dict(
                    color=points_err,
                    colorscale='Turbo',
                    line=dict(width=0)
                ),
                hovertemplate='Error: %{x:.2f}px<br>Count: %{y}<extra></extra>',
                showlegend=False
            ),
            row=1, col=3
        )
        
        # Add threshold line as a shape to the histogram subplot (row=1, col=3, which is subplot 3 in a 2x3 grid)
        fig.add_shape(
            type="line",
            x0=error_threshold,
            x1=error_threshold,
            y0=0,  # Start from bottom of y-axis
            y1=1,  # End at top of y-axis (normalized coordinates)
            xref='x3',  # Refers to the x-axis of the 3rd subplot (row=1, col=3)
            yref='y3',  # Refers to the y-axis of the 3rd subplot
            line=dict(color='red', width=2, dash='dash')
        )

        # Add annotation for the threshold line
        fig.add_annotation(
            x=error_threshold,
            y=0.98,  # Position near the top of the subplot
            xref='x3',
            yref='y3',
            text=f'Threshold={error_threshold:.2f}px',
            showarrow=False,
            font=dict(color='red', size=12),
            align='center'
        )
        
        # 4. XY Projection
        if num_unreliable > 0:
            fig.add_trace(
                go.Scatter(
                    x=points_unreliable[:, 0],
                    y=points_unreliable[:, 1],
                    mode='markers',
                    marker=dict(size=2, color='lightcoral', opacity=0.3),
                    name='Unreliable (XY)',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        if num_reliable > 0:
            fig.add_trace(
                go.Scatter(
                    x=points_reliable[:, 0],
                    y=points_reliable[:, 1],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=points_reliable[:, 2],  # Color by Z
                        colorscale='Viridis',
                        opacity=0.8,
                        colorbar=dict(
                            title="Z (m)",
                            x=0.28,
                            len=0.35,
                            y=0.22
                        )
                    ),
                    name='Reliable (XY)',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 5. XZ Projection
        if num_unreliable > 0:
            fig.add_trace(
                go.Scatter(
                    x=points_unreliable[:, 0],
                    y=points_unreliable[:, 2],
                    mode='markers',
                    marker=dict(size=2, color='lightcoral', opacity=0.3),
                    name='Unreliable (XZ)',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        if num_reliable > 0:
            fig.add_trace(
                go.Scatter(
                    x=points_reliable[:, 0],
                    y=points_reliable[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=points_reliable[:, 1],  # Color by Y
                        colorscale='Plasma',
                        opacity=0.8,
                        colorbar=dict(
                            title="Y (m)",
                            x=0.64,
                            len=0.35,
                            y=0.22
                        )
                    ),
                    name='Reliable (XZ)',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # 6. YZ Projection
        if num_unreliable > 0:
            fig.add_trace(
                go.Scatter(
                    x=points_unreliable[:, 1],
                    y=points_unreliable[:, 2],
                    mode='markers',
                    marker=dict(size=2, color='lightcoral', opacity=0.3),
                    name='Unreliable (YZ)',
                    showlegend=False
                ),
                row=2, col=3
            )
        
        if num_reliable > 0:
            fig.add_trace(
                go.Scatter(
                    x=points_reliable[:, 1],
                    y=points_reliable[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=points_reliable[:, 0],  # Color by X
                        colorscale='Cividis',
                        opacity=0.8,
                        colorbar=dict(
                            title="X (m)",
                            x=1.0,
                            len=0.35,
                            y=0.22
                        )
                    ),
                    name='Reliable (YZ)',
                    showlegend=False
                ),
                row=2, col=3
            )
        
        # Update layout
        fig.update_layout(
            title_text=f"<b>Point Cloud Quality Analysis by Reprojection Error</b><br>"
                    f"<span style='font-size:18px'>Threshold: {error_threshold:.2f}px | "
                    f"Reliable: {num_reliable:,}/{total_points:,} ({100*reliable_ratio:.1f}%) | "
                    f"Mean Error: {points_err.mean():.3f}px</span>",
            title_x=0.5,
            title_font_size=22,
            height=1000,
            width=1800,
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
        )
        
        # Update 3D scenes
        fig.update_layout(scene1=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode='cube'
        ))
        
        # Update 2D subplot axes
        fig.update_xaxes(title_text="X (m)", row=2, col=1)
        fig.update_yaxes(title_text="Y (m)", row=2, col=1)
        fig.update_xaxes(title_text="X (m)", row=2, col=2)
        fig.update_yaxes(title_text="Z (m)", row=2, col=2)
        fig.update_xaxes(title_text="Y (m)", row=2, col=3)
        fig.update_yaxes(title_text="Z (m)", row=2, col=3)
        
        # Histogram axes
        fig.update_xaxes(title_text="Reprojection Error (pixels)", row=1, col=3)
        fig.update_yaxes(title_text="Count", row=1, col=3)
        
        # Save HTML
        plot(fig, filename=save_path, auto_open=False)
        print(f"\nVisualization saved to: {save_path}")
        print(f"{'='*60}\n")
        
        # Return statistics for further analysis
        return {
            'total_points': total_points,
            'num_reliable': num_reliable,
            'num_unreliable': num_unreliable,
            'reliable_ratio': reliable_ratio,
            'error_threshold': error_threshold,
            'mean_error_all': float(points_err.mean()),
            'median_error_all': float(np.median(points_err)),
            'mean_error_reliable': float(errors_reliable.mean()) if num_reliable > 0 else None,
            'mean_error_unreliable': float(errors_unreliable.mean()) if num_unreliable > 0 else None,
            'points_reliable': points_reliable,
            'colors_reliable': colors_reliable,
            'errors_reliable': errors_reliable
        }
    
    def vis_points_by_error_and_distance(
        self, 
        error_threshold=0.3, 
        distance_percentile=95,
        save_path="points3d_reliable_analysis.html"
    ):
        """
        Visualize point cloud filtered by BOTH reprojection error AND distance from center.
        This helps identify truly reliable points for metric scale anchors.
        
        Args:
            error_threshold: Maximum reprojection error (in pixels) for "reliable" points
            distance_percentile: Percentile threshold for distance filtering (e.g., 95 = keep closest 95%)
            save_path: Path to save HTML file
        """
        print(f"\n{'='*60}")
        print(f"Analyzing Point Cloud by Reprojection Error + Distance")
        print(f"{'='*60}")
        
        # Get original points and errors
        points = self.points_original
        points_err = self.points_err_original
        points_rgb = self.points_rgb_original
        
        # Compute distances from origin
        distances = np.linalg.norm(points, axis=1)
        distance_threshold = np.percentile(distances, distance_percentile)
        
        # Apply filters progressively to show the effect
        error_mask = points_err <= error_threshold
        distance_mask = distances <= distance_threshold
        combined_mask = error_mask & distance_mask
        
        # Create filtered point sets
        points_error_only = points[error_mask]
        colors_error_only = points_rgb[error_mask]
        
        points_reliable = points[combined_mask]
        colors_reliable = points_rgb[combined_mask]
        errors_reliable = points_err[combined_mask]
        distances_reliable = distances[combined_mask]
        
        points_unreliable = points[~combined_mask]
        
        # Points that pass error but fail distance (the problematic outliers!)
        outlier_mask = error_mask & ~distance_mask
        points_outliers = points[outlier_mask]
        errors_outliers = points_err[outlier_mask]
        distances_outliers = distances[outlier_mask]
        
        # Convert colors to RGB strings
        rgb_reliable = [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in colors_reliable]
        
        # Statistics
        total_points = len(points)
        num_error_only = len(points_error_only)
        num_reliable = len(points_reliable)
        num_outliers = len(points_outliers)
        num_unreliable = len(points_unreliable)
        
        print(f"\nFiltering Analysis:")
        print(f"  Error threshold: {error_threshold:.2f} pixels")
        print(f"  Distance threshold: {distance_threshold:.2f}m ({distance_percentile}th percentile)")
        print(f"  Total points: {total_points:,}")
        print(f"\nFiltering Results:")
        print(f"  Pass error only: {num_error_only:,} ({100*num_error_only/total_points:.1f}%)")
        print(f"  Pass both filters (RELIABLE): {num_reliable:,} ({100*num_reliable/total_points:.1f}%)")
        print(f"  Pass error but fail distance (OUTLIERS): {num_outliers:,} ({100*num_outliers/total_points:.1f}%)")
        print(f"  Fail filters: {num_unreliable:,} ({100*num_unreliable/total_points:.1f}%)")
        
        print(f"\nError Statistics:")
        print(f"  All points - mean: {points_err.mean():.3f}px, median: {np.median(points_err):.3f}px")
        if num_reliable > 0:
            print(f"  Reliable points - mean: {errors_reliable.mean():.3f}px, median: {np.median(errors_reliable):.3f}px")
        if num_outliers > 0:
            print(f"  Outliers (low error, far distance) - mean: {errors_outliers.mean():.3f}px, median: {np.median(errors_outliers):.3f}px")
        
        print(f"\nDistance Statistics:")
        print(f"  All points - mean: {distances.mean():.3f}m, median: {np.median(distances):.3f}m")
        if num_reliable > 0:
            print(f"  Reliable points - mean: {distances_reliable.mean():.3f}m, median: {np.median(distances_reliable):.3f}m")
            print(f"  Reliable points - max: {distances_reliable.max():.3f}m")
        if num_outliers > 0:
            print(f"  Outliers - mean: {distances_outliers.mean():.3f}m, median: {np.median(distances_outliers):.3f}m")
            print(f"  Outliers - min: {distances_outliers.min():.3f}m, max: {distances_outliers.max():.3f}m")
        
        # Spatial statistics for reliable points
        if num_reliable > 0:
            print(f"\nReliable Points Spatial Distribution:")
            print(f"  XYZ range: X[{points_reliable[:, 0].min():.2f}, {points_reliable[:, 0].max():.2f}], "
                f"Y[{points_reliable[:, 1].min():.2f}, {points_reliable[:, 1].max():.2f}], "
                f"Z[{points_reliable[:, 2].min():.2f}, {points_reliable[:, 2].max():.2f}]")
        
        # Create subplots: 2x3 layout
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                f'3D: Reliable (green) vs Outliers (orange) vs Others (gray)',
                f'3D View: Reliable (colored by RGB) vs Unreliable (red)',
                f'3D View: All Points Colored by Error'
            ),
            specs=[
                [{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        # 1. 3D: Show three categories (green/orange/gray)
        if num_unreliable > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=points_unreliable[:, 0],
                    y=points_unreliable[:, 1],
                    z=points_unreliable[:, 2],
                    mode='markers',
                    marker=dict(size=1, color='darkgray', opacity=0.3),
                    name='Fail Both',
                    hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        if num_outliers > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=points_outliers[:, 0],
                    y=points_outliers[:, 1],
                    z=points_outliers[:, 2],
                    mode='markers',
                    marker=dict(size=3, color='orange', opacity=0.7),
                    name='Outliers (low err, far)',
                    hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>'
                                'Error: %{customdata[0]:.3f}px<br>Distance: %{customdata[1]:.2f}m<extra></extra>',
                    customdata=np.stack([errors_outliers, distances_outliers], axis=1),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        if num_reliable > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=points_reliable[:, 0],
                    y=points_reliable[:, 1],
                    z=points_reliable[:, 2],
                    mode='markers',
                    marker=dict(size=2, color='green', opacity=0.8),
                    name='Reliable',
                    hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # 2. 3D: Reliable (RGB) vs Unreliable (red)
        # First add unreliable points in red
        unreliable_combined = points[~combined_mask]
        if len(unreliable_combined) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=unreliable_combined[:, 0],
                    y=unreliable_combined[:, 1],
                    z=unreliable_combined[:, 2],
                    mode='markers',
                    marker=dict(size=1, color='red', opacity=0.3),
                    name='Unreliable',
                    hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Then overlay reliable points with RGB colors
        if num_reliable > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=points_reliable[:, 0],
                    y=points_reliable[:, 1],
                    z=points_reliable[:, 2],
                    mode='markers',
                    marker=dict(size=2, color=rgb_reliable, opacity=0.8),
                    name='Reliable (RGB)',
                    hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. 3D: All points colored by error
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=points_err,
                    colorscale='Turbo',
                    cmin=0,
                    cmax=np.percentile(points_err, 95),
                    colorbar=dict(
                        title="Error (px)",
                        x=0.99,
                        len=0.4,
                        y=0.75
                    ),
                    opacity=0.7
                ),
                name='By Error',
                hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>Error: %{marker.color:.3f}px<extra></extra>',
                showlegend=False
            ),
            row=1, col=3
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"<b>Reliable Points Analysis: Error + Distance Filtering</b><br>"
                    f"<span style='font-size:12px'>Error<={error_threshold:.2f}px AND Distance<={distance_threshold:.1f}m ({distance_percentile}%ile) | "
                    f"Reliable: {num_reliable:,}/{total_points:,} ({100*num_reliable/total_points:.1f}%) | "
                    f"Outliers: {num_outliers:,} ({100*num_outliers/total_points:.1f}%)</span>",
            title_x=0.5,
            title_font_size=20,
            height=1000,
            width=1800,
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
        )
        
        # Update 3D scenes
        for scene_num in [1, 2, 3]:
            fig.update_layout(**{
                f'scene{scene_num}': dict(
                    xaxis=dict(title='X (m)'),
                    yaxis=dict(title='Y (m)'),
                    zaxis=dict(title='Z (m)'),
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                    aspectmode='cube'
                )
            })
        
        # Save HTML
        plot(fig, filename=save_path, auto_open=False)
        print(f"\nVisualization saved to: {save_path}")
        print(f"{'='*60}\n")
        
        # Return comprehensive statistics
        return {
            'total_points': total_points,
            'num_reliable': num_reliable,
            'num_outliers': num_outliers,
            'num_unreliable': num_unreliable,
            'reliable_ratio': num_reliable / total_points if total_points > 0 else 0,
            'outlier_ratio': num_outliers / total_points if total_points > 0 else 0,
            'error_threshold': error_threshold,
            'distance_threshold': distance_threshold,
            'distance_percentile': distance_percentile,
            'mean_error_reliable': float(errors_reliable.mean()) if num_reliable > 0 else None,
            'mean_distance_reliable': float(distances_reliable.mean()) if num_reliable > 0 else None,
            'mean_error_outliers': float(errors_outliers.mean()) if num_outliers > 0 else None,
            'mean_distance_outliers': float(distances_outliers.mean()) if num_outliers > 0 else None,
            'points_reliable': points_reliable,
            'colors_reliable': colors_reliable,
            'errors_reliable': errors_reliable,
            'distances_reliable': distances_reliable,
        }
    
    def analyze_error_thresholds(self, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0], save_path="error_threshold_analysis.html"):
        """
        Analyze how different error thresholds affect the number of reliable points.
        Helps determine optimal threshold for metric scale anchors.
        
        Args:
            thresholds: List of error thresholds to test (in pixels)
            save_path: Path to save analysis HTML
        """
        print(f"\n{'='*60}")
        print(f"Error Threshold Analysis")
        print(f"{'='*60}")
        
        points_err = self.points_err_original
        total_points = len(points_err)
        
        # Compute statistics for each threshold
        results = []
        for threshold in thresholds:
            reliable_mask = points_err <= threshold
            num_reliable = np.sum(reliable_mask)
            ratio = num_reliable / total_points
            mean_err = points_err[reliable_mask].mean() if num_reliable > 0 else 0
            
            results.append({
                'threshold': threshold,
                'num_reliable': num_reliable,
                'ratio': ratio,
                'mean_error': mean_err
            })
            
            print(f"  Threshold {threshold:.2f}px: {num_reliable:,}/{total_points:,} points "
                f"({100*ratio:.1f}%), mean error: {mean_err:.3f}px")
        
        # Create visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                'Number of Reliable Points vs Threshold',
                'Percentage of Reliable Points vs Threshold'
            ),
            specs=[[{"type": "xy"}, {"type": "xy"}]]
        )
        
        # Plot 1: Absolute count
        fig.add_trace(
            go.Scatter(
                x=[r['threshold'] for r in results],
                y=[r['num_reliable'] for r in results],
                mode='lines+markers',
                marker=dict(size=10, color='blue'),
                line=dict(width=2),
                name='Reliable Points',
                hovertemplate='Threshold: %{x:.2f}px<br>Points: %{y:,}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Plot 2: Percentage
        fig.add_trace(
            go.Scatter(
                x=[r['threshold'] for r in results],
                y=[r['ratio'] * 100 for r in results],
                mode='lines+markers',
                marker=dict(size=10, color='green'),
                line=dict(width=2),
                name='Percentage',
                hovertemplate='Threshold: %{x:.2f}px<br>Percentage: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Error Threshold (pixels)", row=1, col=1)
        fig.update_yaxes(title_text="Number of Points", row=1, col=1)
        fig.update_xaxes(title_text="Error Threshold (pixels)", row=1, col=2)
        fig.update_yaxes(title_text="Percentage (%)", row=1, col=2)
        
        fig.update_layout(
            title_text=f"<b>Error Threshold Analysis</b><br>"
                    f"<span style='font-size:16px'>Total Points: {total_points:,}</span>",
            title_x=0.5,
            height=500,
            width=1400,
            showlegend=False
        )
        
        plot(fig, filename=save_path, auto_open=False)
        print(f"\nThreshold analysis saved to: {save_path}")
        print(f"{'='*60}\n")
        
        return results

class FisheyeDataset:
    def __init__(
        self,
        parser: Parser,             # The Parser object containing dataset information
        split: str = "train",       # "train" or "test"
        patch_size: Optional[int] = None, # If not None, random crop to patch_size x patch_size
        load_depths: bool = False,  # Whether to load mono depths and dynamic masks
        pattern_length: int = 10,   # Total length of the pattern (train + test)
        train_length: int = 7,      # Length of training block in the pattern
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        
        indices = np.arange(len(self.parser.image_names))
        if split == "train":  # 7 train + 3 test
            # Select indices 0-6, 10-16, 20-26, etc
            self.indices = indices[indices % pattern_length < train_length]
        else:
            # Select indices 7, 8, 9, 17, 18, 19, 27, 28, 29, etc
            self.indices = indices[indices % pattern_length >= train_length]
        
        """ 10 split
        ### train: 1, 2, 3, ..., 7, 8, 9, 11, ...
        ### test: 0, 10, 20, ...
            
            if split == "train":
                self.indices = indices[indices % self.parser.test_every != 0]
            else:
                self.indices = indices[indices % self.parser.test_every == 0]
        """
        """ 10 split
        ### train: 0, 1, 2, 3, ..., 7, 8, 10, ...
        ### test: 9, 19, 29, ...

            if split == "train":
                self.indices = indices[(indices+1) % self.parser.test_every != 0]
            else:
                self.indices = indices[(indices+1) % self.parser.test_every == 0]
        """

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]  # index of the image in the dataset
        image = imageio.imread(self.parser.image_paths[index])[..., :3] # np.ndarray, (height, width, 3)
        image_name = self.parser.image_names[index]  # str
        camera_id = self.parser.camera_ids[index]    # int
        K = self.parser.Ks_dict[camera_id].copy()    # np.ndarray, (3, 3)
        camtoworlds = self.parser.camtoworlds[index] # np.ndarray, (4, 4)
        params = self.parser.params_dict[camera_id].copy()     # np.ndarray, (4,)
        soft_masks = self.parser.soft_masks[camera_id].copy()  # np.ndarray, (height, width)
        dynamic_masks = self.parser.dynamic_masks[index]       # np.ndarray, (height, width)
        mono_depths = self.parser.mono_depths[index] # np.ndarray, (height, width), in [0, 1]
        mask = self.parser.shared_mask               # np.ndarray, (height, width)
        eroded_mask = self.parser.eroded_mask        # np.ndarray, (height, width)
        time_id = self.parser.times[index]           # int
        rig_center = self.parser.rig_centers.get(time_id, None) # np.ndarray, (3,)
        cam_depth_min, cam_depth_max = self.parser.camera_depth_ranges[camera_id]
        frame_time_id = self.parser.time_ids[index]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(), # (3, 3)
            "camtoworld": torch.from_numpy(camtoworlds).float(),  # (4, 4)
            "image": torch.from_numpy(image).float(), # (height, width, 3)
            "image_id": item,  # the index of the image in the dataset
            "camera_id": camera_id,   # int
            "image_name": image_name, # str
            "poly_coeffs": torch.from_numpy(params).float(),    # (4,)
            "rig_center": torch.from_numpy(rig_center).float(), # (3,)
            "frame_time_id": frame_time_id,  # int
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()  # (height, width)
        if eroded_mask is not None:
            data["eroded_mask"] = torch.from_numpy(eroded_mask).bool()  # (height, width)
        if time_id is not None:
            data["time_id"] = time_id  # int
        if dynamic_masks is not None:
            data["dynamic_masks"] = torch.from_numpy(dynamic_masks).float()  # (height, width)
        if soft_masks is not None:
            data["soft_masks"] = torch.from_numpy(soft_masks).float()  # (height, width)
        if self.load_depths and mono_depths is not None:
            data["mono_depths"] = torch.from_numpy(mono_depths).float()  # (height, width)
            data["mono_depth_min"] = cam_depth_min  # float
            data["mono_depth_max"] = cam_depth_max  # float
        
        # Load metric depths if available
        if self.load_depths and hasattr(self.parser, 'metric_depths') and self.parser.metric_depths is not None:
            metric_depth = self.parser.metric_depths[index]  # np.ndarray, (height, width), in meters
            data["metric_depths"] = torch.from_numpy(metric_depth).float()  # (height, width)

        return data