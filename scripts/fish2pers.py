#!/usr/bin/env python3
"""
Convert rendered results from fisheye domain to perspective domain.

This script:
1. Loads perspective crop configurations from perspective_config.json
2. Applies the same fisheye-to-perspective transformation to rendered results
3. Converts GT and rendered images from train/test splits
4. Saves perspective results for fair comparison with pinhole-based methods

Input structure:
    results/{scene}/
        train/
            gt_final/lens01/frame_0001.png
            renders_final/lens01/frame_0001.png
        test/
            gt_final/lens01/frame_0008.png
            renders_final/lens01/frame_0008.png

Output structure:
    results/{scene}/
        train/
            gt_final_pers/crop01/frame_0001.png
            renders_final_pers/crop01/frame_0001.png
        test/
            gt_final_pers/crop01/frame_0008.png
            renders_final_pers/crop01/frame_0008.png
"""

import numpy as np
import cv2
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm


def parse_camera_params(camera_params_path: Path) -> Dict[int, Dict]:
    """
    Parse COLMAP-style camera parameters file.
    
    Returns:
        Dictionary mapping camera_id to parameters
    """
    cameras = {}
    with open(camera_params_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            
            if model == 'OPENCV_FISHEYE':
                cameras[camera_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'fx': params[0],
                    'fy': params[1],
                    'cx': params[2],
                    'cy': params[3],
                    'k1': params[4],
                    'k2': params[5],
                    'k3': params[6],
                    'k4': params[7],
                }
            elif model == 'PINHOLE':
                cameras[camera_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'fx': params[0],
                    'fy': params[1],
                    'cx': params[2],
                    'cy': params[3],
                }
            else:
                raise ValueError(f"Unsupported camera model: {model}")
    
    return cameras


def load_perspective_config(perspective_config_path: Path) -> Dict:
    """Load perspective crop configuration from JSON file."""
    with open(perspective_config_path, 'r') as f:
        config = json.load(f)
    return config


def undistort_and_crop_fisheye(
    fisheye_img: np.ndarray,
    fisheye_params: Dict,
    crop_config: Dict,
) -> Optional[np.ndarray]:
    """
    Undistort fisheye image and extract perspective crop.
    
    Args:
        fisheye_img: Input fisheye image (H, W, 3)
        fisheye_params: Fisheye camera parameters
        crop_config: Perspective crop configuration
    
    Returns:
        Perspective crop image (crop_height, crop_width, 3) or None if invalid
    """
    # Fisheye intrinsics
    K_fisheye = np.array([
        [fisheye_params['fx'], 0, fisheye_params['cx']],
        [0, fisheye_params['fy'], fisheye_params['cy']],
        [0, 0, 1]
    ])
    D_fisheye = np.array([
        fisheye_params['k1'],
        fisheye_params['k2'],
        fisheye_params['k3'],
        fisheye_params['k4']
    ])
    
    # Perspective crop intrinsics
    K_crop = np.array([
        [crop_config['fx'], 0, crop_config['cx']],
        [0, crop_config['fy'], crop_config['cy']],
        [0, 0, 1]
    ])
    
    # Rotation from crop to fisheye (reconstruct from yaw/pitch)
    yaw = np.radians(crop_config['yaw_deg'])
    pitch = np.radians(crop_config['pitch_deg'])
    
    R_yaw = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    R = R_pitch @ R_yaw
    
    # Create pixel grid for perspective crop
    crop_height = crop_config['height']
    crop_width = crop_config['width']
    
    # Generate 3D rays in crop camera frame
    u, v = np.meshgrid(np.arange(crop_width), np.arange(crop_height))
    pixels_crop = np.stack([u, v, np.ones_like(u)], axis=-1).astype(np.float32)
    
    # Convert to normalized camera coordinates
    rays_crop = (pixels_crop @ np.linalg.inv(K_crop).T).reshape(-1, 3)
    
    # Rotate to fisheye frame
    rays_fisheye = (R @ rays_crop.T).T
    
    # Project to fisheye image using cv2.fisheye.projectPoints
    rays_fisheye = rays_fisheye.reshape(-1, 1, 3).astype(np.float32)
    
    # Project using fisheye model
    pixels_fisheye, _ = cv2.fisheye.projectPoints(
        rays_fisheye,
        np.zeros(3),  # No rotation (already in fisheye frame)
        np.zeros(3),  # No translation
        K_fisheye,
        D_fisheye
    )
    pixels_fisheye = pixels_fisheye.reshape(crop_height, crop_width, 2)
    
    # Remap from fisheye to perspective
    map_x = pixels_fisheye[:, :, 0].astype(np.float32)
    map_y = pixels_fisheye[:, :, 1].astype(np.float32)
    
    perspective_crop = cv2.remap(
        fisheye_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    # Check if crop is valid (not too much black area)
    valid_mask = (map_x >= 0) & (map_x < fisheye_params['width']) & \
                 (map_y >= 0) & (map_y < fisheye_params['height'])
    valid_ratio = valid_mask.sum() / (crop_height * crop_width)
    
    if valid_ratio < 0.5:  # Less than 50% valid pixels
        return None
    
    return perspective_crop


def convert_results_to_perspective(
    scene_name: str,
    results_root: Path,
    camera_params_path: Path,
    perspective_config_path: Path,
    splits: List[str] = ['train', 'test'],
    result_types: List[str] = ['gt_final', 'renders_final'],
    verbose: bool = True,
):
    """
    Convert rendered results from fisheye domain to perspective domain.
    
    Args:
        scene_name: Name of the scene (e.g., 'concert')
        results_root: Root directory containing results
        data_root: Root directory containing scene data (for camera params)
        camera_params_path: Path to fisheye camera parameters
        perspective_config_path: Path to perspective crop configuration
        splits: List of splits to process (default: ['train', 'test'])
        result_types: List of result types to process (default: ['gt_final', 'renders_final'])
        verbose: Print progress
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Converting results to perspective domain: {scene_name}")
        print(f"{'='*60}")
    
    # Load camera parameters and perspective config
    fisheye_cameras = parse_camera_params(camera_params_path)
    perspective_config = load_perspective_config(perspective_config_path)
    
    if verbose:
        print(f"Loaded {len(fisheye_cameras)} fisheye cameras")
        print(f"Loaded {len(perspective_config['crops'])} perspective crop configurations")
    
    # Setup paths
    scene_results_root = results_root / scene_name
    scene_results_dir =scene_results_root / "init_colmap_metric" / "renders"
    print("results dir:", scene_results_dir)

    if not scene_results_dir.exists():
        print(f"Error: Results directory not found: {scene_results_dir}")
        return
    
    # Process each split (train/test)
    for split in splits:
        split_dir = scene_results_dir / split
        
        if not split_dir.exists():
            if verbose:
                print(f"  Skipping {split} (directory not found)")
            continue
        
        if verbose:
            print(f"\nProcessing {split} split...")
        
        # Process each result type (gt_final, renders_final)
        for result_type in result_types:
            fisheye_result_dir = split_dir / result_type
            
            if not fisheye_result_dir.exists():
                if verbose:
                    print(f"  Skipping {result_type} (directory not found)")
                continue
            
            perspective_result_dir = split_dir / f"{result_type}_pers"
            perspective_result_dir.mkdir(exist_ok=True)
            
            if verbose:
                print(f"  Converting {result_type}...")
            
            # Process each perspective crop
            for crop_config in tqdm(perspective_config['crops'], desc=f"    Crops", disable=not verbose):
                crop_id = crop_config['crop_id']
                source_lens_id = crop_config['source_lens_id']
                lens_name = f"lens{source_lens_id:02d}"
                
                # Check if source lens directory exists
                source_lens_dir = fisheye_result_dir / lens_name
                if not source_lens_dir.exists():
                    continue
                
                # Get fisheye camera parameters
                fisheye_params = fisheye_cameras[source_lens_id]
                
                # Create output directory for this crop
                crop_output_dir = perspective_result_dir / f"crop{crop_id:02d}"
                crop_output_dir.mkdir(exist_ok=True)
                
                # Get all frame files from source lens
                frame_files = sorted(source_lens_dir.glob("frame_*.png"))
                
                # Process each frame
                for frame_file in frame_files:
                    # Load fisheye image
                    fisheye_img = cv2.imread(str(frame_file))
                    if fisheye_img is None:
                        continue
                    
                    # Generate perspective crop
                    perspective_crop = undistort_and_crop_fisheye(
                        fisheye_img=fisheye_img,
                        fisheye_params=fisheye_params,
                        crop_config=crop_config,
                    )
                    
                    if perspective_crop is None:
                        continue
                    
                    # Save perspective crop with same frame name
                    output_path = crop_output_dir / frame_file.name
                    cv2.imwrite(str(output_path), perspective_crop)
            
            if verbose:
                print(f"    Saved to {perspective_result_dir}")
    
    if verbose:
        print(f"\n Completed perspective conversion for {scene_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert rendered results from fisheye domain to perspective domain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert results for single scene
    python fish2pers.py --scene suite

    # Convert all scenes
    python fish2pers.py --all

    # Custom paths
    python fish2pers.py --scene suite \\
        --results_root ./results \\
        --data_root ./data/OmniFisheye_plus
        """
    )
    
    parser.add_argument(
        '--scene',
        type=str,
        help='Scene name to process (e.g., suite)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all scenes'
    )
    parser.add_argument(
        '--scenes',
        type=str,
        nargs='+',
        help='List of scene names to process'
    )
    parser.add_argument(
        '--results_root',
        type=Path,
        default=Path('results'),
        help='Root directory containing results'
    )
    parser.add_argument(
        '--data_root',
        type=Path,
        default=Path('data/OmniFisheye_plus'),
        help='Root directory containing scene data'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'test'],
        help='Splits to process (default: train test)'
    )
    parser.add_argument(
        '--result_types',
        type=str,
        nargs='+',
        default=['gt_final', 'renders_final'],
        help='Result types to process (default: gt_final renders_final)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print verbose output'
    )
    
    args = parser.parse_args()
    
    # Determine which scenes to process
    if args.all:
        # List all scenes in results_root
        if args.results_root.exists():
            scenes = [d.name for d in args.results_root.iterdir() if d.is_dir()]
        else:
            print(f"Error: Results root not found: {args.results_root}")
            return
    elif args.scenes:
        scenes = args.scenes
    elif args.scene:
        scenes = [args.scene]
    else:
        print("Error: Must specify --scene, --scenes, or --all")
        parser.print_help()
        return
    
    if args.verbose:
        print(f"Processing {len(scenes)} scene(s): {', '.join(scenes)}")
    
    # Process each scene
    for scene_name in scenes:
        try:
            # Paths for this scene
            scene_data_dir = args.data_root / scene_name
            camera_params_path = scene_data_dir / 'sparse' / '0' / 'cameras.txt'
            if scene_name == "lounge" or scene_name == "hall":
                perspective_config_path = Path('data') / 'camera_configs' / 'perspective_config_960.json'
            else:
                perspective_config_path = Path('data') / 'camera_configs' / 'perspective_config.json'
            
            # Check if required files exist
            if not camera_params_path.exists():
                print(f"Error: Camera parameters not found: {camera_params_path}")
                continue
            
            if not perspective_config_path.exists():
                print(f"Error: Perspective config not found: {perspective_config_path}")
                print(f"       Please run fish2pes.py first to generate perspective crops")
                continue
            
            # Convert results
            convert_results_to_perspective(
                scene_name=scene_name,
                results_root=args.results_root,
                camera_params_path=camera_params_path,
                perspective_config_path=perspective_config_path,
                splits=args.splits,
                result_types=args.result_types,
                verbose=args.verbose,
            )
        except Exception as e:
            print(f"Error processing scene {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if args.verbose:
        print(f"\n{'='*60}")
        print(f" Completed processing {len(scenes)} scene(s)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
