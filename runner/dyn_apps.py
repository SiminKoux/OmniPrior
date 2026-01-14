import os
import json
import time
import tqdm
import tyro
import viser
import imageio
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, Tuple, List
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure, 
    MultiScaleStructuralSimilarityIndexMeasure
)

from typing_extensions import Literal, assert_never

from dataloader.dataset import FisheyeDataset, Parser
from dataloader.sampler import CamBatchSampler

from modules.dyn_deform_module import DeformOptModule
from modules.dyn_init_module import create_splats_with_optimizers
from losses import (
    DynamicAwareGuidanceLoss,
    AdaptiveMonoRankingLoss,
    MetricDepthLoss,
    DepthSmoothLoss
)

from config import Config
from utils import (
    set_random_seed,
    scatter_map,
    extract_dyn_bboxes,
    rotmat_to_quat,
    quat_to_rotmat,
    slerp_quat
)

from gsplat import export_splats
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer

def create_camera_frustum(R, t, scale=0.1):
    """Create camera frustum vertices for visualization.
    
    Args:
        R: Rotation matrix [3, 3] (camera-to-world)
        t: Translation vector [3] (camera position in world)
        scale: Scale factor for frustum size
    
    Returns:
        pts_world: [5, 3] array of frustum vertices in world space
    """
    # Define the center and four front frustum corners in camera space
    # 4 ---- 3
    # |      |
    # |      |
    # 1 ---- 2
    pts_camera = np.array([
        [0, 0, 0],        # Camera center (origin)
        [-0.5, -0.5, 1],  # Point 1: bottom-left 
        [0.5, -0.5, 1],   # Point 2: bottom-right
        [0.5, 0.5, 1],    # Point 3: top-right
        [-0.5, 0.5, 1],   # Point 4: top-left
    ]) * scale

    # Transform to world space
    pts_world = pts_camera @ R.T + t

    return pts_world

def visualize_camera_positions(
    poses_list,
    frame_indices,
    output_path,
    scene_name="trajectory",
    colormap="Viridis"
):
    """Visualize camera positions with common colormap.
    
    Args:
        poses_list: List of camera-to-world poses [N, 4, 4]
        frame_indices: List of frame indices corresponding to each pose
        output_path: Path to save HTML visualization
        scene_name: Name for the visualization title
        colormap: Colormap name (e.g., 'Viridis', 'Plasma', 'Inferno', 'Cividis', 'Magma')
    """
    fig = go.Figure()
    
    poses = np.array(poses_list)
    frame_indices = np.array(frame_indices)
    n_frames = len(poses)
    
    # Extract camera positions
    cam_positions = poses[:, :3, 3]
    
    # Use specified colormap for common gradient
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    # Generate colors using specified colormap
    cmap = cm.get_cmap(colormap.lower())
    colors_hex = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        rgba = cmap(t)
        hex_color = mcolors.rgb2hex(rgba[:3])
        colors_hex.append(hex_color)
    
    # Plot camera positions as colored points with colorbar
    fig.add_trace(go.Scatter3d(
        x=cam_positions[:, 0],
        y=cam_positions[:, 1],
        z=cam_positions[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=frame_indices,  # Use frame indices for colorscale
            colorscale=colormap,
            colorbar=dict(
                title="Frame Index",
                thickness=15,
                len=0.7,
                x=1.02
            ),
            line=dict(width=0.5, color='white'),
            showscale=True
        ),
        text=[f'Frame {idx}' for idx in frame_indices],
        hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
        name='Camera Positions'
    ))
    
    # Draw camera path as a line
    fig.add_trace(go.Scatter3d(
        x=cam_positions[:, 0],
        y=cam_positions[:, 1],
        z=cam_positions[:, 2],
        mode='lines',
        line=dict(color='gray', width=3),
        name='Camera Path',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Add start and end markers with colors matching the gradient
    fig.add_trace(go.Scatter3d(
        x=[cam_positions[0, 0]],
        y=[cam_positions[0, 1]],
        z=[cam_positions[0, 2]],
        mode='markers',
        marker=dict(size=12, color=colors_hex[0], symbol='diamond', line=dict(width=2, color='white')),
        name=f'Start (Frame {frame_indices[0]})',
        showlegend=True,
        hovertemplate='<b>Start</b><br>Frame: %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
        text=[str(frame_indices[0])]
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[cam_positions[-1, 0]],
        y=[cam_positions[-1, 1]],
        z=[cam_positions[-1, 2]],
        mode='markers',
        marker=dict(size=12, color=colors_hex[-1], symbol='diamond', line=dict(width=2, color='white')),
        name=f'End (Frame {frame_indices[-1]})',
        showlegend=True,
        hovertemplate='<b>End</b><br>Frame: %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
        text=[str(frame_indices[-1])]
    ))
    
    # Update layout - keep grid lines but remove background faces
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            camera=dict(up=dict(x=0, y=1, z=0)),
            xaxis=dict(
                title='X',
                showbackground=False,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title='Y',
                showbackground=False,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            zaxis=dict(
                title='Z',
                showbackground=False,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
        ),
        title=f'Camera Positions ({colormap}) - {scene_name} ({n_frames} frames)',
        showlegend=True,
        legend=dict(x=0.7, y=0.95)
    )
    
    # Save to HTML
    fig.write_html(output_path)
    print(f"Camera positions ({colormap}) visualization saved to: {output_path}")

def visualize_camera_orientations(
    poses_list,
    frame_indices,
    output_path,
    scene_name="trajectory"
):
    """Visualize camera orientations with frustums colored by frame index.
    
    Args:
        poses_list: List of camera-to-world poses [N, 4, 4]
        frame_indices: List of frame indices corresponding to each pose
        output_path: Path to save HTML visualization
        scene_name: Name for the visualization title
    """
    fig = go.Figure()
    
    poses = np.array(poses_list)
    frame_indices = np.array(frame_indices)
    n_frames = len(poses)
    
    # Define base colors for each camera (in order)
    base_colors_rgb = [
        (240, 128, 128),  # lightcoral - Camera 1
        (100, 149, 237),  # cornflowerblue - Camera 2
        (255, 215, 0),    # gold - Camera 3
        (60, 179, 113),   # mediumseagreen - Camera 4
        (221, 160, 221),  # plum - Camera 5
        (255, 165, 0)     # orange - Camera 6
    ]
    
    # Generate colors with smooth gradient transitioning through camera colors
    colors_hex = []
    for i in range(n_frames):
        # Calculate position in gradient (0 to 5)
        position = (i / max(n_frames - 1, 1)) * 5  # Scale to 0-5 range
        
        # Find which two colors to interpolate between
        color_idx = int(position)
        if color_idx >= 5:
            color_idx = 4  # Last segment
        
        # Calculate interpolation weight
        weight = position - color_idx
        
        # Interpolate between current and next color
        color1 = base_colors_rgb[color_idx]
        color2 = base_colors_rgb[color_idx + 1]
        
        r = int(color1[0] * (1 - weight) + color2[0] * weight)
        g = int(color1[1] * (1 - weight) + color2[1] * weight)
        b = int(color1[2] * (1 - weight) + color2[2] * weight)
        
        hex_color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        colors_hex.append(hex_color)
    
    # Create smooth colorscale for the colorbar (6 colors)
    colorscale = [
        [0.0, 'rgb(240, 128, 128)'],      # lightcoral
        [0.2, 'rgb(100, 149, 237)'],      # cornflowerblue
        [0.4, 'rgb(255, 215, 0)'],        # gold
        [0.6, 'rgb(60, 179, 113)'],       # mediumseagreen
        [0.8, 'rgb(221, 160, 221)'],      # plum
        [1.0, 'rgb(255, 165, 0)']         # orange
    ]
    
    # Compute appropriate frustum scale based on camera distribution
    cam_positions = poses[:, :3, 3]
    avg_distance = np.mean(np.linalg.norm(
        cam_positions[1:] - cam_positions[:-1], axis=1
    )) if len(cam_positions) > 1 else 1.0
    frustum_scale = max(0.05, avg_distance * 0.3)
    
    # Add invisible scatter trace for colorbar (colorbar trick)
    fig.add_trace(go.Scatter3d(
        x=[cam_positions[0, 0]], y=[cam_positions[0, 1]], z=[cam_positions[0, 2]],
        mode='markers',
        marker=dict(
            size=0.001,  # Nearly invisible
            color=[frame_indices[0]],
            colorscale=colorscale,
            cmin=frame_indices[0],
            cmax=frame_indices[-1],
            colorbar=dict(
                title="Frame Index",
                thickness=15,
                len=0.7,
                x=1.02
            ),
            showscale=True
        ),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Plot each camera frustum
    legend_shown = set()
    for i, (pose, frame_idx, color) in enumerate(zip(poses, frame_indices, colors_hex)):
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # Create frustum
        pts = create_camera_frustum(R, t, scale=frustum_scale)
        
        # Frustum edges
        edges = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # From center to corners
            [1, 2], [2, 3], [3, 4], [4, 1]   # Square connecting corners
        ]
        
        # Show legend for evenly spaced frames
        show_in_legend = (i % max(n_frames // 10, 1) == 0) and ('frustum' not in legend_shown)
        if show_in_legend:
            legend_shown.add('frustum')
        
        # Add edges
        for edge_idx, edge in enumerate(edges):
            fig.add_trace(go.Scatter3d(
                x=pts[edge, 0],
                y=pts[edge, 1],
                z=pts[edge, 2],
                mode='lines',
                line=dict(color=color, width=2),
                name=f'Frame {frame_idx}' if edge_idx == 0 and show_in_legend else None,
                showlegend=(edge_idx == 0 and show_in_legend),
                hovertemplate=f'<b>Frame {frame_idx}</b><br>Pos: ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})<extra></extra>'
            ))
        
        # Add frustum front face (semi-transparent)
        fig.add_trace(go.Mesh3d(
            x=pts[[1, 2, 3, 4], 0],
            y=pts[[1, 2, 3, 4], 1],
            z=pts[[1, 2, 3, 4], 2],
            color=color,
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Draw camera path as a line
    fig.add_trace(go.Scatter3d(
        x=cam_positions[:, 0],
        y=cam_positions[:, 1],
        z=cam_positions[:, 2],
        mode='lines',
        line=dict(color='gray', width=3, dash='dash'),
        name='Camera Path',
        showlegend=True
    ))
    
    # Add start and end markers with colors matching the gradient
    fig.add_trace(go.Scatter3d(
        x=[cam_positions[0, 0]],
        y=[cam_positions[0, 1]],
        z=[cam_positions[0, 2]],
        mode='markers',
        marker=dict(size=10, color=colors_hex[0], symbol='diamond', line=dict(width=2, color='white')),
        name='Start',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[cam_positions[-1, 0]],
        y=[cam_positions[-1, 1]],
        z=[cam_positions[-1, 2]],
        mode='markers',
        marker=dict(size=10, color=colors_hex[-1], symbol='diamond', line=dict(width=2, color='white')),
        name='End',
        showlegend=True
    ))
    
    # Update layout - keep grid lines but remove background faces
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            camera=dict(up=dict(x=0, y=1, z=0)),
            xaxis=dict(
                title='X',
                showbackground=False,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title='Y',
                showbackground=False,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            zaxis=dict(
                title='Z',
                showbackground=False,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
        ),
        title=f'Camera Orientations (Frustums) - {scene_name} ({n_frames} frames)',
        showlegend=True,
        legend=dict(x=0.7, y=0.95)
    )
    
    # Save to HTML
    fig.write_html(output_path)
    print(f"Camera orientations visualization saved to: {output_path}")

class Runner:
    """Engine for training and testing."""

    def __init__(self, local_rank: int, world_rank, world_size: int, cfg: Config):
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        self.log_dir = f"{cfg.result_dir}/logs"

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            init_type=cfg.init_type,
            filter=cfg.filter
        )

        # self.parser.vis_rig_centers(save_path='rig_centers_3d.html')
        self.trainset = FisheyeDataset(
            parser=self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=True,
            pattern_length=10,
            train_length=7
        )
        self.valset = FisheyeDataset(
            parser=self.parser,
            split="val",
            patch_size=cfg.patch_size,
            load_depths=True,
            pattern_length=10,
            train_length=7
        )
        self.max_time_id = self.parser.max_time_id

        print("-" * 10 + f" [Initial] " + "-" * 10)
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print(f"Scene scale: {self.scene_scale}")
        xyz_max = self.parser.scene_stats['xyz_max']
        xyz_min = self.parser.scene_stats['xyz_min']
        self.eval_steps = cfg.eval_steps

        # Model
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print(f"Model initialized.")
        print(f"Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            print(f"absgrad: {self.cfg.strategy.absgrad}")
            print(f"revised_opacity: {self.cfg.strategy.revised_opacity}")
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        self.deform_optimizers = []
        if cfg.deform_opt:
            print("Using Transformation Optimization for Dynamics...")

            self.deformation = DeformOptModule(args=cfg.deform).to(self.device)
            self.deformation.st_deform.set_aabb(xyz_max, xyz_min)
            self.deformation.st_deform.freeze_mlps()
            self.deformation.st_deform.grid.freeze_planes()
            self.deform_optimizers = [
                torch.optim.Adam(
                    self.deformation.get_mlp_params(),
                    lr=cfg.deform.deform_opt_lr,
                ),
                torch.optim.Adam(
                    self.deformation.get_grid_params(),
                    lr=cfg.deform.grid_opt_lr,
                ),
            ]
            if world_size > 1:
                self.deformation = DDP(self.deformation)
        
        # Initalize schedulers
        self.schedulers = [ # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / cfg.max_steps)
            ),
        ]

        self.dyn_guide_regulator = DynamicAwareGuidanceLoss(config=cfg.deform)

        # Metric depth regularization
        self.metric_depth_regulator = MetricDepthLoss(
            loss_type=cfg.metric_loss_type,
            huber_delta=cfg.metric_huber_delta,
            start_step=cfg.metric_depth_reg_start,
            end_step=cfg.metric_depth_reg_end,
            warmup_ratio=cfg.metric_depth_warmup_ratio,
            decay_ratio=cfg.metric_depth_decay_ratio,
        )

        # Depth smoothness regularization (edge-aware structure prior)
        self.depth_smoothness_regulator = DepthSmoothLoss(
            start_step=cfg.depth_smooth_start,
            end_step=cfg.depth_smooth_end,
            max_weight=cfg.depth_smooth_reg,
            edge_lambda=cfg.smooth_edge_lambda,
            use_edge_aware=cfg.smooth_edge_aware,
        )

        # Monocular depth ranking
        self.mono_rank_regulator = AdaptiveMonoRankingLoss(
            margin=0.2,
            sample_pairs=1024,
            min_distance=10,
            pre_rank_end=self.cfg.pre_rank_end,
            ramp_up_end=self.cfg.rank_ramp_up,
            peak_end=self.cfg.rank_peak,
            decay_end=self.cfg.rank_fade_end,
            min_weight=cfg.rank_min_weight,
        )
        
        # Phase indicator variables
        self._is_gaussian_phase = True
        self._is_deform_phase = False

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.msssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            print("Using LPIPS Alex.")
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(self.device)
        elif cfg.lpips_net == "vgg":
            print("Using LPIPS VGG.")
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training")
        
        # Store final FPS from evaluation
        self.final_fps = None
    
    def get_dyn_gate(self):
        """Obtain the dynamicness probability of the current Gaussians."""
        prob = self.splats["dynamicness"] # [N]
        dyn_prob = torch.sigmoid(prob)    # [N], in [0, 1]
        # Modulate the dynamicness probability with the strategy state
        gate = torch.sigmoid(self.cfg.deform.gate_k * (dyn_prob - self.cfg.deform.gate_tau))
        return gate
    
    def get_dyn_prob(self):
        """Obtain the dynamicness probability of the current Gaussians."""
        prob = self.splats["dynamicness"] # [N]
        dyn_prob = torch.sigmoid(prob)    # [N], in [0, 1]
        return dyn_prob
    
    def rasterize_splats(
        self,
        camtoworlds: Tensor,   # [B, 4, 4]
        Ks: Tensor,            # [B, 3, 3]
        radial_coeffs: Tensor, # [B, 4]
        width: int,
        height: int,
        sh_degree: Optional[int] = None,
        masks: Optional[Tensor] = None,  # [B, H, W]
        deform_opt: bool = False,
        times: Optional[Tensor] = None,  # [B], default [6]
        render_mode: Optional[Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"]] = "RGB",
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        scales = self.splats["scales"] # [N, 3]
        opacities = self.splats["opacities"] # [N,]
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        if deform_opt and self._is_deform_phase:
            if times is not None:
                times = times.to(dtype=means.dtype) # [B]
                times = times.repeat(means.shape[0], 1) 
            else:
                times = torch.zeros_like(means[:, 0]) # [N, B]
            cano_dyn = self.splats["dynamicness"].unsqueeze(-1)  # [N, 1]
            
            # Learn offsets of each Gaussian Parameter
            deform_params, deformed_params = self.deformation(
                point=means, 
                scale=scales, 
                rotation=quats, 
                opacity=opacities, 
                app=colors, 
                times_sel=times, 
                cano_dyn=cano_dyn
            )

            # Get deformed Gaussian parameters
            means_temp = deformed_params.means
            scales_temp = deformed_params.scales
            quats_temp = deformed_params.rotations
            opacities_temp = deformed_params.opacities
            colors_temp = deformed_params.colors
        else:
            deform_params = None
            deformed_params = None
            means_temp, scales_temp, quats_temp = means, scales, quats
            opacities_temp, colors_temp = opacities, colors

        scales_temp = torch.exp(scales_temp)  # [N, 3]
        opacities_temp = torch.sigmoid(opacities_temp)  # [N,]

        renders, render_alphas, info = rasterization(
            means=means_temp,
            quats=quats_temp,
            scales=scales_temp,
            opacities=opacities_temp,
            colors=colors_temp,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            sh_degree=sh_degree,
            render_mode=render_mode,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=self.cfg.antialiased,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            radial_coeffs=(
                radial_coeffs 
                if isinstance(self.cfg.strategy, MCMCStrategy) and self.cfg.use_rad_coef
                else None
            ),
            **kwargs,
        )

        if masks is not None:
            renders[~masks] = 0

        return renders, render_alphas, info, deform_params, deformed_params

    @torch.no_grad()
    def vis_info(self, deform_params, means2d, gt_rgb, render_rgb, dyn_mask, H, W, save_dir, filename):
        os.makedirs(save_dir, exist_ok=True)

        # --- Prepare Inputs ---
        mean_mag = deform_params.means_offset.norm(dim=-1).detach().cpu().numpy()         # [N]
        color_mag = deform_params.colors_offset.norm(dim=(-2, -1)).detach().cpu().numpy() # [N]
        scales_offset = deform_params.scales_offset.norm(dim=-1).detach().cpu().numpy()         # [N]
        opacity_mag = deform_params.opacities_offset.abs().detach().cpu().numpy().flatten() # [N]
        quat = deform_params.rotations_offset  # [N, 4], unit quaternions
        w = quat[..., 0].clamp(-1 + 1e-6, 1- 1e-6) # Clamp the 'w' component to avoid domain errors with acos
        # The angle of rotation from a quaternion is 2 * acos(|w|)
        angle_rad = 2.0 * torch.acos(torch.abs(w))  # [N], in radians
        angle_deg = torch.rad2deg(angle_rad).cpu().numpy() # [N], in degrees

        means2d_np = means2d.detach().cpu().numpy()       # [B, N, 2]
        gt_rgb_np = gt_rgb.detach().cpu().numpy()         # [B, H, W, 3]
        render_rgb_np = render_rgb.detach().cpu().numpy() # [B, H, W, 3]
        dyn_mask_np = dyn_mask.detach().cpu().numpy()     # [B, H, W]

        # --------- Visualization per image ---------
        B = means2d_np.shape[0]
        for b in range(B):
            u, v = means2d_np[b, :, 0], means2d_np[b, :, 1]
            gt_rgb_img = gt_rgb_np[b]
            render_rgb_img = render_rgb_np[b]
            dyn_mask_img = dyn_mask_np[b]

            # Prepare all maps
            maps = {
                "Mean Magnitude": (scatter_map(H, W, u, v, mean_mag), 'plasma'),
                "Scale Magnitude": (scatter_map(H, W, u, v, scales_offset), 'plasma'),
                "Rotation Magnitude": (scatter_map(H, W, u, v, angle_deg), 'plasma'),
                "Opacity Magnitude": (scatter_map(H, W, u, v, opacity_mag), 'bone'),
                "SH Coefficient Magnitude": (scatter_map(H, W, u, v, color_mag), 'viridis')
            }

            total_maps = len(maps) + 1  # +1 for RGB
            n_cols = min(4, total_maps)
            n_rows = (total_maps + n_cols - 1) // n_cols
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            axs = axs.flatten()

            axs[0].imshow(gt_rgb_img)
            axs[0].set_title("Ground-Truth RGB")
            axs[0].axis('off')

            axs[1].imshow(render_rgb_img)
            axs[1].set_title("Rendered RGB")
            axs[1].axis('off')

            im2 = axs[2].imshow(dyn_mask_img, cmap='hot')
            axs[2].set_title("Per-frame Binary Mask")
            axs[2].axis('off')
            fig.colorbar(im2, ax=axs[2], shrink=0.7)

            for i, (name, (img, cmap)) in enumerate(maps.items(), start=3):
                im = axs[i].imshow(img, cmap=cmap)
                axs[i].set_title(f"{name}\nmin={img.min():.3f}, max={img.max():.3f}")
                axs[i].axis('off')
                fig.colorbar(im, ax=axs[i], shrink=0.7)

            for j in range(i + 1, len(axs)):
                axs[j].axis('off')

            plt.tight_layout()
            plt.savefig(f"{save_dir}/{filename}")
            plt.close()
    
    @torch.no_grad()
    def render_dyn_map(
        self,
        camtoworlds: Tensor,   # [B, 4, 4]
        Ks: Tensor,            # [B, 3, 3]
        radial_coeffs: Tensor, # [B, 4]
        masks: Tensor,         # [B, H, W]
        width: int,
        height: int,
        times: Optional[Tensor] = None,
        threshold: float = 0.5,
        apply_colormap: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Render dynamicness visualization maps.
        
        Applications:
        1. Segmentation: Binary dynamic/static masks
        2. Heatmap: Continuous dynamicness probability visualization
        3. Object tracking: Identify and track dynamic regions over time
        4. Dataset analysis: Understand scene dynamics distribution
        
        Args:
            camtoworlds: [B, 4, 4] camera poses
            Ks: [B, 3, 3] intrinsics
            radial_coeffs: [B, 4] fisheye distortion
            width, height: image dimensions
            times: [B] time indices for deformation
            threshold: float in [0, 1] for binary segmentation (default: 0.5)
            apply_colormap: whether to apply colormap for visualization
            
        Returns:
            dict with:
                'dynamicness_prob': [B, H, W] continuous probability in [0, 1]
                'dynamic_mask': [B, H, W] binary mask (1=dynamic, 0=static)
                'dynamicness_rgb': [B, H, W, 3] colormap visualization (if apply_colormap)
                'static_rgb': [B, H, W, 3] RGB with dynamic regions masked out
                'dynamic_rgb': [B, H, W, 3] RGB with static regions masked out
        """
        device = camtoworlds.device
        B = camtoworlds.shape[0]
        
        # Get Gaussian parameters
        means = self.splats["means"]  # [N, 3]
        quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        
        # Get dynamicness probability [N] in [0, 1]
        dyn_prob = self.get_dyn_prob()
        
        # Apply deformation if enabled and in deform phase
        if self.cfg.deform_opt and times is not None:
            times = times.to(dtype=means.dtype) # [B]
            times = times.repeat(means.shape[0], 1) 
            cano_dyn = self.splats["dynamicness"].unsqueeze(-1)  # [N, 1]
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
            
            # Learn offsets of each Gaussian Parameter
            deform_params, deformed_params = self.deformation(
                point=means, 
                scale=self.splats["scales"], 
                rotation=quats, 
                opacity=self.splats["opacities"], 
                app=colors, 
                times_sel=times, 
                cano_dyn=cano_dyn
            )
            
            means = deformed_params.means
            scales = torch.exp(deformed_params.scales)
            quats = deformed_params.rotations
            opacities = torch.sigmoid(deformed_params.opacities)
            colors = deformed_params.colors
            
            # Update dynamicness with offset
            dyn_prob = torch.sigmoid(self.splats["dynamicness"] + deform_params.dynamic_offset) # [N]
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
        
        # Convert dynamicness to "colors" for rasterization
        # Replicate probability across 3 channels: [N] -> [N, 1, 3]
        dyn_colors = dyn_prob.view(-1, 1, 1).expand(-1, 1, 3)  # [N, 1, 3]
        
        # Rasterize dynamicness as grayscale "RGB"
        dyn_renders, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=dyn_colors,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            sh_degree=0,  # No SH, just use raw colors
            render_mode="RGB",
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=self.cfg.antialiased,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            radial_coeffs=(
                radial_coeffs 
                if isinstance(self.cfg.strategy, MCMCStrategy) and self.cfg.use_rad_coef
                else None
            )
        )
        
        # Also render RGB for overlay visualization
        rgb_renders, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            sh_degree=self.cfg.sh_degree,
            render_mode="RGB",
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=self.cfg.antialiased,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            radial_coeffs=(
                radial_coeffs 
                if isinstance(self.cfg.strategy, MCMCStrategy) and self.cfg.use_rad_coef
                else None
            )
        )
        
        # Set invalid regions to Nan
        if masks is not None:
            dyn_renders[~masks] = float('nan')
            rgb_renders[~masks] = float('nan')
        
        # Extract probability map [B, H, W]
        dyn_map = dyn_renders[..., 0].clamp(0, 1)  # Use R channel

        # Create binary mask
        dynamic_mask = (dyn_map > threshold).float()  # [B, H, W]
        
        result = {
            'raw_means': means,
            'raw_quats': quats,
            'raw_scales': scales,
            'raw_opacities': opacities,
            'raw_dyn_prob': dyn_prob,          # [N]
            'dynamicness_prob': dyn_map,
            'dynamic_mask': dynamic_mask,
        }
        
        # Apply colormap for visualization
        if apply_colormap:
            # Convert to numpy for matplotlib colormap
            dyn_np = dyn_map.cpu().numpy()  # [B, H, W] with Nan for invalid regions
            colored = np.full((B, height, width, 3), np.nan)   # Initialize with Nan
            
            for i in range(B):
                # Apply "viridis" colormap: blue (static) -> yellow (dynamic)
                # matplotlib colormap automatically handles Nan (renders as white)
                # colored_frame = plt.cm.viridis(dyn_np[i])[..., :3]   # [H, W, 3]
                # colored_frame = plt.cm.inferno(dyn_np[i])[..., :3]   # [H, W, 3]
                colored_frame = plt.cm.plasma(dyn_np[i])[..., :3]   # [H, W, 3]

                # Preserve Nan regions
                valid_mask_np = ~np.isnan(dyn_np[i])
                colored[i][valid_mask_np] = colored_frame[valid_mask_np]
            
            result['dynamicness_rgb'] = torch.from_numpy(colored).float().to(device)
        
        # Create masked RGB views
        rgb = rgb_renders[..., :3].clamp(0, 1)  # [B, H, W, 3]
        
        # Static-only view (mask out dynamic regions)
        # static_rgb = rgb * (1 - dynamic_mask.unsqueeze(-1))
        static_rgb = torch.where(
            dynamic_mask.unsqueeze(-1) > 0.5,
            torch.full_like(rgb, float('nan')), # dynamic -> Nan
            rgb # static -> keep rgb
        )
        result['static_rgb'] = static_rgb
        
        # Dynamic-only view (mask out static regions)
        # dynamic_rgb = rgb * dynamic_mask.unsqueeze(-1)
        dynamic_rgb = torch.where(
            dynamic_mask.unsqueeze(-1) > 0.5,
            rgb, # dynamic -> keep rgb
            torch.full_like(rgb, float('nan')) # static -> Nan
        )
        result['dynamic_rgb'] = dynamic_rgb
        
        return result
    
    @torch.no_grad()
    def vis_dyn(
        self,
        data: Dict[str, Tensor],
        save_dir: str,
        filename: str = "dynamicness_vis.png",
        dyn_thresh: float = 0.5
    ):
        """
        Create a visualization grid showing:
            - Original RGB
            - Dynamicness probability heatmap
            - Binary segmentation mask
            - Static-only view
            - Dynamic-only view
            - GT dynamic mask
        Args:
            data: Dictionary from dataloader
            save_dir: Dictionary to save visualizations
            filename: Output filename
            dyn_thresh: Threshold for binary segmentation (default: 0.5)
        """
        os.makedirs(save_dir, exist_ok=True)

        camtoworlds = data["camtoworld"].to(self.device)
        Ks = data["K"].to(self.device)
        radial_coeffs = data["poly_coeffs"].to(self.device)
        pixels = data["image"].to(self.device) / 255.0
        times = data["time_id"].to(self.device) if "time_id" in data else None
        height, width = pixels.shape[1:3]
        masks = data["mask"].to(self.device) if "mask" in data else None

        # Render dynamicness maps
        dyn_result = self.render_dyn_map(
            camtoworlds=camtoworlds,
            Ks=Ks,
            radial_coeffs=radial_coeffs,
            masks=masks,
            width=width,
            height=height,
            times=times,
            threshold=dyn_thresh,
            apply_colormap=True,
        )

        # Move to CPU for visualization
        dyn_prob = dyn_result['dynamicness_prob'][0].cpu().numpy() # [H, W]
        dyn_mask = dyn_result['dynamic_mask'][0].cpu().numpy()     # [H, W]
        dyn_rgb = dyn_result['dynamicness_rgb'][0].cpu().numpy()   # [H, W, 3]
        static_rgb = dyn_result['static_rgb'][0].cpu().numpy()     # [H, W, 3]
        dynamic_rgb = dyn_result['dynamic_rgb'][0].cpu().numpy()   # [H, W, 3]
        
        # Get GT RGB and dynamic mask if available
        gt_rgb = pixels[0].cpu().numpy()  # [H, W, 3]
        has_gt_mask = "dynamic_masks" in data
        
        # Create visualization
        n_plots = 6 if has_gt_mask else 5
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        axs = axs.flatten()
        
        # 1. Original RGB
        axs[0].imshow(gt_rgb)
        axs[0].set_title("Original RGB", fontsize=14, fontweight='bold')
        axs[0].axis('off')
        
        # 2. Dynamicness heatmap
        im1 = axs[1].imshow(dyn_rgb, interpolation='none')
        axs[1].set_title(f"Dynamicness Heatmap\n(viridis: blue=static, yellow=dynamic)", fontsize=14, fontweight='bold')
        axs[1].axis('off')
        fig.colorbar(im1, ax=axs[1], fraction=0.046)
        
        # 3. Binary mask
        valid_mask = ~np.isnan(dyn_prob)
        dyn_ratio = (dyn_mask[valid_mask].mean() * 100) if valid_mask.any() else 0.0
        im2 = axs[2].imshow(dyn_mask, cmap='hot', vmin=0, vmax=1, interpolation='none')
        axs[2].set_title(f"Binary Segmentation (threshold={dyn_thresh})\nDynamic: {dyn_ratio:.1f}%", 
                        fontsize=14, fontweight='bold')
        axs[2].axis('off')
        fig.colorbar(im2, ax=axs[2], fraction=0.046)
        
        # 4. Static-only view
        axs[3].imshow(static_rgb, interpolation='none')
        axs[3].set_title("Static-Only View\n(Dynamic regions masked)", fontsize=14, fontweight='bold')
        axs[3].axis('off')
        
        # 5. Dynamic-only view
        axs[4].imshow(dynamic_rgb, interpolation='none')
        axs[4].set_title("Dynamic-Only View\n(Static regions masked)", fontsize=14, fontweight='bold')
        axs[4].axis('off')
        
        # 6. GT dynamic mask (if available)
        if has_gt_mask:
            gt_dyn_mask = data["dynamic_masks"][0].cpu().numpy()
            im3 = axs[5].imshow(gt_dyn_mask, cmap='hot', vmin=0, vmax=1)
            gt_dyn_ratio = gt_dyn_mask.mean() * 100
            axs[5].set_title(f"GT Dynamic Mask\nDynamic: {gt_dyn_ratio:.1f}%", 
                            fontsize=14, fontweight='bold')
            axs[5].axis('off')
            fig.colorbar(im3, ax=axs[5], fraction=0.046)
        else:
            # Probability distribution histogram
            valid_probs = dyn_prob[~np.isnan(dyn_prob)]
            if valid_probs.size > 0:
                axs[5].hist(valid_probs.flatten(), bins=50, color='steelblue', alpha=0.7)
                axs[5].axvline(dyn_thresh, color='red', linestyle='--', linewidth=2, label=f'Threshold={dyn_thresh}')
                axs[5].set_xlabel('Dynamicness Probability', fontsize=12)
                axs[5].set_ylabel('Pixel Count', fontsize=12)
                axs[5].set_title('Probability Distribution', fontsize=14, fontweight='bold')
                axs[5].legend()
                axs[5].grid(alpha=0.3)
            else:
                axs[5].text(0.5, 0.5, 'No valid data', ha='center', va='center', fontsize=14)
                axs[5].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{filename}", dpi=150, bbox_inches='tight')
        plt.close()
        
        valid_probs = dyn_prob[~np.isnan(dyn_prob)]
        print(f"Dynamicness visualization saved: {save_dir}/{filename}")
        print(f"  - Dynamic pixels: {dyn_ratio:.1f}%")
        if valid_probs.size > 0:
            print(f"  - Mean dynamicness: {valid_probs.mean():.3f}")
            print(f"  - Std dynamicness: {valid_probs.std():.3f}")
        else:
            print(f"  - No valid dynamicness data")
    
    @torch.no_grad()
    def vis_dynamic(
        self,
        data: Dict[str, Tensor],
        save_dir: str,
        filename: str = None,
        heatmap_path: str = None,
        distribution_path: str = None,
        dynamic_path: str = None,
        dpi: int = 300,
        dyn_threshold: float = 0.5,
        show_decomposition: bool = True,
        show_threshold: bool = True
    ):
        """
        Publication-ready dynamicness visualization for paper figures.
        
        Creates visualizations:
        - If filename is provided: Combined 2-row layout
        - If heatmap_path provided: Separate heatmap with colorbar
        - If distribution_path provided: Separate distribution histogram
        - If dynamic_path provided: Separate dynamic components image
        
        Args:
            data: Dictionary from dataloader
            save_dir: Directory to save visualizations
            filename: Output filename for combined visualization (optional)
            heatmap_path: Path to save heatmap only (optional)
            distribution_path: Path to save distribution only (optional)
            dynamic_path: Path to save dynamic components only (optional)
            dpi: Resolution for publication (default: 300)
            show_decomposition: Whether to show static/dynamic decomposition
            show_threshold: Whether to show threshold line in distribution (default: True)
        """
        os.makedirs(save_dir, exist_ok=True)

        camtoworlds = data["camtoworld"].to(self.device)
        Ks = data["K"].to(self.device)
        radial_coeffs = data["poly_coeffs"].to(self.device)
        pixels = data["image"].to(self.device) / 255.0
        times = data["time_id"].to(self.device) if "time_id" in data else None
        height, width = pixels.shape[1:3]
        masks = data["mask"].to(self.device) if "mask" in data else None

        # Render dynamicness maps
        dyn_result = self.render_dyn_map(
            camtoworlds=camtoworlds,
            Ks=Ks,
            radial_coeffs=radial_coeffs,
            masks=masks,
            width=width,
            height=height,
            times=times,
            threshold=dyn_threshold, # suite: 0.6, concert: 0.66
            apply_colormap=True,
        )

        # Move to CPU for visualization
        dyn_prob = dyn_result['dynamicness_prob'][0].cpu().numpy()  # [H, W]
        dyn_mask = dyn_result['dynamic_mask'][0].cpu().numpy()      # [H, W]
        dyn_rgb = dyn_result['dynamicness_rgb'][0].cpu().numpy()    # [H, W, 3]
        static_rgb = dyn_result['static_rgb'][0].cpu().numpy()      # [H, W, 3]
        dynamic_rgb = dyn_result['dynamic_rgb'][0].cpu().numpy()    # [H, W, 3]
        
        gt_rgb = pixels[0].cpu().numpy()  # [H, W, 3]
        has_gt_mask = "dynamic_masks" in data
        
        valid_mask = ~np.isnan(dyn_prob)
        dyn_ratio = (dyn_mask[valid_mask].mean() * 100) if valid_mask.any() else 0.0
        valid_probs = dyn_prob[~np.isnan(dyn_prob)]
        
        # Save separate heatmap if requested
        if heatmap_path is not None:
            os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
            fig_heat, ax_heat = plt.subplots(1, 1, figsize=(8, 6))
            im = ax_heat.imshow(dyn_rgb, interpolation='none')
            ax_heat.axis('off')
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = fig_heat.colorbar(sm, ax=ax_heat, fraction=0.046, pad=0.04)
            cbar.set_label('Dynamicness Probability', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(heatmap_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig_heat)
        
        # Save separate distribution if requested
        if distribution_path is not None:
            os.makedirs(os.path.dirname(distribution_path), exist_ok=True)
            if valid_probs.size > 0:
                fig_dist, ax_dist = plt.subplots(1, 1, figsize=(8, 6))
                ax_dist.hist(valid_probs.flatten(), bins=50, 
                             color='steelblue', alpha=0.7, edgecolor='black')
                if show_threshold:
                    ax_dist.axvline(0.5, color='red', linestyle='--', 
                                    linewidth=2, label=f'Threshold={dyn_threshold}')
                    ax_dist.legend(fontsize=12)
                ax_dist.set_xlabel('Dynamicness Probability', fontsize=14)
                ax_dist.set_ylabel('Pixel Count', fontsize=14)
                ax_dist.grid(alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(distribution_path, dpi=dpi, bbox_inches='tight')
                plt.close(fig_dist)
        
        # Save separate dynamic components if requested
        if dynamic_path is not None:
            os.makedirs(os.path.dirname(dynamic_path), exist_ok=True)
            fig_dyn, ax_dyn = plt.subplots(1, 1, figsize=(8, 6))
            ax_dyn.imshow(dynamic_rgb, interpolation='none')
            ax_dyn.axis('off')
            
            plt.tight_layout()
            plt.savefig(dynamic_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close(fig_dyn)
        
        # Save combined visualization if filename provided
        if filename is not None:
            # Create clean publication layout
            if show_decomposition:
                fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            else:
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            axs = axs.flatten() if show_decomposition else axs
            
            # Row 1: Input and predictions
            axs[0].imshow(gt_rgb)
            axs[0].set_title("(a) Input RGB", fontsize=16, fontweight='bold', pad=10)
            axs[0].axis('off')
            
            axs[1].imshow(dyn_rgb, interpolation='none')
            axs[1].set_title(f"(b) Learned Dynamicness\n(Abstract Feature, {dyn_ratio:.1f}% dynamic)", 
                            fontsize=16, fontweight='bold', pad=10)
            axs[1].axis('off')
            
            # Add colorbar for dynamicness
            sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axs[1], fraction=0.046, pad=0.04)
            cbar.set_label('Dynamicness Probability', fontsize=14)
            
            # Show weak supervision (GT mask) if available
            if has_gt_mask:
                gt_dyn_mask = data["dynamic_masks"][0].cpu().numpy()
                # Apply same colormap for consistency
                cmap = plt.cm.plasma
                gt_colored = cmap(gt_dyn_mask)[:, :, :3]
                axs[2].imshow(gt_colored)
                gt_ratio = gt_dyn_mask.mean() * 100
                axs[2].set_title(f"(c) Weak Supervision\n(Binary guidance, {gt_ratio:.1f}% dynamic)", 
                               fontsize=16, fontweight='bold', pad=10)
            else:
                axs[2].imshow(dyn_mask, cmap='RdYlGn_r', vmin=0, vmax=1, interpolation='none')
                axs[2].set_title(f"(c) Binary Segmentation\n", 
                               fontsize=16, fontweight='bold', pad=10)
            axs[2].axis('off')
            
            if show_decomposition:
                # Row 2: Scene decomposition
                axs[3].imshow(static_rgb, interpolation='none')
                axs[3].set_title("(d) Static Components", fontsize=16, fontweight='bold', pad=10)
                axs[3].axis('off')
                
                axs[4].imshow(dynamic_rgb, interpolation='none')
                axs[4].set_title("(e) Dynamic Components", fontsize=16, fontweight='bold', pad=10)
                axs[4].axis('off')
                
                # Show distribution histogram
                if valid_probs.size > 0:
                    axs[5].hist(valid_probs.flatten(), bins=50, color='steelblue', 
                               alpha=0.7, edgecolor='black')
                    if show_threshold:
                        axs[5].axvline(0.5, color='red', linestyle='--', 
                                       linewidth=2, label='Threshold')
                        axs[5].legend(fontsize=12)
                    axs[5].set_xlabel('Dynamicness Probability', fontsize=14)
                    axs[5].set_ylabel('Pixel Count', fontsize=14)
                    axs[5].set_title('(f) Distribution', fontsize=16, fontweight='bold', pad=10)
                    axs[5].grid(alpha=0.3)
                else:
                    axs[5].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{filename}", dpi=dpi, bbox_inches='tight')
            plt.close()
            
            print(f"Figure saved: {save_dir}/{filename}")
    
    @torch.no_grad()
    def render_6dof(
        self, 
        output_dir: str,
        num_frames: int = 120,
        radius: float = 3.0,
        height: float = 0.5,
        time_start: float = 0.0,
        time_end: float = 1.0,
        render_width: int = 520,
        render_height: int = 520,
        colormap: str = "Plasma"
    ):
        """
        Render video along a camera trajectory for dynamic scene.
        
        Args:
            output_dir: Output directory for rendered frames
            num_frames: Number of frames to render
            colormap: Colormap for time series visualization (e.g., 'Viridis', 'Plasma', 'Inferno', 'Cividis', 'Magma')
            radius: Orbit radius
            height: Camera height
            time_range: (start_time, end_time) for temporal interpolation
            resolution: (width, height) for rendering
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Rendering Trajectory Video")
        print(f"{'='*80}")
        print(f"Output: {output_dir}")
        print(f"Frames: {num_frames}")
        print(f"Time range: {(time_start, time_end)}")
        print(f"Resolution: {render_width}x{render_height}")
        print(f"{'='*80}\n")
        
        # Set to deformation phase
        self._is_deform_phase = self.cfg.deform_opt
        
        # Rename to avoid variable name collision
        img_width, img_height = render_width, render_height
        traj_height = height  # Save trajectory height parameter before it gets overwritten
        
        # Get scene center and scale from training cameras
        train_c2ws = torch.from_numpy(self.parser.camtoworlds.astype(np.float32))  # [N, 4, 4]
        scene_center = train_c2ws[:, :3, 3].mean(dim=0)  # Average camera positions [3]
        scene_extent = float((train_c2ws[:, :3, 3] - scene_center).norm(dim=1).mean())
        
        # If scene_extent is too small, use scene_scale
        if scene_extent < 0.1:
            scene_extent = float(self.scene_scale)
            print("Warning: Computed scene extent too small, using scene_scale")
        
        # Debug: print training camera statistics
        print(f"Training cameras: {len(train_c2ws)} cameras")
        print(f"Camera positions (first 3):")
        for i in range(min(3, len(train_c2ws))):
            print(f"  Camera {i}: {train_c2ws[i, :3, 3]}")
        
        # Scale trajectory radius based on scene scale
        actual_radius = float(radius * scene_extent) if scene_extent > 0 else float(radius)
        actual_height = float(scene_center[2] + traj_height * scene_extent)  # Height relative to scene center
        
        print(f"Scene center: {scene_center}")
        print(f"Scene extent: {scene_extent:.3f}")
        print(f"Trajectory radius: {actual_radius:.3f}")
        print(f"Camera height: {actual_height:.3f}\n")
        
        # For free camera trajectory with pinhole model, compute appropriate intrinsics
        # Use a reasonable field of view (~60 degrees)
        ref_focal = float(img_width * 0.8)  # ~60 degree FOV
        # ref_focal = float(img_width * 0.6)  # ~73 degree FOV for fisheye-like view
        # ref_focal = float(img_width * 0.5)  # ~90 degree FOV for fisheye-like view
        ref_cx = float(img_width / 2)
        ref_cy = float(img_height / 2)
        print(f"Using pinhole intrinsics for trajectory: focal={ref_focal:.1f}, cx={ref_cx:.1f}, cy={ref_cy:.1f}")
        
        # Get unique camera positions and group by lens (assuming 6 fisheye lenses in circular array)
        # Group cameras by their time/frame to get one representative camera per time
        unique_positions = []
        unique_c2ws = []
        unique_times = []
        
        # Get camera IDs if available
        camera_ids = getattr(self.parser, 'camera_ids', None)
        if camera_ids is not None:
            # Group by camera ID to get representative cameras for each lens
            unique_cam_ids = sorted(set(camera_ids))
            for cam_id in unique_cam_ids:
                # Find first occurrence of this camera
                idx = list(camera_ids).index(cam_id)
                unique_positions.append(train_c2ws[idx, :3, 3])
                unique_c2ws.append(train_c2ws[idx])
                unique_times.append(idx)
        else:
            # Fallback: detect unique positions by clustering
            for cam_idx in range(len(train_c2ws)):
                pos = train_c2ws[cam_idx, :3, 3]
                is_unique = True
                for existing_pos in unique_positions:
                    if (pos - existing_pos).norm() < 0.001:
                        is_unique = False
                        break
                if is_unique and len(unique_positions) < 10:  # Limit to prevent too many
                    unique_positions.append(pos)
                    unique_c2ws.append(train_c2ws[cam_idx])
                    unique_times.append(cam_idx)
        
        unique_positions = torch.stack(unique_positions)  # [K, 3]
        unique_c2ws = torch.stack(unique_c2ws)  # [K, 4, 4]
        num_keyframes = len(unique_positions)
        
        print(f"Found {num_keyframes} unique camera viewpoints (fisheye lenses)")
        print(f"Keyframe positions:")
        for i, pos in enumerate(unique_positions):
            print(f"  Lens {i+1}: {pos}")
        
        # Initialize lists to store trajectory data for visualization
        trajectory_poses = []
        trajectory_frame_indices = []
        
        frames = []
        for i in tqdm.tqdm(range(num_frames), desc="Rendering frames"):
            # Progress through all camera positions in a loop
            t = i / num_frames  # Normalized time [0, 1]
            
            # Divide time into segments: each lens gets a viewing period + transition
            viewing_ratio = 0.85  # 85% viewing, 15% transition
            
            # Determine current lens and progress within that lens's segment
            current_segment = t * num_keyframes  # Which lens segment we're in [0, num_keyframes]
            keyframe_idx = int(current_segment) % num_keyframes
            segment_progress = current_segment - int(current_segment)  # [0, 1] within segment
            
            if segment_progress < viewing_ratio:
                # VIEWING PHASE: Stay at current lens with 6DOF movements
                # Use current lens camera as base
                c2w_base = unique_c2ws[keyframe_idx]  # [4, 4]
                base_pos = c2w_base[:3, 3]  # [3]
                R_base = c2w_base[:3, :3]  # [3, 3]
                
                # Normalize segment progress to viewing period
                view_progress = segment_progress / viewing_ratio  # [0, 1] during viewing
                
                # 6DOF movements within this lens's viewing region:
                # 1. Zoom in/out (forward/backward)
                # 2. Left/right movement
                # 3. Up/down movement
                phase = 2 * np.pi * view_progress
                
                # Much larger movements to make 6DOF clearly visible
                # Smooth zoom: in and out once during viewing period
                zoom_amount = 0.5 * scene_extent * np.sin(phase)  # Increased from 0.1 to 0.3
                
                # Left/right: smooth sweep (2 cycles)
                sideways_amount = 0.2 * scene_extent * np.sin(2 * phase)  # Increased from 0.06 to 0.2
                
                # Up/down: gentle bob (1.5 cycles)
                vertical_amount = 0.25 * scene_extent * np.sin(1.5 * phase)  # Increased from 0.05 to 0.15
                
                # Apply offsets in lens's local frame
                forward_dir = -R_base[:, 2] # [3]
                right_dir = R_base[:, 0]    # [3]
                up_dir = R_base[:, 1]       # [3]
                
                cam_pos = (base_pos + 
                           zoom_amount * forward_dir +
                           sideways_amount * right_dir +
                           vertical_amount * up_dir)  # [3]
                
                # # Keep original orientation (mostly looking where the lens looks)
                # R_final = R_base  # [3, 3]

                # Compute rotation angles (in radians)
                # Different frequencies for visual variety:
                # - Yaw (horizontal): 2 cycles (left-right pan)
                # - Pitch (vertical): 1.5 cycles (up-down tilt)
                # - Roll (tilt): 1 cycle (barrel roll effect, kept small)

                rotation_amplitude = np.deg2rad(2.0)  # Base rotation amplitude: 2 degrees
                
                # Yaw (horizontal pan): Most prominent
                yaw_angle = rotation_amplitude * 1.0 * np.sin(2 * phase)    # 100% amplitude
                # Pitch (vertical tilt): Secondary
                pitch_angle = rotation_amplitude * 0.5 * np.sin(1.5 * phase)  # 50% amplitude
                # Roll (barrel tilt): Minimal
                roll_angle = rotation_amplitude * 0.01 * np.sin(phase)   # 1% amplitude
                
                # Convert Euler angles (ZYX convention) to rotation matrix
                # This is the DELTA rotation to apply on top of base orientation
                from scipy.spatial.transform import Rotation as R_scipy
                delta_rotation = R_scipy.from_euler('zyx', [yaw_angle, pitch_angle, roll_angle]).as_matrix()
                delta_rotation = torch.from_numpy(delta_rotation.astype(np.float32))
                
                # Apply rotation delta: R_final = R_delta @ R_base
                # This rotates the camera orientation relative to its base orientation
                R_final = torch.matmul(delta_rotation, R_base)
                
                
            else:
                # TRANSITION PHASE: Smoothly move to next lens
                next_keyframe_idx = (keyframe_idx + 1) % num_keyframes
                
                # Normalize transition progress
                trans_progress = (segment_progress - viewing_ratio) / (1.0 - viewing_ratio)  # [0, 1]
                
                # Smooth ease-in-ease-out for transition
                trans_weight = trans_progress * trans_progress * (3.0 - 2.0 * trans_progress)
                
                # Interpolate position
                pos1 = unique_c2ws[keyframe_idx][:3, 3]  # [3]
                pos2 = unique_c2ws[next_keyframe_idx][:3, 3]  # [3]
                cam_pos = pos1 * (1 - trans_weight) + pos2 * trans_weight  # [3]
                
                # Smooth rotation interpolation using SLERP
                R1 = unique_c2ws[keyframe_idx][:3, :3]      # [3, 3]
                R2 = unique_c2ws[next_keyframe_idx][:3, :3] # [3, 3]
                q1 = rotmat_to_quat(R1)  # [4]
                q2 = rotmat_to_quat(R2)  # [4]
                q_interp = slerp_quat(q1, q2, trans_weight)  # [4]
                R_final = quat_to_rotmat(q_interp.unsqueeze(0))[0]  # [3, 3]
            
            # Build camera-to-world matrix (torch tensor)
            c2w = torch.eye(4, dtype=torch.float32)
            c2w[:3, :3] = R_final  # Use original/interpolated rotation
            c2w[:3, 3] = cam_pos   # Use interpolated/offset position
            
            # Store for trajectory visualization
            trajectory_poses.append(c2w.cpu().numpy())
            trajectory_frame_indices.append(i)
            
            c2w_tensor = c2w.to(self.device)
            
            # Use reference intrinsics scaled to render resolution
            K = torch.tensor([
                [ref_focal, 0.0, ref_cx],
                [0.0, ref_focal, ref_cy],
                [0.0, 0.0, 1.0]
            ], dtype=torch.float32, device=self.device)
            
            # Temporal interpolation
            t_start, t_end = time_start, time_end
            current_time = float(t_start + (t_end - t_start) * (i / num_frames))
            times = torch.tensor([current_time], dtype=torch.float32, device=self.device)
            
            # Debug: print first frame info
            if i == 0:
                print(f"First frame info:")
                print(f"  Camera position: {cam_pos}")
                print(f"  Looking at scene center: {scene_center}")
                print(f"  Camera distance to center: {(cam_pos - scene_center).norm():.3f}")
                print(f"  Forward direction: {-c2w[:3, 2]}")
                print(f"  Right direction: {c2w[:3, 0]}")
                print(f"  Up direction: {c2w[:3, 1]}")
                print(f"  Time: {current_time:.3f}")
                print(f"  Number of Gaussians: {len(self.splats['means'])}")
                print(f"  Intrinsics - focal: {ref_focal:.1f}, cx: {ref_cx:.1f}, cy: {ref_cy:.1f}")
                print(f"  c2w matrix:\n{c2w}")
            
            # Render - use pinhole model for trajectory (fisheye not needed for free camera)
            renders, _, info, _, _ = self.rasterize_splats(
                camtoworlds=c2w_tensor[None],
                Ks=K[None],
                radial_coeffs=None,
                width=img_width,
                height=img_height,
                sh_degree=self.cfg.sh_degree,
                masks=None,
                deform_opt=self.cfg.deform_opt,
                times=times,
                render_mode="RGB"
            )
            
            # Debug: check if any gaussians are rendered
            if i == 0:
                visible_gs = (info["radii"] > 0).sum().item()
                print(f"  Visible Gaussians: {visible_gs}/{len(self.splats['means'])}")
                print(f"  Render min/max: {renders.min():.3f}/{renders.max():.3f}")
            
            # Convert to image
            colors = renders[..., :3].clamp(0.0, 1.0)
            img = (colors[0].cpu().numpy() * 255).astype(np.uint8)
            
            frames.append(img)
            
            # Save frame
            frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
            imageio.imwrite(frame_path, img)
        
        # Create video from rendered frames
        print("\nCreating video from rendered frames...")
        video_path = os.path.join(output_dir, "trajectory_video.mp4")
        
        # Use imageio to create video
        fps = 10
        try:
            imageio.mimwrite(video_path, frames, fps=fps, quality=8, macro_block_size=1)
            print(f"Video saved to: {video_path}")
        except Exception as e:
            print(f"Warning: Failed to create video: {e}")
            print("You can manually create video from frames using ffmpeg:")
            print(f"  ffmpeg -framerate {fps} -i {output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {video_path}")
        
        # Visualize camera trajectory
        print("\nGenerating camera trajectory visualizations...")
        
        # Determine scene name from output directory
        scene_name = os.path.basename(os.path.normpath(output_dir))
        
        # Generate visualizations: positions (camera colors) and positions (common colormap)
        positions_camera_html_path = os.path.join(output_dir, "camera_positions_camera_colors.html")
        positions_colormap_html_path = os.path.join(output_dir, f"camera_positions_{colormap.lower()}.html")
        # orientations_html_path = os.path.join(output_dir, "camera_orientations.html")
        
        # Visualize camera positions with specified colormap (all frames)
        visualize_camera_positions(
            poses_list=trajectory_poses,
            frame_indices=trajectory_frame_indices,
            output_path=positions_colormap_html_path,
            scene_name=scene_name,
            colormap=colormap
        )
        
        # # Visualize camera orientations with frustums (all frames) - commented out
        # visualize_camera_orientations(
        #     poses_list=trajectory_poses,
        #     frame_indices=trajectory_frame_indices,
        #     output_path=orientations_html_path,
        #     scene_name=scene_name
        # )
        
        print(f"\nTrajectory rendering complete!")
        print(f"  - Rendered frames: {len(frames)}")
        print(f"  - Output directory: {output_dir}")
        print(f"  - Video: {video_path}")
        print(f"  - Camera positions (camera colors): {positions_camera_html_path}")
        print(f"  - Camera positions ({colormap}): {positions_colormap_html_path}")
        # print(f"  - Camera orientations: {orientations_html_path}")

    @torch.no_grad()
    def render_motion_freeze(
        self,
        output_dir: str,
        freeze_time: int = 3,
        freeze_threshold: float = 0.5,
        render_width: int = 576,
        render_height: int = 768,
        time_start: int = 1,
        time_end: int = 7,
        num_frames: int = 30,
        camera_index: int = 0,
        filter_floaters: bool = True,
        opacity_threshold: float = -2.0,
        freeze_mode: str = "static",
        camera_time_sync: bool = True,
        min_opacity_moving: float = -1.0,
        min_opacity_frozen: float = -3.0,
        scale_damping: float = 1.0,
        scale_damping_frozen: float = 1.0,
        use_motion_magnitude: bool = False,
        motion_threshold: float = 0.01,
        hybrid_weight: float = 0.5,
        background_priority: bool = False,
        fov_scale: float = 1.0,
        use_dataset_intrinsics: bool = True
    ):
        """
        Freeze dynamic objects at a specific timestamp while others continue moving.
        
        This application demonstrates selective temporal control by freezing
        high-dynamicness Gaussians at a chosen time while low-dynamicness
        Gaussians continue their temporal evolution.
        
        Args:
            output_dir: Output directory for rendered frames
            freeze_time: Integer time frame at which to freeze dynamic objects (e.g., 3, 4, 5)
            freeze_threshold: Dynamicness threshold (objects with prob > threshold are frozen)
            render_width: Output image width
            render_height: Output image height
            time_start: Start integer time frame for temporal range (e.g., 1)
            time_end: End integer time frame for temporal range (e.g., 7)
            num_frames: Number of frames to render
            camera_index: Which training camera view to use (0-indexed, corresponds to lens 1-6)
            filter_floaters: Whether to filter out low-opacity floaters
            opacity_threshold: Opacity threshold in logit space (default: -2.0, sigmoid ? 0.12)
            freeze_mode: "static" = freeze low-dynamicness (static background), animate high-dynamicness
                        "dynamic" = freeze high-dynamicness (moving objects), animate low-dynamicness
            camera_time_sync: If True, camera moves through time following recorded trajectory.
                            If False, camera stays fixed at one position (first frame).
            min_opacity_moving: Additional opacity filter for moving Gaussians to reduce floaters
                              (logit space, default: -1.0, sigmoid ? 0.27). More aggressive than opacity_threshold.
            min_opacity_frozen: Separate opacity filter for frozen Gaussians (default: -3.0, sigmoid ? 0.047).
                              Lower (more permissive) than min_opacity_moving to preserve background detail.
                              Use same as min_opacity_moving if static background still has artifacts.
            scale_damping: Scale damping factor for moving Gaussians (0.5-1.0). Lower values shrink
                          Gaussians to reduce artifacts from view interpolation. Default: 1.0 (no damping).
            scale_damping_frozen: Scale damping for frozen Gaussians (0.5-1.0). Usually keep at 1.0 to
                                 preserve background detail. Lower if frozen layer has artifacts.
            use_motion_magnitude: If True, use actual motion magnitude instead of learned dynamicness.
                                 Measures real deformation between time frames. More reliable than
                                 learned dynamicness when predictions are imperfect.
            motion_threshold: Motion magnitude threshold in world units (default: 0.01). Gaussians that
                            move less than this are considered static. Typical range: 0.005-0.05.
            hybrid_weight: When use_motion_magnitude=True, blend dynamicness and motion (0-1).
                          0.0 = pure motion magnitude, 0.5 = equal blend, 1.0 = pure dynamicness.
                          Default: 0.5 (balanced hybrid).
            background_priority: If True, composite frozen layer on TOP of moving layer.
                               This ensures static background is fully opaque and dominates the image.
                               Use when background appears too transparent. Default: False.
            fov_scale: Field of view scale factor (default: 1.0). Values > 1.0 zoom out (wider FOV),
                      values < 1.0 zoom in (narrower FOV). Adjusts focal length: focal *= fov_scale.
                      Example: 0.8 for ~60�, 0.6 for ~73�, 0.5 for ~90� FOV.
            use_dataset_intrinsics: If True, use camera intrinsics from dataset (fisheye parameters).
                                   If False, use pinhole model with adjustable FOV (fov_scale applies).
                                   Default: True (use dataset cameras). Set to False for custom FOV control.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Motion Freeze Application")
        print(f"{'='*80}")
        print(f"Output: {output_dir}")
        print(f"Freeze time: {freeze_time} (normalized: {freeze_time / self.max_time_id:.4f})")
        print(f"Freeze mode: {freeze_mode}")
        print(f"Dynamicness threshold: {freeze_threshold}")
        print(f"Resolution: {render_width}x{render_height}")
        print(f"Time range: [{time_start}, {time_end}] (normalized: [{time_start / self.max_time_id:.4f}, {time_end / self.max_time_id:.4f}])")
        print(f"Frames: {num_frames}")
        print(f"Camera index: {camera_index}")
        print(f"Camera mode: {'Moving through time' if camera_time_sync else 'Fixed at first frame'}")
        print(f"{'='*80}\n")
        
        # Set to deformation phase
        self._is_deform_phase = self.cfg.deform_opt
        
        if not self.cfg.deform_opt:
            print("Error: Deformation module not enabled. Please train with deform_opt=True.")
            return
        
        if not hasattr(self.splats, 'dynamicness'):
            print("Error: Model does not have dynamicness module.")
            return
        
        # FOV and intrinsics setup
        if not use_dataset_intrinsics:
            # Use custom pinhole model with adjustable FOV
            # Compute intrinsics based on fov_scale and render resolution
            # Use different focal lengths to ensure equal FOV in both directions
            ref_focal_x = float(render_width * fov_scale)
            ref_focal_y = float(render_height * fov_scale)
            ref_cx = float(render_width / 2)
            ref_cy = float(render_height / 2)
            
            custom_K = torch.tensor([[
                [ref_focal_x, 0.0, ref_cx],
                [0.0, ref_focal_y, ref_cy],
                [0.0, 0.0, 1.0]
            ]], dtype=torch.float32, device=self.device)  # shape: [1, 3, 3]
            
            # Disable fisheye distortion for pinhole model
            custom_radial_coeffs = None
            
            print(f"\nCustom Pinhole Camera (Equal FOV):")
            print(f"  Focal length: fx={ref_focal_x:.1f}px, fy={ref_focal_y:.1f}px")
            print(f"  Principal point: ({ref_cx:.1f}, {ref_cy:.1f})")
            print(f"  FOV scale: {fov_scale:.2f}")
            
            # Calculate FOV (should be equal for both directions)
            fov_horizontal_deg = 2 * np.arctan(render_width / (2 * ref_focal_x)) * 180 / np.pi
            fov_vertical_deg = 2 * np.arctan(render_height / (2 * ref_focal_y)) * 180 / np.pi
            print(f"  FOV: {fov_horizontal_deg:.1f} (H) = {fov_vertical_deg:.1f} (V) - Equal in both directions")
        else:
            custom_K = None
            custom_radial_coeffs = None
            if fov_scale != 1.0:
                print(f"\nWarning: fov_scale={fov_scale} will be ignored (use_dataset_intrinsics=True)")
                print(f"  To use custom FOV, set use_dataset_intrinsics=False")
        
        # Get dynamicness probabilities
        dyn_probs = torch.sigmoid(self.splats['dynamicness']).squeeze(-1)  # [N]

        # Diagnostic: Print dynamicness distribution
        print(f"\n{'='*80}")
        print(f"Dynamicness Distribution Analysis:")
        print(f"{'='*80}")
        print(f"Total Gaussians: {len(dyn_probs):,}")
        print(f"Dynamicness statistics:")
        print(f"  Mean: {dyn_probs.mean().item():.3f}")
        print(f"  Std: {dyn_probs.std().item():.3f}")
        print(f"  Min: {dyn_probs.min().item():.3f}")
        print(f"  Max: {dyn_probs.max().item():.3f}")
        print(f"  Median: {dyn_probs.median().item():.3f}")
        
        # Percentile analysis
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"\nPercentiles:")
        for p in percentiles:
            val = torch.quantile(dyn_probs, p/100.0).item()
            print(f"  {p}th: {val:.3f}")
        
        # Count Gaussians at different threshold levels
        print(f"\nGaussians above different thresholds:")
        for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
            count = (dyn_probs > thresh).sum().item()
            pct = count / len(dyn_probs) * 100
            print(f"  >{thresh:.1f}: {count:,} ({pct:.1f}%)")
        print(f"{'='*80}\n")

        if use_motion_magnitude:
            print(f"\nComputing motion magnitude and time-varying dynamicness...")
            
            t1_normalized = time_start / self.max_time_id
            t2_normalized = time_end / self.max_time_id
            
            means_canonical = self.splats['means']
            scales_canonical = self.splats['scales']
            quats_canonical = self.splats['quats']
            opacities_canonical = self.splats['opacities']
            colors_canonical = torch.cat([self.splats['sh0'], self.splats['shN']], dim=1)
            
            N = len(means_canonical)
            
            # Deform to time_start
            times_t1 = torch.full((N, 1), t1_normalized, device=self.device, dtype=torch.float32)
            deform_params_t1, deformed_t1 = self.deformation(
                point=means_canonical,
                scale=scales_canonical,
                rotation=F.normalize(quats_canonical, dim=-1),
                opacity=opacities_canonical,
                app=colors_canonical,
                times_sel=times_t1,
                cano_dyn=dyn_probs.unsqueeze(-1) if dyn_probs.dim() == 1 else dyn_probs
            )
            
            # Deform to time_end
            times_t2 = torch.full((N, 1), t2_normalized, device=self.device, dtype=torch.float32)
            deform_params_t2, deformed_t2 = self.deformation(
                point=means_canonical,
                scale=scales_canonical,
                rotation=F.normalize(quats_canonical, dim=-1),
                opacity=opacities_canonical,
                app=colors_canonical,
                times_sel=times_t2,
                cano_dyn=dyn_probs.unsqueeze(-1) if dyn_probs.dim() == 1 else dyn_probs
            )
            
            # Compute motion magnitude (spatial displacement)
            motion_magnitude = torch.norm(deformed_t2.means - deformed_t1.means, dim=-1)  # [N]
            
            # STABILITY FILTERING (keep existing noise reduction)
            opacity_t1 = torch.sigmoid(deformed_t1.opacities.squeeze(-1))
            opacity_t2 = torch.sigmoid(deformed_t2.opacities.squeeze(-1))
            opacity_variance = torch.abs(opacity_t1 - opacity_t2)
            
            scale_t1 = torch.exp(deformed_t1.scales).mean(dim=-1)
            scale_t2 = torch.exp(deformed_t2.scales).mean(dim=-1)
            scale_variance = torch.abs(scale_t1 - scale_t2) / (scale_t1 + 1e-6)
            
            stability_score = 1.0 - (opacity_variance + scale_variance * 0.5)
            motion_magnitude_filtered = motion_magnitude * (2.0 - stability_score.clamp(0, 1))
            motion_magnitude = motion_magnitude_filtered
            
            # Normalize motion magnitude
            motion_max = motion_magnitude.max().item()
            motion_normalized = motion_magnitude / (motion_max + 1e-10)
            
            # Extract time-varying dynamicness (canonical + learned time-specific offset)
            if deform_params_t1.dynamic_offset is not None:
                dyn_t1 = torch.sigmoid(
                    self.splats["dynamicness"] + deform_params_t1.dynamic_offset
                ).squeeze(-1)  # [N] - Time-specific dynamicness at t1
                dyn_t2 = torch.sigmoid(
                    self.splats["dynamicness"] + deform_params_t2.dynamic_offset
                ).squeeze(-1)  # [N] - Time-specific dynamicness at t2
                
                # Compute dynamicness change over time (temporal confidence variation)
                dyn_change = torch.abs(dyn_t2 - dyn_t1)  # [N]
                
                # Correlation analysis for diagnostic purposes
                correlation_spatial = torch.corrcoef(torch.stack([dyn_t1, motion_normalized]))[0, 1].item()
                correlation_temporal = torch.corrcoef(torch.stack([dyn_change, motion_normalized]))[0, 1].item()
                
                print(f"  Time-varying dynamicness analysis:")
                print(f"    Dynamicness at t={time_start}: mean={dyn_t1.mean().item():.3f}, std={dyn_t1.std().item():.3f}")
                print(f"    Dynamicness at t={time_end}: mean={dyn_t2.mean().item():.3f}, std={dyn_t2.std().item():.3f}")
                print(f"    Dynamicness change: mean={dyn_change.mean().item():.3f}, max={dyn_change.max().item():.3f}")
                print(f"    Correlation (dyn_t1 vs motion_magnitude): {correlation_spatial:.3f}")
                print(f"    Correlation (dyn_change vs motion_magnitude): {correlation_temporal:.3f}")
                
                # Hybrid formula: Combine time-varying dynamicness + motion magnitude + dynamicness change
                # This captures three complementary signals:
                #   1. Time-specific learned confidence (dyn_t1) - what network believes at this time
                #   2. Spatial displacement (motion_magnitude) - actual measured movement
                #   3. Temporal confidence variation (dyn_change) - how belief changes over time
                if hybrid_weight > 0.0:
                    combined_score = (
                        hybrid_weight * dyn_t1 +                           # Time-specific dynamicness (learned)
                        (1 - hybrid_weight) * 0.7 * motion_normalized +    # Spatial motion (measured)
                        (1 - hybrid_weight) * 0.3 * dyn_change             # Temporal variation (learned)
                    )
                    print(f"\n  Hybrid Filtering Strategy:")
                    print(f"    {hybrid_weight*100:.0f}% time-varying dynamicness (learned confidence at t={time_start})")
                    print(f"    {(1-hybrid_weight)*70:.0f}% motion magnitude (measured spatial displacement)")
                    print(f"    {(1-hybrid_weight)*30:.0f}% dynamicness change (temporal confidence variation)")
                    print(f"  Rationale: Combines learned beliefs with measured motion for robust filtering")
                    
                    # Provide guidance based on correlations
                    if correlation_spatial < 0.3 and hybrid_weight > 0.5:
                        print(f"  Warning: Low dyn_t1 vs motion correlation ({correlation_spatial:.2f})")
                        print(f"     Consider reducing hybrid_weight to rely more on motion_magnitude")
                    if correlation_temporal < 0.2:
                        print(f"  Note: Low dyn_change vs motion correlation ({correlation_temporal:.2f})")
                        print(f"     Temporal signal may be noisy - this is normal if dynamicness is stable")
                else:
                    combined_score = motion_normalized
                    print(f"\n  Pure Motion-Based Filtering (hybrid_weight=0.0)")
                    print(f"    Using only measured spatial displacement, ignoring learned dynamicness")
            else:
                # Fallback: No dynamicness offset learned (older model checkpoint)
                print(f"  Warning: Deformation network has no dynamicness offset")
                print(f"  Falling back to canonical dynamicness + motion_magnitude")
                if hybrid_weight > 0.0:
                    combined_score = hybrid_weight * dyn_probs + (1 - hybrid_weight) * motion_normalized
                    print(f"  Using: {hybrid_weight*100:.0f}% canonical dynamicness + {(1-hybrid_weight)*100:.0f}% motion")
                else:
                    combined_score = motion_normalized
                    print(f"  Using: 100% motion magnitude only")
        else:
            # Original behavior: Use canonical dynamicness only
            motion_magnitude = None
            combined_score = dyn_probs
            print(f"\nDynamicness-Only Filtering:")
            print(f"  Using canonical dynamicness (no time-varying offset)")
            print(f"  This ignores temporal variations in learned confidence")
            print(f"  Consider using use_motion_magnitude=True for more robust filtering")
        
        # Determine freeze mask based on mode
        if freeze_mode == "static":
            # Freeze STATIC (low dynamicness), animate DYNAMIC (high dynamicness)
            # This is the typical use case: background stays still, moving objects animate
            if use_motion_magnitude:
                # Use motion threshold instead of dynamicness threshold
                freeze_mask = motion_magnitude <= motion_threshold
                print(f"Freeze mode: STATIC (motion-based) - Freezing Gaussians with motion <={motion_threshold:.4f}")
            else:
                freeze_mask = combined_score <= freeze_threshold
                print(f"Freeze mode: STATIC - Freezing low-dynamicness objects (<={freeze_threshold}), animating high-dynamicness objects (>{freeze_threshold})")
        elif freeze_mode == "dynamic":
            # Freeze DYNAMIC (high dynamicness), animate STATIC (low dynamicness)
            # Special effect: freeze moving objects, animate background (unusual but creative)
            if use_motion_magnitude:
                freeze_mask = motion_magnitude > motion_threshold
                print(f"Freeze mode: DYNAMIC (motion-based) - Freezing Gaussians with motion >{motion_threshold:.4f}")
            else:
                freeze_mask = combined_score > freeze_threshold
                print(f"Freeze mode: DYNAMIC - Freezing high-dynamicness objects (>{freeze_threshold}), animating low-dynamicness objects (?{freeze_threshold})")
        else:
            raise ValueError(f"Invalid freeze_mode: {freeze_mode}. Must be 'static' or 'dynamic'")
        
        num_total = len(dyn_probs)
        num_frozen = freeze_mask.sum().item()
        num_moving = num_total - num_frozen
        
        # Optional: Filter out low-opacity floaters to reduce artifacts
        valid_mask = torch.ones(num_total, dtype=torch.bool, device=self.device)
        if filter_floaters:
            opacities = self.splats['opacities'].squeeze(-1)
            opacity_mask = opacities > opacity_threshold  # Keep Gaussians with opacity > threshold
            num_filtered = (~opacity_mask).sum().item()
            valid_mask = opacity_mask
            
            # Update masks to account for filtering
            freeze_mask = freeze_mask & valid_mask
            num_frozen = freeze_mask.sum().item()
            num_moving = valid_mask.sum().item() - num_frozen
            
            print(f"\nFloater Filtering:")
            print(f"  Opacity threshold: {opacity_threshold:.2f} (logit) = {torch.sigmoid(torch.tensor(opacity_threshold)):.3f} (probability)")
            print(f"  Filtered out: {num_filtered:,} Gaussians ({num_filtered/num_total*100:.1f}%)")
            print(f"  Remaining: {valid_mask.sum().item():,} Gaussians")
        
        # Asymmetric filtering: Different thresholds for moving vs frozen Gaussians
        # This preserves background quality while aggressively filtering dynamic artifacts
        frozen_mask_additional = torch.ones(num_total, dtype=torch.bool, device=self.device)
        moving_mask_additional = torch.ones(num_total, dtype=torch.bool, device=self.device)
        
        if min_opacity_frozen > opacity_threshold or min_opacity_moving > opacity_threshold:
            print(f"\nAsymmetric Artifact Reduction:")
            opacities = self.splats['opacities'].squeeze(-1)
            
            # Filter frozen Gaussians (background) - more permissive
            if min_opacity_frozen > opacity_threshold:
                frozen_opacity_mask = opacities > min_opacity_frozen
                frozen_mask_additional = (~freeze_mask) | frozen_opacity_mask  # Keep all moving, filter frozen
                num_frozen_filtered = (~frozen_mask_additional & freeze_mask & valid_mask).sum().item()
                if num_frozen_filtered > 0:
                    print(f"  Frozen (static) layer:")
                    print(f"    Opacity threshold: {min_opacity_frozen:.2f} (logit) = {torch.sigmoid(torch.tensor(min_opacity_frozen)):.3f} (prob)")
                    print(f"    Filtered: {num_frozen_filtered:,} Gaussians")
            
            # Filter moving Gaussians (dynamic) - more aggressive
            if min_opacity_moving > opacity_threshold:
                moving_opacity_mask = opacities > min_opacity_moving
                moving_mask_additional = freeze_mask | moving_opacity_mask  # Keep all frozen, filter moving
                num_moving_filtered = (~moving_mask_additional & (~freeze_mask) & valid_mask).sum().item()
                if num_moving_filtered > 0:
                    print(f"  Moving (dynamic) layer:")
                    print(f"    Opacity threshold: {min_opacity_moving:.2f} (logit) = {torch.sigmoid(torch.tensor(min_opacity_moving)):.3f} (prob)")
                    print(f"    Filtered: {num_moving_filtered:,} Gaussians")
            
            # Combine both filters
            valid_mask = valid_mask & frozen_mask_additional & moving_mask_additional
            freeze_mask = freeze_mask & valid_mask
            num_frozen = freeze_mask.sum().item()
            num_moving = valid_mask.sum().item() - num_frozen
        
        if scale_damping < 1.0 or scale_damping_frozen < 1.0:
            print(f"\nScale Damping:")
            if scale_damping < 1.0:
                print(f"  Moving Gaussians: {scale_damping:.2f}x (reduce dynamic artifacts)")
            if scale_damping_frozen < 1.0:
                print(f"  Frozen Gaussians: {scale_damping_frozen:.2f}x (reduce static artifacts)")
        
        print(f"\nGaussian Statistics:")
        print(f"  Total Gaussians: {num_total:,}")
        if freeze_mode == "static":
            if use_motion_magnitude:
                print(f"  Frozen Gaussians (motion<={motion_threshold:.4f}): {num_frozen:,} ({num_frozen/num_total*100:.1f}%)")
                print(f"  Moving Gaussians (motion>{motion_threshold:.4f}): {num_moving:,} ({num_moving/num_total*100:.1f}%)")
            else:
                print(f"  Frozen Gaussians (dyn?{freeze_threshold}, static): {num_frozen:,} ({num_frozen/num_total*100:.1f}%)")
                print(f"  Moving Gaussians (dyn>{freeze_threshold}, dynamic): {num_moving:,} ({num_moving/num_total*100:.1f}%)")
        else:
            if use_motion_magnitude:
                print(f"  Frozen Gaussians (motion>{motion_threshold:.4f}): {num_frozen:,} ({num_frozen/num_total*100:.1f}%)")
                print(f"  Moving Gaussians (motion<={motion_threshold:.4f}): {num_moving:,} ({num_moving/num_total*100:.1f}%)")
            else:
                print(f"  Frozen Gaussians (dyn>{freeze_threshold}, dynamic): {num_frozen:,} ({num_frozen/num_total*100:.1f}%)")
                print(f"  Moving Gaussians (dyn<={freeze_threshold}, static): {num_moving:,} ({num_moving/num_total*100:.1f}%)")
        print(f"\nDynamicness Distribution:")
        print(f"  Min: {dyn_probs.min().item():.3f}")
        print(f"  Max: {dyn_probs.max().item():.3f}")
        print(f"  Mean: {dyn_probs.mean().item():.3f}")
        print(f"  Median: {dyn_probs.median().item():.3f}")
        print(f"  >0.9: {(dyn_probs > 0.9).sum().item():,} ({(dyn_probs > 0.9).sum().item()/num_total*100:.1f}%)")
        print(f"  >0.7: {(dyn_probs > 0.7).sum().item():,} ({(dyn_probs > 0.7).sum().item()/num_total*100:.1f}%)")
        print(f"  >0.5: {(dyn_probs > 0.5).sum().item():,} ({(dyn_probs > 0.5).sum().item()/num_total*100:.1f}%)")
        print(f"  >0.3: {(dyn_probs > 0.3).sum().item():,} ({(dyn_probs > 0.3).sum().item()/num_total*100:.1f}%)")
        if use_motion_magnitude:
            print(f"\nMotion Magnitude vs Dynamicness Correlation:")
            # Compute correlation between motion and dynamicness
            motion_norm = motion_normalized
            correlation = torch.corrcoef(torch.stack([dyn_probs, motion_norm]))[0, 1].item()
            print(f"  Pearson correlation: {correlation:.3f}")
            if correlation < 0.5:
                print(f"  Low correlation - learned dynamicness may be unreliable!")
                print(f"  Consider using pure motion-based filtering (hybrid_weight=0.0)")
        
        # Get reference camera by finding all frames with the desired camera_id
        # Dataset structure: images are sorted by name, which groups by lens then by frame
        # e.g., lens01/frame_0001.png, lens01/frame_0002.png, ..., lens02/frame_0001.png, ...
        
        # Get camera_ids from parser (1-indexed: 1, 2, 3, 4, 5, 6 for 6 lenses)
        target_camera_id = camera_index + 1  # Convert 0-indexed to 1-indexed
        
        # Collect all frames for this camera_id, organized by time
        camera_frames = {}  # {time_id: data}
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=1, shuffle=False, num_workers=0
        )
        
        print(f"  Collecting frames for camera_id={target_camera_id} (lens {target_camera_id})...")
        for data in trainloader:
            cam_id = data.get("camera_id")
            if cam_id is not None:
                if isinstance(cam_id, torch.Tensor):
                    cam_id = cam_id.item()
                if cam_id == target_camera_id:
                    time_id = data['time_id']
                    if isinstance(time_id, torch.Tensor):
                        time_id = time_id.item()
                    camera_frames[time_id] = data
        
        if len(camera_frames) == 0:
            print(f"Error: Camera ID {target_camera_id} not found in dataset.")
            print(f"Available camera IDs: {sorted(set(self.parser.camera_ids))}")
            return
        
        print(f"  Found {len(camera_frames)} frames for this camera")
        sorted_times = sorted(camera_frames.keys())
        print(f"  Time range: [{sorted_times[0]:.4f}, {sorted_times[-1]:.4f}]")
        
        # If camera_time_sync is False, just use the first frame (old behavior)
        if not camera_time_sync:
            ref_data = camera_frames[sorted_times[0]]
            camtoworlds_static = ref_data["camtoworld"].to(self.device)
            Ks_static = ref_data["K"].to(self.device)
            masks_static = ref_data.get("mask", None)
            if masks_static is not None:
                masks_static = masks_static.to(self.device)
            radial_coeffs_static = ref_data.get("poly_coeffs", None)
            if radial_coeffs_static is not None:
                radial_coeffs_static = radial_coeffs_static.to(self.device)
            print(f"  Camera fixed at time_id={sorted_times[0]:.4f}")
        
        print(f"Rendering {num_frames} frames with motion freeze at t={freeze_time}...")
        
        # Store original splats
        original_splats = self.splats
        
        frames = []
        for i in tqdm.tqdm(range(num_frames), desc="Rendering frames"):
            # Current time for moving objects - interpolate between time_start and time_end
            t = i / (num_frames - 1) if num_frames > 1 else 0.0
            current_time_int = time_start + (time_end - time_start) * t  # Integer time interpolated
            current_time_normalized = current_time_int / self.max_time_id  # Convert to normalized [0, 1]
            freeze_time_normalized = freeze_time / self.max_time_id  # Convert freeze_time to normalized
            
            # Get camera parameters for this frame
            if camera_time_sync:
                # Interpolate camera pose based on current_time_normalized
                # Find two nearest frames and interpolate
                nearest_times = sorted(camera_frames.keys(), key=lambda x: abs(x - current_time_normalized))
                
                if abs(nearest_times[0] - current_time_normalized) < 1e-6:
                    # Exact match
                    frame_data = camera_frames[nearest_times[0]]
                else:
                    # Interpolate between two nearest frames
                    # Find bracketing times
                    lower_times = [t for t in sorted_times if t <= current_time_normalized]
                    upper_times = [t for t in sorted_times if t > current_time_normalized]
                    
                    if len(lower_times) == 0:
                        # Before first frame, use first frame
                        frame_data = camera_frames[sorted_times[0]]
                    elif len(upper_times) == 0:
                        # After last frame, use last frame
                        frame_data = camera_frames[sorted_times[-1]]
                    else:
                        # Interpolate between lower and upper
                        t_lower = lower_times[-1]
                        t_upper = upper_times[0]
                        alpha = (current_time_normalized - t_lower) / (t_upper - t_lower + 1e-10)
                        
                        data_lower = camera_frames[t_lower]
                        data_upper = camera_frames[t_upper]
                        
                        # Interpolate camera parameters
                        c2w_lower = data_lower["camtoworld"].to(self.device)
                        c2w_upper = data_upper["camtoworld"].to(self.device)
                        # Linear interpolation for translation
                        c2w_interp = c2w_lower * (1 - alpha) + c2w_upper * alpha
                        
                        K_lower = data_lower["K"].to(self.device)
                        K_upper = data_upper["K"].to(self.device)
                        K_interp = K_lower * (1 - alpha) + K_upper * alpha
                        
                        # Use lower frame for masks and radial coeffs (assume constant)
                        camtoworlds = c2w_interp
                        Ks = K_interp
                        masks = data_lower.get("mask", None)
                        if masks is not None:
                            masks = masks.to(self.device)
                        radial_coeffs = data_lower.get("poly_coeffs", None)
                        if radial_coeffs is not None:
                            radial_coeffs = radial_coeffs.to(self.device)
                        frame_data = None  # Mark as interpolated
                
                if frame_data is not None:
                    # Use exact frame
                    camtoworlds = frame_data["camtoworld"].to(self.device)
                    Ks = frame_data["K"].to(self.device)
                    masks = frame_data.get("mask", None)
                    if masks is not None:
                        masks = masks.to(self.device)
                    radial_coeffs = frame_data.get("poly_coeffs", None)
                    if radial_coeffs is not None:
                        radial_coeffs = radial_coeffs.to(self.device)
            else:
                # Use static camera
                camtoworlds = camtoworlds_static
                Ks = Ks_static
                masks = masks_static
                radial_coeffs = radial_coeffs_static
            
            # Create frozen splats (high-dynamicness objects at freeze_time)
            frozen_splats = {}
            for key, value in original_splats.items():
                if isinstance(value, torch.Tensor) and len(value) == num_total:
                    # Apply both freeze_mask and valid_mask
                    frozen_splats[key] = value[freeze_mask & valid_mask]
                else:
                    frozen_splats[key] = value
            
            # Override intrinsics if using custom pinhole model
            if not use_dataset_intrinsics:
                Ks = custom_K
                radial_coeffs = custom_radial_coeffs
            else:
                # Scale dataset intrinsics to match render resolution
                # Dataset K was created for training resolution, need to scale for render resolution
                K_orig = Ks.clone()
                
                # Get original resolution from K matrix (assume principal point was at center)
                orig_cx = K_orig[0, 0, 2].item()
                orig_cy = K_orig[0, 1, 2].item()
                orig_width = orig_cx * 2
                orig_height = orig_cy * 2
                
                # Scale factors
                scale_x = render_width / orig_width
                scale_y = render_height / orig_height
                
                # Scale intrinsics
                Ks = K_orig.clone()
                Ks[0, 0, 0] *= scale_x  # fx
                Ks[0, 1, 1] *= scale_y  # fy
                Ks[0, 0, 2] *= scale_x  # cx
                Ks[0, 1, 2] *= scale_y  # cy
                
                if i == 0:  # Print once
                    print(f"\nScaled dataset intrinsics to render resolution:")
                    print(f"  Original: {orig_width:.0f}x{orig_height:.0f}")
                    print(f"  Render: {render_width}x{render_height}")
                    print(f"  Scale: {scale_x:.3f}x (H), {scale_y:.3f}x (V)")
                    print(f"  Original fx={K_orig[0,0,0]:.1f}, fy={K_orig[0,1,1]:.1f}, cx={orig_cx:.1f}, cy={orig_cy:.1f}")
                    print(f"  Scaled fx={Ks[0,0,0]:.1f}, fy={Ks[0,1,1]:.1f}, cx={Ks[0,0,2]:.1f}, cy={Ks[0,1,2]:.1f}")
            
            # Apply scale damping to frozen Gaussians if specified
            if scale_damping_frozen < 1.0 and 'scales' in frozen_splats:
                frozen_splats['scales'] = frozen_splats['scales'] + torch.log(torch.tensor(scale_damping_frozen, device=self.device))
            
            # Render frozen objects at freeze_time
            freeze_times = torch.tensor([freeze_time_normalized], dtype=torch.float32, device=self.device)
            self.splats = frozen_splats
            
            frozen_renders, frozen_alphas, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                radial_coeffs=radial_coeffs,
                width=render_width,
                height=render_height,
                sh_degree=self.cfg.sh_degree,
                # masks=masks,
                masks=None,
                deform_opt=True,
                times=freeze_times,
                render_mode="RGB",
            )
            
            # Create moving splats (low-dynamicness objects at current_time)
            moving_splats = {}
            for key, value in original_splats.items():
                if isinstance(value, torch.Tensor) and len(value) == num_total:
                    # Apply valid_mask but NOT freeze_mask (moving = not frozen)
                    moving_splats[key] = value[(~freeze_mask) & valid_mask]
                else:
                    moving_splats[key] = value
            
            # Apply scale damping to moving Gaussians to reduce view-dependent artifacts
            if scale_damping < 1.0 and 'scales' in moving_splats:
                moving_splats['scales'] = moving_splats['scales'] + torch.log(torch.tensor(scale_damping, device=self.device))
            
            # Render moving objects at current_time
            moving_times = torch.tensor([current_time_normalized], dtype=torch.float32, device=self.device)
            self.splats = moving_splats
            
            moving_renders, moving_alphas, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                radial_coeffs=radial_coeffs,
                width=render_width,
                height=render_height,
                sh_degree=self.cfg.sh_degree,
                # masks=masks,
                masks=None,
                deform_opt=True,
                times=moving_times,
                render_mode="RGB",
            )
            
            # Composite frozen and moving renders using alpha blending
            frozen_colors = frozen_renders[..., :3]  # [B, H, W, 3]
            moving_colors = moving_renders[..., :3]  # [B, H, W, 3]
            
            # Show progress every 10 frames
            if i % 10 == 0 and i > 0:
                print(f"  Frame {i}: time={current_time_int:.2f} (normalized={current_time_normalized:.4f})")
            
            # Ensure alphas have consistent shape [B, H, W, 1]
            # Explicitly handle all cases to avoid dimension errors
            if frozen_alphas.dim() == 3:
                # [B, H, W] -> [B, H, W, 1]
                frozen_alpha = frozen_alphas.unsqueeze(-1)
            elif frozen_alphas.dim() == 4 and frozen_alphas.shape[-1] == 1:
                # Already [B, H, W, 1]
                frozen_alpha = frozen_alphas
            else:
                # Unexpected shape - take first channel and add dimension
                print(f"Warning: Unexpected frozen_alpha shape {frozen_alphas.shape}")
                frozen_alpha = frozen_alphas[..., :1] if frozen_alphas.dim() == 4 else frozen_alphas.unsqueeze(-1)
            
            if moving_alphas.dim() == 3:
                # [B, H, W] -> [B, H, W, 1]
                moving_alpha = moving_alphas.unsqueeze(-1)
            elif moving_alphas.dim() == 4 and moving_alphas.shape[-1] == 1:
                # Already [B, H, W, 1]
                moving_alpha = moving_alphas
            else:
                # Unexpected shape - take first channel and add dimension
                print(f"Warning: Unexpected moving_alpha shape {moving_alphas.shape}")
                moving_alpha = moving_alphas[..., :1] if moving_alphas.dim() == 4 else moving_alphas.unsqueeze(-1)
            
            if i == 0:
                print(f"  frozen_alpha (after processing): {frozen_alpha.shape}")
                print(f"  moving_alpha (after processing): {moving_alpha.shape}")
            
            # Alpha composite with optional background priority
            # Standard: moving on top of frozen (moving occludes frozen)
            # Background priority: frozen on top of moving (frozen occludes moving)
            if background_priority:
                # Frozen (background) layer on TOP - ensures static background dominates
                composite = frozen_colors * frozen_alpha + moving_colors * moving_alpha * (1 - frozen_alpha)
                if i == 0:
                    print(f"  Compositing: FROZEN on top (background priority)")
            else:
                # Moving (dynamic) layer on TOP - standard behavior
                composite = moving_colors * moving_alpha + frozen_colors * frozen_alpha * (1 - moving_alpha)
                if i == 0:
                    print(f"  Compositing: MOVING on top (standard)")
            
            composite = composite.clamp(0.0, 1.0)
            
            if i == 0:
                print(f"  composite: {composite.shape}")
            
            # Remove batch dimension and convert to numpy
            img = composite[0].cpu().numpy()  # Should be [H, W, 3]
            
            if i == 0:
                print(f"  img (after [0]): {img.shape}")
                print(f"  img dtype: {img.dtype}, range: [{img.min():.3f}, {img.max():.3f}]")
            
            # Convert to uint8
            img = (img * 255).astype(np.uint8)
            
            # Ensure img is valid for imageio (H, W, C) where C in [1, 3, 4]
            if img.ndim != 3 or img.shape[-1] not in [1, 3, 4]:
                print(f"Warning: Invalid image shape {img.shape}, skipping frame {i}")
                continue
            
            frames.append(img)
            
            # Save frame
            frame_path = os.path.join(output_dir, f"{i+1:04d}.png")
            imageio.imwrite(frame_path, img)
        
        # Restore original splats
        self.splats = original_splats
        
        print(f"\n Rendered {len(frames)} frames with motion freeze")
        print(f" Frames saved: {output_dir}/*.png")
        
        print(f"\n{'='*80}")
        print(f"Motion Freeze Complete!")
        print(f"{'='*80}\n")


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]

        # Load splats parameters
        for k in runner.splats.keys():
            runner.splats[k].data = ckpts[0]["splats"][k]
        
        # Load deform module parameters
        if "deform_module" in ckpts[0]:
            if cfg.deform_opt:
                runner.deformation.load_state_dict(ckpts[0]["deform_module"])
                print("Deformation module loaded.")

        # Check if user wants interactive viewer or trajectory rendering
        if hasattr(cfg, 'view_mode') and cfg.view_mode:
            print("Starting interactive viewer...")
            if not cfg.disable_viewer:
                runner.viewer.start()
        elif hasattr(cfg, 'render_6dof') and cfg.render_6dof:
            print("Rendering 6DoF views...")
            runner.render_6dof(
                output_dir=f"{cfg.result_dir}/6dof_views",
                num_frames=120,
                radius=3.0,
                height=0.5,
                time_start=0.0,
                time_end=1.0,
                colormap='plasma',
            )
        elif hasattr(cfg, 'render_freeze') and cfg.render_freeze:
            if cfg.freeze_mode == "dynamic":
                camera_time_sync = True
                background_priority = True
            elif cfg.freeze_mode == "static":
                camera_time_sync = False
                background_priority = False
            runner.render_motion_freeze(
                output_dir=f"{cfg.result_dir}/motion_freeze/{cfg.freeze_mode}",
                freeze_time=10,
                freeze_threshold=cfg.dyn_app_threshold, # 0.7 for loft
                render_width=520,
                render_height=520,
                time_start=1,
                time_end=13,
                num_frames=30,
                camera_index=4,  # +1 would be the true camera id
                freeze_mode=cfg.freeze_mode,   # "static" or "dynamic"
                camera_time_sync=camera_time_sync,  # Camera moves through time, "dynamic" with True, "static" with False
                # Filtering - NOISE REDUCTION enabled
                filter_floaters=True,
                opacity_threshold=-4.0,         # -4.0 for loft, Very permissive general filter
                min_opacity_frozen=-4.5,        # -4.5 for loft, Very permissive for background (noise reduced by stability)
                min_opacity_moving=-2.5,        # Moderate for dynamic (good structure)
                # Scale control
                scale_damping=1.0,              # Slight shrink for dynamic
                scale_damping_frozen=1.0,       # Full size background
                # Motion-based with STABILITY FILTERING (reduces noise)
                use_motion_magnitude=True,
                motion_threshold=0.001,  # 0.001 for loft, Conservative threshold
                hybrid_weight=0.8,  # 0.8 for loft, 80% dynamicness + 20% motion (stability-filtered)
                # Compositing - standard (dynamic on top)
                background_priority=background_priority, # "dynamic" with True, "static" with False
                fov_scale=0.5,  # 0.8->60, 0.7->73, 0.5->90
                use_dataset_intrinsics=False
            )
        elif hasattr(cfg, 'vis_dynamicness') and cfg.vis_dynamicness:
            print("\n" + "="*80)
            print("Visualizing Dynamicness for All Frames")
            print("="*80)
            
            # Process both train and test sets
            for dataset_name, dataset in [("train", runner.trainset), ("test", runner.valset)]:
                print(f"\nProcessing {dataset_name} set...")
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                )
                
                for idx, data in enumerate(dataloader):
                    # Get camera_id to determine lens folder
                    camera_id = data.get("camera_id")
                    if camera_id is not None:
                        if isinstance(camera_id, torch.Tensor):
                            camera_id = camera_id.item()
                        lens_name = f"lens{camera_id:02d}"
                    else:
                        lens_name = "unknown_lens"
                    
                    # Get actual frame name from image_name field (e.g., "frame_0001.png")
                    if "image_name" in data:
                        # image_name is typically a list with one element
                        image_name = data["image_name"][0] if isinstance(data["image_name"], list) else data["image_name"]
                        # Extract just the filename (e.g., "frame_0001.png" from "lens01/frame_0001.png")
                        frame_name = Path(image_name).name
                    else:
                        # Fallback: construct from time_id or index
                        time_id = data.get("time_id")
                        if time_id is not None:
                            if isinstance(time_id, torch.Tensor):
                                time_id = time_id.item()
                            frame_name = f"frame_{int(time_id):04d}.png"
                        else:
                            frame_name = f"frame_{idx:04d}.png"
                    
                    # Create output directories
                    heatmap_dir = Path(cfg.result_dir) / "dyn_vis" / "heatmap" / lens_name
                    distribution_dir = Path(cfg.result_dir) / "dyn_vis" / "distribution" / lens_name
                    dynamic_dir = Path(cfg.result_dir) / "dyn_vis" / "dynamic" / lens_name
                    heatmap_dir.mkdir(parents=True, exist_ok=True)
                    distribution_dir.mkdir(parents=True, exist_ok=True)
                    dynamic_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Render dynamicness for this frame
                    runner.vis_dynamic(
                        data=data,
                        save_dir=str(heatmap_dir.parent.parent),  # vis/
                        filename=None,  # Will be handled below
                        heatmap_path=str(heatmap_dir / frame_name),
                        distribution_path=str(distribution_dir / frame_name),
                        dynamic_path=str(dynamic_dir / frame_name),
                        dyn_threshold=cfg.dyn_app_threshold,
                        show_threshold=True,
                    )
                    
                    if (idx + 1) % 50 == 0:
                        print(f"  Processed {idx + 1}/{len(dataloader)} frames...")
                
                print(f"Completed {dataset_name} set: {len(dataloader)} frames")
            
            print("\n" + "="*80)
            print("Dynamicness Visualization Complete!")
            print(f"Heatmaps saved to: {cfg.result_dir}/vis/heatmap/")
            print(f"Distributions saved to: {cfg.result_dir}/vis/distribution/")
            print("="*80 + "\n")

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python trainer_app.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python trainer_app.py default --steps_scaler 0.25
    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Vanilla densification heuristics.",
            Config(
                strategy=DefaultStrategy(
                    absgrad=True, 
                    revised_opacity=True, 
                    verbose=True
                ),
            ),
        ),
        "mcmc": (
            "3D Gaussian Splatting as Markov Chain Monte Carlo.",
            Config(
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    cli(main, cfg, verbose=True)
