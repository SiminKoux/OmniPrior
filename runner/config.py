from dataclasses import dataclass, field
from typing import List, TypedDict, Optional, Union
from typing_extensions import Literal, assert_never

from gsplat.strategy import DefaultStrategy, MCMCStrategy

class KPlanesConfig(TypedDict):
    grid_dim: int
    input_dim: int
    output_dim: int
    resolution: List[int]

@dataclass
class DeformConfig:
    # Feature encoder configuration (MLP)
    net_width: int = 64  # Width of the MLP
    net_depth: int = 2   # Depth of the MLP ('1' means only one input layer from feature_dim to n_width)
    
    # Positional encoding dimension, 0 means no positional encoding
    posebase_pe: int = 0       # default is 10
    scale_rotation_pe: int = 0 # default is 2
    
    # Scene bounds
    bounds:float = 3.0  # default is 1.6

    # Feature plane configuration
    kplanes_config: KPlanesConfig = field(default_factory=lambda: {
        'grid_dim': 2,    # Dimension of the feature grid
        'input_dim': 4,   # Dimension of the input vectors
        'output_dim': 16, # Dimension of the output features
        'resolution': [64, 64, 64, 128] # Resolution of the feature grid
    })

    # Mult-resolution of voxel grid, default is [1, 2, 4, 8]
    multires: List[int] = field(default_factory=lambda: [1, 2])

    use_dynamic_mask: bool = False # Whether use the dynamic mask (for hard embedding)
    enable_ddyn: bool = False      # enable learning offsets for canonical dynamicness

    # Learning rate for the deformation module
    deform_opt_lr: float = 1.6e-3  # default: 1.6e-4
    grid_opt_lr: float = 1.6e-2    # default: 1.6e-3

    # Weight of regularization terms
    tv_loss: bool = False         # Whether to use TV loss
    s_tv_reg: float = 1.0e-4      # default: 0.0002
    st_tv_reg: float = 5.0e-4     # default: 0.001
    st_l1_reg: float = 1.0e-4     # default: 0.0001
    tv_mode: str = 'charbonnier'  # ['l2','huber','charbonnier']
    tv_eps: float = 1.0e-5        # epsilon for numerical stability in charbonnier
    tv_huber_delta: float = 0.01  # huber delta for TV loss
    time_dim: int = -2            # Time dimension for temporal features
    
    # Dynamicness guidance regularization
    guidance_reg: float = 5.0e-2  # 5.0e-3

    gate_k: float = 8.0     # k for sigmoid gate
    gate_tau: float = 0.5   # tau for sigmoid gate
    use_gate: float = False # Whether to use the gating mechanism

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None

    # Path to the dataset
    data_dir: str = "data/OmniFisheye_plus/lab"
    # Downsample factor for the dataset
    data_factor: int = 1
    # Directory to save results
    result_dir: str = "results/lab"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = False
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "fisheye"
    # Whether to apply filtering on initial sfm point cloud
    filter: bool = False
    # Whether to use the pre-calibrated radial coefficients
    use_rad_coef: bool = False

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # ===== Interactive 6DOF Viewer Options =====
    # Enable interactive viewing mode (requires --ckpt)
    view_mode: bool = False
    # Initial time for viewer (0.0 to 1.0, 0.5 = middle of sequence)
    view_time: float = 0.5
    # Port for interactive viewer server
    viewer_port: int = 8080

    # ===== Trajectory Rendering Options =====
    # Enable trajectory video rendering (requires --ckpt)
    render_6dof: bool = False
    # Number of frames for trajectory video
    num_frames: int = 120
    # Orbit radius for trajectory
    trajectory_radius: float = 3.0
    # Camera height for trajectory
    trajectory_height: float = 0.5
    # Time range for dynamic interpolation (start, end)
    time_start: float = 0.0
    time_end: float = 1.0
    # Resolution for rendered video (width, height)
    render_width: int = 576
    render_height: int = 768

    # Enable depth evaluation
    eval_depth: bool = False

    # Number of training steps
    max_steps: int = 30_000
    # Number of steps for the initial gaussian phase
    init_steps: int = 10_000
    # Number of steps for each phase
    gaussian_phase_length: int = 10_000
    deform_phase_length: int = 10_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 15_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 15_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 15_000, 30_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "metric"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 10_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 0.5
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.5
    # Initial scale of GS
    init_scale: float = 0.1
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01 # default: 0.01
    # Far plane clipping distance
    far_plane: float = 1e10  # default: 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0   # 0.001
    # Scale regularization
    scale_reg: float = 0.0     # 0.03
    # Mono-depth Ranking regularization
    ranking_reg: float = 0.03  # 0.05
    pre_rank_end: int = 1_000
    rank_ramp_up: int = 5_000
    rank_peak: int = 15_000
    rank_fade_end: int = 25_000
    rank_min_weight: float = 0.2
    # Metric depth regularization
    metric_depth_reg: float = 0.01 # 0.01, set to 0.0 to disable
    metric_depth_reg_start: int = 5_000
    metric_depth_reg_end: int = 25_000
    metric_depth_warmup_ratio: float = 0.15 # 15% (3k steps)
    metric_depth_decay_ratio: float = 0.2   # 20% (4k steps)
    metric_huber_delta: float = 0.5  # Delta for Huber (smooth l1) loss (in meters)
    metric_loss_type: Literal["l1", "l2", "huber", "si_log"] = "huber"
    # Depth smoothness regularization (edge-aware)
    depth_smooth_reg: float = 0.01   # 0.01-0.02, set to 0.0 to disable
    depth_smooth_start: int = 15_000 # Start after depth stablizes
    depth_smooth_end: int = 30_000   # Through deformation phase
    smooth_edge_lambda: float = 10.0 # Edge sensitivity (higher=stronger edge preservation)
    smooth_edge_aware: bool = True   # Use image gradients for edge detection
    
    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "vgg"

    # Dynamic or static representation
    deform_opt: bool = False
    deform: DeformConfig = field(default_factory=DeformConfig)

    # Dynamicness Application
    render_freeze: bool = False
    freeze_mode: Literal["dynamic", "static"] = "static"
    vis_dynamicness: bool = False
    dyn_app_threshold: float = 0.5

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)