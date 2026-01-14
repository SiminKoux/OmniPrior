import os
import math
import json
import time
import tqdm
import tyro
import yaml
import viser
import imageio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from fused_ssim import fused_ssim
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
    masked_psnr,
    masked_l1_loss,
    set_random_seed,
    save_hist,
    save_scatter,
    scatter_map,
    extract_dyn_bboxes
)

from gsplat import export_splats
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap

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
        # self.parser.vis_filtered_points(save_path="filtered_points_3d.html")
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
            print("Using GS Deformation for Dynamics...")
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

    def _handle_resume(self, start_step: int, max_steps: int):
        """Handle optimizer and scheduler state when resuming."""
        
        # Recompute decayed LR for means optimizer
        if "means" in self.optimizers:
            BS = self.cfg.batch_size * self.world_size
            base_lr_means = self.cfg.means_lr * math.sqrt(BS)
            gamma = 0.01 ** (1.0 / max_steps)
            resumed_lr = base_lr_means * (gamma ** start_step)
            
            for g in self.optimizers["means"].param_groups:
                g["lr"] = resumed_lr
        
        # Rebuild scheduler with correct last_epoch
        self.schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"],
                gamma=0.01 ** (1.0 / max_steps),
                last_epoch=start_step - 1  # -1 because scheduler will increment on first step
            )
        ]
        
        if self.world_rank == 0:
            print(f"Resumed from step {start_step}, Learning Rate set to {resumed_lr:.2e}")
    
    def train(self, start_step=0):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        print("-" * 9 + f" [Training] " + "-" * 9)

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)
                print("[config] saved.")

        max_steps = cfg.max_steps
        gaussian_steps = cfg.gaussian_phase_length
        
        sampler_seed = 20250606 + self.world_rank
        batch_sampler = CamBatchSampler(self.trainset, seed=sampler_seed)

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_sampler=CamBatchSampler(self.trainset),
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        
        # LOG INITIAL SETUP INFO
        if self.world_rank == 0:  # Only log from main process
            initial_info = batch_sampler.get_epoch_info()
            print(f"\n=== TRAINING SETUP ===")
            print(f"[Dataset]: {initial_info['total_samples']} samples, "
                  f"{initial_info['num_timestamps']} timestamps")
            print(f"Base seed: {sampler_seed}, Epoch 0 seed: {initial_info['epoch_seed']}")
            print(f"Batches per epoch: {initial_info['num_batches']}")
            print(f"Expected steps: {max_steps} ({max_steps // initial_info['num_batches']} epochs)\n")

        steps_per_epoch = len(trainloader)

        # Calculate initial epoch and position
        start_epoch = start_step // steps_per_epoch
        offset_in_epoch = start_step % steps_per_epoch
        current_epoch = start_epoch

        # LOG RESUME INFO (if resuming)
        if start_step > 0:
            batch_sampler.set_epoch(current_epoch)
            resume_info = batch_sampler.get_epoch_info()
            if self.world_rank == 0:
                print(f"=== RESUMING TRAINING ===")
                print(f"Resuming from step {start_step} (epoch {current_epoch})")
                print(f"Resume epoch seed: {resume_info['epoch_seed']}")
            
            self._handle_resume(start_step, max_steps)

        # Set initial epoch for sampler
        batch_sampler.set_epoch(current_epoch)
        trainloader_iter = iter(trainloader)

        # Skip batches if resuming mid-epoch
        if offset_in_epoch > 0:
            print(f"Skipping {offset_in_epoch} batches to resume from step {start_step}\n")
            for _ in range(offset_in_epoch):
                try:
                    next(trainloader_iter)
                except StopIteration:
                    # This shouldn't happen but handle gracefully
                    current_epoch += 1
                    batch_sampler.set_epoch(current_epoch)
                    trainloader_iter = iter(trainloader)
                    break

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(start_step, max_steps))
        
        for step in pbar:
            if cfg.deform_opt:
                if step == gaussian_steps:
                    print(f"\nStep {step} - Transformation Learning Started...")
                    self._is_gaussian_phase = False
                    self._is_deform_phase = True
                    self.splats["dynamicness"].requires_grad_(False)

                    ### Initial schedulers for the deform optimizer
                    remaining_steps = max_steps - (step + 1)
                    # make sure warmup_steps less than half of the remaining_steps
                    warmup_steps = min(1000, remaining_steps // 2)
                    if remaining_steps <= warmup_steps:
                        print(f"Warning: Only {remaining_steps} steps remaining, using {warmup_steps} warmup steps")
                    self.schedulers.extend([
                        torch.optim.lr_scheduler.ChainedScheduler([
                            torch.optim.lr_scheduler.LinearLR(
                                self.deform_optimizers[0],
                                start_factor=0.01, # Start at 1% of base lr
                                total_iters=warmup_steps,  # Warmup steps
                            ), # Warmup Phase: Linear ramp-up (0.0000016 -> 0.00016)
                            # Prevents gradient explosions when unfreezing the module 
                            # by starting with very low LRs.
                            torch.optim.lr_scheduler.ExponentialLR(
                                self.deform_optimizers[0], 
                                gamma=0.1 ** (1.0 / max(1, remaining_steps - warmup_steps))
                            ) # Decay Phase: Exponential decay to 10% (0.00016 -> 0.000016)
                        ]), # MLP scheduler
                        torch.optim.lr_scheduler.ChainedScheduler([
                            torch.optim.lr_scheduler.LinearLR(
                                self.deform_optimizers[1],
                                start_factor=0.01, # Start at 1% of base lr
                                total_iters=warmup_steps,  # Warmup steps (0.000016 -> 0.0016)
                            ), # Warmup Phase: Linear ramp-up (0.000016 -> 0.0016)
                            torch.optim.lr_scheduler.ExponentialLR(
                                self.deform_optimizers[1], 
                                gamma=0.1 ** (1.0 / max(1, remaining_steps - warmup_steps))
                            ) # Decay Phase: Exponential decay to 10% (0.0016 -> 0.00016)
                        ]) # Grid scheduler
                    ])

                    # Unfreeze deform module
                    self.deformation.st_deform.unfreeze_mlps()
                    self.deformation.st_deform.grid.unfreeze_planes()
            
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                current_epoch += 1  # Track epoch progression
                batch_sampler.set_epoch(current_epoch)  # Update RNG seed for new shuffle
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = data["camtoworld"].to(device)    # [B, 4, 4]
            Ks = data["K"].to(device)                      # [B, 3, 3]
            pixels = data["image"].to(device) / 255.0      # [B, H, W, 3]
            num_train_rays_per_step = (pixels.shape[0] * pixels.shape[1] * pixels.shape[2])
            radial_coeffs = data["poly_coeffs"].to(device) # [B, 4]
            height, width = pixels.shape[1:3]
            masks = data["mask"].to(device) if "mask" in data else None   # [B, H, W]
            times = data["time_id"].to(device) if "time_id" in data else None   # [B]
            dynamic_masks = data["dynamic_masks"].to(device) if "dynamic_masks" in data else None # [B, H, W]
            soft_masks = data["soft_masks"].to(device) if "soft_masks" in data else None          # [B, H, W]
            mono_depths = data["mono_depths"].to(device) if "mono_depths" in data else None       # [B, H, W]
            metric_depths = data["metric_depths"].to(device) if "metric_depths" in data else None # [B, H, W]
            eroded_masks = data["eroded_mask"].to(device) if "eroded_mask" in data else None      # [B, H, W]
            frame_time_id = int(data["frame_time_id"][0])       # int in [0, max_time_id], e.g. 6 for 10-frame scene
            matches = self.parser.sparse_matches[frame_time_id] # Dict: ['0->1', '1->2', '2->3', '3-4', '4->5', '5->0']

            # sh scheduling
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, _, info, deform_params, final_params = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                radial_coeffs=radial_coeffs,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                masks=masks,
                deform_opt=cfg.deform_opt,
                times=times,
                render_mode="RGB+ED" #if cfg.depth_reg > 0.0 else "RGB",
            )

            colors = renders[..., :3].clamp(0.0, 1.0)  # [B, H, W, 3], RGB colors, [0, 1]
            if renders.shape[-1] == 4:
                depths = renders[..., 3:4].squeeze(-1) # [B, H, W], metric depth
            else:
                depths = None
            
            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            # L1 Loss
            l1loss = masked_l1_loss(colors, pixels, masks)
            
            # SSIM Loss
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), 
                pixels.permute(0, 3, 1, 2), 
                padding="valid"
            )

            # Photometric Loss
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            # Regularizations
            if cfg.opacity_reg > 0.0:
                opacity_values = torch.sigmoid(self.splats["opacities"])
                opacity_loss = cfg.opacity_reg * opacity_values.mean()
                loss += opacity_loss
            if cfg.scale_reg > 0.0:
                scale_loss = cfg.scale_reg * torch.exp(self.splats["scales"]).mean()
                loss += scale_loss

            # Metric depth regularization
            metric_depth_loss_computed = False
            if cfg.metric_depth_reg > 0.0 and depths is not None and metric_depths is not None:
                metric_depth_dict = self.metric_depth_regulator(
                    rendered_depth=depths,
                    metric_depth=metric_depths,
                    valid_mask=masks,
                    step=step
                )
                # Extract loss components
                weight = metric_depth_dict['weight']
                weighted_metric_depth_loss = cfg.metric_depth_reg * metric_depth_dict['loss']
                # Add to total loss if weight > 0
                if weight > 0.0:
                    loss += weighted_metric_depth_loss
                    metric_depth_loss_computed = True
            
            # Depth smoothness regularization (edge-aware structure prior)
            smoothness_loss_computed = False
            if cfg.depth_smooth_reg > 0.0 and depths is not None:
                smoothness_dict = self.depth_smoothness_regulator(
                    rendered_depth=depths,
                    rgb=colors if cfg.smooth_edge_aware else None,
                    valid_mask=masks,
                    step=step
                )
                # Extract loss components
                smooth_weight = smoothness_dict['weight']
                weighted_smoothness_loss = smoothness_dict['loss']
                # Add to total loss if weight > 0
                if smooth_weight > 0.0:
                    loss += weighted_smoothness_loss
                    smoothness_loss_computed = True
            
            # Depth-aware ranking regularizations
            if cfg.ranking_reg > 0.0 and matches is not None:
                mono_rank_weight = self.mono_rank_regulator._compute_weight(step)
                mono_rank_active = (mono_rank_weight > 0.0)
                # collate_active = bool((step % 5000 == 0 or step==max_steps-1) and mono_rank_active)
                collate_active = False

                if mono_rank_active:
                    mono_rank_loss_dict = self.mono_rank_regulator(
                        step=step,
                        rendered_depth=depths,     # [B, H, W], metric depth
                        mono_depth=mono_depths,    # [B, H, W], relative depth
                        valid_mask=eroded_masks,   # [B, H, W] valid pixels
                        collect_vis=collate_active # whether to collect visualization data
                    )
                    mono_rank_loss = mono_rank_weight * mono_rank_loss_dict['loss']
                    loss += cfg.ranking_reg * mono_rank_loss

                if collate_active:
                    mono_rank_vis_dir = f"{self.log_dir}/ranking_vis"
                    os.makedirs(mono_rank_vis_dir, exist_ok=True)
                    self.mono_rank_regulator.vis_sampled_pairs(
                        save_path=mono_rank_vis_dir,
                        step=step,
                        max_vis_pairs=50,
                    )
            
            if cfg.deform_opt and self._is_deform_phase:
                if cfg.deform.tv_loss:
                    deform_loss, s_tv, st_tv, st_l1 = self.deformation.st_deform_loss()
                    loss += deform_loss
            
            if cfg.deform.guidance_reg > 0.0 and step >= cfg.init_steps:
                if self._is_gaussian_phase:
                    # Canonical phase: supervise canonical dynamicness with soft masks
                    cano_guide_loss_dict = self.dyn_guide_regulator(
                        dynamicness=self.splats["dynamicness"],
                        info=info,
                        dynamic_masks=soft_masks,
                        valid_mask=masks
                    )
                    cano_dyn_loss = cfg.deform.guidance_reg * cano_guide_loss_dict["guidance"]
                    # cano_loss_s = cano_guide_loss_dict["guidance_static"]
                    # cano_loss_d = cano_guide_loss_dict["guidance_dynamic"]
                    loss += cano_dyn_loss
                elif self._is_deform_phase and cfg.deform.enable_ddyn and deform_params.dynamic_offset is not None:
                    # Deform phase: supervise deformed dynamicness with dynamic masks
                    combined_dynamicness = self.splats["dynamicness"] + deform_params.dynamic_offset
                    delta_guide_loss_dict = self.dyn_guide_regulator(
                        dynamicness=combined_dynamicness,
                        info=info,
                        dynamic_masks=dynamic_masks,
                        valid_mask=masks
                    )
                    delta_dyn_loss = cfg.deform.guidance_reg * delta_guide_loss_dict["guidance"]
                    # delta_loss_s = delta_guide_loss_dict["guidance_static"]
                    # delta_loss_d = delta_guide_loss_dict["guidance_dynamic"]
                    loss += delta_dyn_loss

            loss.backward()

            desc = f"total loss={loss.item():.3f}| "
            if self._is_gaussian_phase:
                if cfg.opacity_reg > 0.0:
                    desc += f"opacity reg={opacity_loss.item():.6f}| "
                if cfg.scale_reg > 0.0:
                    desc += f"scale reg={scale_loss.item():.6f}| " 
            if metric_depth_loss_computed:
                desc += f"metric={weighted_metric_depth_loss.item():.6f}| "
            if smoothness_loss_computed:
                desc += f"smooth={weighted_smoothness_loss.item():.6f}| "
            if cfg.ranking_reg > 0.0:
                if mono_rank_active:
                    desc += f"rank={mono_rank_loss:.6f}| "
            if cfg.deform_opt and self._is_deform_phase:
                desc += f"tv reg={deform_loss.item():.6f}| "
            if cfg.deform.guidance_reg > 0.0:
                if step >= cfg.init_steps and step < gaussian_steps:
                    desc += f"cano dyn={cano_dyn_loss.item():.6f}| "
                elif step >= gaussian_steps and cfg.deform.enable_ddyn:
                    desc += f"delta dyn={delta_dyn_loss.item():.6f}| "
            pbar.set_description(desc)

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                self.writer.add_scalar("train/total_loss", loss.item(), step)
                self.writer.add_scalar("train/l1_loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssim_loss", ssimloss.item(), step)
                if cfg.opacity_reg > 0.0:
                    self.writer.add_scalar("train/opacity_reg", opacity_loss.item(), step)
                if cfg.scale_reg > 0.0:
                    self.writer.add_scalar("train/scale_reg", scale_loss.item(), step)
                # Log metric depth regularization
                if metric_depth_loss_computed:
                    # self.writer.add_scalar("train/metric_depth_reg", weighted_metric_depth_loss.item(), step)
                    self.writer.add_scalar("train/metric_depth_reg_raw", metric_depth_dict['raw_loss'].item(), step)
                    # self.writer.add_scalar("train/metric_depth_weight", metric_depth_dict['weight'], step)
                    # self.writer.add_scalar("train/metric_depth_valid_ratio", metric_depth_dict['valid_ratio'], step)
                # Log smoothness regularization
                if smoothness_loss_computed:
                    # self.writer.add_scalar("train/depth_smooth_reg", weighted_smoothness_loss.item(), step)
                    self.writer.add_scalar("train/depth_smooth_reg_raw", smoothness_dict['raw_loss'].item(), step)
                    # self.writer.add_scalar("train/depth_smooth_weight", smoothness_dict['weight'], step)
                # Only log ranking if it's active
                if cfg.ranking_reg > 0.0:
                    if mono_rank_active:
                        # self.writer.add_scalar("train/mono_rank_reg", mono_rank_loss.item(), step)
                        self.writer.add_scalar("train/mono_rank_reg_raw", mono_rank_loss_dict['loss'].item(), step)
                        # self.writer.add_scalar("train/mono_rank_weight", mono_rank_weight, step)
                if cfg.deform_opt and self._is_deform_phase:
                    if cfg.deform.tv_loss:
                        self.writer.add_scalar(f"train/deform_tv_reg", deform_loss.item(), step)
                        # self.writer.add_scalar(f"train/s_tv_reg", s_tv.item(), step)
                        # self.writer.add_scalar(f"train/st_tv_reg", st_tv.item(), step)
                        # self.writer.add_scalar(f"train/st_l1_reg", st_l1.item(), step)
                if step >= cfg.init_steps:
                    if cfg.deform.guidance_reg > 0.0:
                        if step >= cfg.init_steps and step < gaussian_steps:
                            self.writer.add_scalar("train/cano_dyn_reg", cano_dyn_loss.item(), step)
                        elif step >= gaussian_steps and cfg.deform.enable_ddyn:
                            self.writer.add_scalar("train/delta_dyn_reg", delta_dyn_loss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # Save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                self.save_checkpoint(step=step, global_tic=global_tic)
            
            # Save Gaussian point cloud as .ply files
            if (step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1) and cfg.save_ply:
                self.save_ply(step=step)
            # if self._is_deform_phase and (step % 2000 == 0 or step == max_steps - 1):
            #     self.save_gauss(
            #         times=times,
            #         final_params=final_params,
            #         step=step
            #     )

            # Optimization step
            for o in self.optimizers.values():
                o.step()
                o.zero_grad(set_to_none=True)
            if cfg.deform_opt and self._is_deform_phase:
                for od in self.deform_optimizers:
                    od.step()
                    od.zero_grad(set_to_none=True)
            
            # update learning rates
            for scheduler in self.schedulers:
                scheduler.step()
                
            # run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=self.schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # Run evaluation
            if step in [i - 1 for i in self.eval_steps] or step == max_steps - 1:
                self.eval(step)
            
            # if step >= gaussian_steps and (step % 5000 == 0 or step == max_steps - 1):
            #     self.log_info(
            #         step=step, 
            #         deform_params=deform_params, 
            #         means2d=info["means2d"],
            #         gt_rgb=pixels,
            #         render_rgb=colors, 
            #         dyn_mask=dynamic_masks,
            #         soft_mask=soft_masks,
            #         H=height, W=width, 
            #         save_dir=f"{self.log_dir}/info_vis"
            #     )
            
            # Visualize depths periodically
            # if depths is not None and step >=10000 and (step % 5000 == 0 or step == max_steps - 1):
            if depths is not None and step == max_steps - 1:
                self.vis_depth(
                    step=step,
                    gt_rgb=pixels,
                    render_rgb=colors,
                    rendered_depth=depths,
                    metric_depth=metric_depths,
                    valid_mask=masks,
                    save_dir=f"{self.log_dir}/depth_vis"
                )

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)
        
        # Calculate and save total training time
        total_training_time = time.time() - global_tic
        hours = int(total_training_time // 3600)
        minutes = int((total_training_time % 3600) // 60)
        seconds = int(total_training_time % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        time_file = os.path.join(self.stats_dir, "training_time_fps.txt")
        with open(time_file, "w") as f:
            f.write(f"Total training time: {time_str}\n")
            f.write(f"Total seconds: {total_training_time:.3f}\n")
            if self.final_fps is not None:
                f.write(f"Final evaluation FPS: {self.final_fps:.3f}\n")
        
        print(f"Total training time: {time_str}")
        if self.final_fps is not None:
            print(f"Final evaluation FPS: {self.final_fps:.3f}")
        
        self.save_checkpoint(step=max_steps)
        self.save_ply(step=max_steps)
        print("Training complete.")

        self.render_all(use_rig_mask=True)
        print("Rendering complete.")
    
    @torch.no_grad()
    def vis_loaded_matches(
        self,
        step: int,
        data: Dict[str, torch.Tensor],
        frame_time_id: int,
        pixels: torch.Tensor,
        mono_depths: Optional[torch.Tensor],
        sparse_matches: Dict[str, Dict[str, torch.Tensor]],
        save_dir: str = None,
    ):
        """
        Visualize loaded sparse matches overlaid on RGB and depth maps to verify correctness.
        Shows numbered points (like vis_match.py) for easy correspondence verification.
        
        Args:
            step: Current training step
            data: Batch data dictionary
            pixels: RGB images [B, H, W, 3]
            mono_depths: Monocular depth maps [B, H, W]
            sparse_matches: Match dictionary {'lens_i->lens_j': {'pts_i', 'pts_j', 'confidence'}}
            save_dir: Output directory
        """
        if save_dir is None:
            save_dir = f"{self.cfg.result_dir}/match_verification"
        os.makedirs(save_dir, exist_ok=True)
        
        camera_ids = data["camera_id"]  # [B]
        masks = data.get("mask", None)  # [B, H, W]
        
        B, H, W = pixels.shape[:3]
        
        print(f"\n{'='*80}")
        print(f"Match Verification @ Step {step}")
        print(f"{'='*80}")
        print(f"Time ID: {frame_time_id}")
        print(f"Batch size: {B} cameras, Image size: {H}x{W}")
        print(f"Camera IDs: {camera_ids.tolist()}")
        print(f"Match pairs: {list(sparse_matches.keys())}")
        
        # Process each lens pair
        for pair_key, match_data in sparse_matches.items():
            lens_i, lens_j = map(int, pair_key.split('->'))
            
            # Extract match data with robust shape handling
            pts_i_raw = match_data['pts_i'].cpu().numpy()
            pts_j_raw = match_data['pts_j'].cpu().numpy()
            confidence_raw = match_data['confidence'].cpu().numpy()
            
            # Ensure 2D for points [M, 2]
            if pts_i_raw.ndim == 1:
                pts_i = pts_i_raw.reshape(1, -1)
            elif pts_i_raw.ndim == 3:
                pts_i = pts_i_raw.squeeze(-1)
            else:
                pts_i = pts_i_raw
            
            if pts_j_raw.ndim == 1:
                pts_j = pts_j_raw.reshape(1, -1)
            elif pts_j_raw.ndim == 3:
                pts_j = pts_j_raw.squeeze(-1)
            else:
                pts_j = pts_j_raw
            
            confidence = confidence_raw.flatten()
            
            M = len(pts_i)
            print(f"\nPair {pair_key}: {M} matches (raw)")
            
            # Length validation
            if len(confidence) != M:
                print(f"  Warning: Confidence length {len(confidence)} != {M}, truncating/padding")
                if len(confidence) > M:
                    confidence = confidence[:M]
                else:
                    confidence = np.pad(confidence, (0, M - len(confidence)), constant_values=0.5)
            
            if len(pts_j) != M:
                print(f"  Warning: pts_j length {len(pts_j)} != {M}, skipping")
                continue
            
            print(f"  Confidence: min={confidence.min():.3f}, max={confidence.max():.3f}, mean={confidence.mean():.3f}")
            
            # Find camera indices in batch
            cam_i_1indexed = lens_i + 1
            cam_j_1indexed = lens_j + 1
            idx_i = (camera_ids == cam_i_1indexed).nonzero(as_tuple=True)[0]
            idx_j = (camera_ids == cam_j_1indexed).nonzero(as_tuple=True)[0]
            
            if len(idx_i) == 0 or len(idx_j) == 0:
                print(f"  Warning: Camera {cam_i_1indexed} or {cam_j_1indexed} not in batch, skipping")
                continue
            
            idx_i = idx_i[0].item()
            idx_j = idx_j[0].item()
            
            # Extract RGB and depth
            rgb_i = pixels[idx_i].cpu().numpy()  # [H, W, 3]
            rgb_j = pixels[idx_j].cpu().numpy()
            
            if mono_depths is not None:
                depth_i = mono_depths[idx_i].cpu().numpy()
                depth_j = mono_depths[idx_j].cpu().numpy()
            else:
                depth_i = np.zeros((H, W))
                depth_j = np.zeros((H, W))
            
            if masks is not None:
                mask_i = masks[idx_i].cpu().numpy()
                mask_j = masks[idx_j].cpu().numpy()
            else:
                mask_i = np.ones((H, W))
                mask_j = np.ones((H, W))
            
            # Sort by confidence and select top matches for visualization
            max_vis = min(20, len(confidence))  # Show top 20 numbered points
            sorted_idx = np.argsort(confidence)[::-1][:max_vis]
            pts_i_vis = pts_i[sorted_idx]
            pts_j_vis = pts_j[sorted_idx]
            conf_vis = confidence[sorted_idx]
            
            # Process depths (normalize with mask)
            depth_i_masked = np.where(mask_i > 0.5, depth_i, np.nan)
            depth_j_masked = np.where(mask_j > 0.5, depth_j, np.nan)
            
            cmap_depth = plt.get_cmap('jet')
            
            # Normalize depth i
            valid_depth_i = depth_i_masked[~np.isnan(depth_i_masked)]
            if len(valid_depth_i) > 0:
                d_min_i, d_max_i = valid_depth_i.min(), valid_depth_i.max()
                depth_i_norm = np.where(
                    ~np.isnan(depth_i_masked),
                    (depth_i_masked - d_min_i) / (d_max_i - d_min_i + 1e-8),
                    np.nan
                )
            else:
                depth_i_norm = depth_i_masked
            
            depth_i_colored = cmap_depth(depth_i_norm)[:, :, :3]
            depth_i_rgb = np.where(np.isnan(depth_i_norm)[..., np.newaxis], [0, 0, 0], depth_i_colored)
            
            # Normalize depth j
            valid_depth_j = depth_j_masked[~np.isnan(depth_j_masked)]
            if len(valid_depth_j) > 0:
                d_min_j, d_max_j = valid_depth_j.min(), valid_depth_j.max()
                depth_j_norm = np.where(
                    ~np.isnan(depth_j_masked),
                    (depth_j_masked - d_min_j) / (d_max_j - d_min_j + 1e-8),
                    np.nan
                )
            else:
                depth_j_norm = depth_j_masked
            
            depth_j_colored = cmap_depth(depth_j_norm)[:, :, :3]
            depth_j_rgb = np.where(np.isnan(depth_j_norm)[..., np.newaxis], [0, 0, 0], depth_j_colored)
            
            # === Create visualization: 2 rows * 2 columns ===
            # Row 1: RGB images with numbered points
            # Row 2: Depth images with numbered points
            
            gap = 4
            gap_strip_h = np.full((H, gap, 3), 1.0, dtype=np.float32)
            
            # Row 1: RGB with numbered points
            rgb_i_marked = rgb_i.copy()
            rgb_j_marked = rgb_j.copy()
            
            # Row 2: Depth with numbered points
            depth_i_marked = depth_i_rgb.copy()
            depth_j_marked = depth_j_rgb.copy()
            
            # Draw numbered points
            for k in range(len(pts_i_vis)):
                # Convert (y, x) to integer coordinates
                y_i, x_i = int(pts_i_vis[k, 0]), int(pts_i_vis[k, 1])
                y_j, x_j = int(pts_j_vis[k, 0]), int(pts_j_vis[k, 1])
                
                # Color based on confidence (green=high, red=low)
                color = plt.cm.RdYlGn(conf_vis[k])[:3]  # RGB tuple
                
                # Draw on RGB images (colored circles)
                if 0 <= y_i < H and 0 <= x_i < W:
                    # Draw circle (radius 8)
                    for dy in range(-8, 9):
                        for dx in range(-8, 9):
                            if dy*dy + dx*dx <= 64:  # radius^2
                                py, px = y_i + dy, x_i + dx
                                if 0 <= py < H and 0 <= px < W:
                                    rgb_i_marked[py, px] = color
                
                if 0 <= y_j < H and 0 <= x_j < W:
                    for dy in range(-8, 9):
                        for dx in range(-8, 9):
                            if dy*dy + dx*dx <= 64:
                                py, px = y_j + dy, x_j + dx
                                if 0 <= py < H and 0 <= px < W:
                                    rgb_j_marked[py, px] = color
                
                # Draw on depth images (white cross, unchanged for contrast)
                if 0 <= y_i < H and 0 <= x_i < W:
                    depth_i_marked[max(0, y_i-6):min(H, y_i+7), x_i] = [1, 1, 1]
                    depth_i_marked[y_i, max(0, x_i-6):min(W, x_i+7)] = [1, 1, 1]
                
                if 0 <= y_j < H and 0 <= x_j < W:
                    depth_j_marked[max(0, y_j-6):min(H, y_j+7), x_j] = [1, 1, 1]
                    depth_j_marked[y_j, max(0, x_j-6):min(W, x_j+7)] = [1, 1, 1]
            
            # Concatenate rows
            row1 = np.concatenate([rgb_i_marked, gap_strip_h, rgb_j_marked], axis=1)
            row2 = np.concatenate([depth_i_marked, gap_strip_h, depth_j_marked], axis=1)
            
            # Stack rows with vertical gap
            gap_strip_v = np.full((gap, row1.shape[1], 3), 1.0, dtype=np.float32)
            final_img = np.concatenate([row1, gap_strip_v, row2], axis=0)
            
            # === Create figure with numbered annotations ===
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.imshow(final_img)
            ax.axis('off')

            # Add numbered text labels (colored by confidence)
            for k in range(len(pts_i_vis)):
                y_i, x_i = int(pts_i_vis[k, 0]), int(pts_i_vis[k, 1])
                y_j, x_j = int(pts_j_vis[k, 0]), int(pts_j_vis[k, 1])
                
                # Color based on confidence
                color = plt.cm.RdYlGn(conf_vis[k])[:3]
                
                # Offset for right image in side-by-side
                x_j_offset = x_j + W + gap
                
                # Add numbers on RGB row (colored text on colored box)
                if 0 <= y_i < H and 0 <= x_i < W:
                    ax.text(x_i, y_i, str(k+1), color='white', fontsize=10, ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.1", facecolor=color, alpha=0.7, edgecolor=color, linewidth=1))
                
                if 0 <= y_j < H and 0 <= x_j < W:
                    ax.text(x_j_offset, y_j, str(k+1), color='white', fontsize=10, ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.1", facecolor=color, alpha=0.7, edgecolor=color, linewidth=1))

                # Add numbers on depth row (colored text on colored box, adjusted for depth background)
                y_depth_offset = H + gap
                if 0 <= y_i < H and 0 <= x_i < W:
                    ax.text(x_i, y_depth_offset + y_i, str(k+1), color='black', fontsize=10, ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.1", facecolor=color, alpha=0.8, edgecolor=color, linewidth=1))
                
                if 0 <= y_j < H and 0 <= x_j < W:
                    ax.text(x_j_offset, y_depth_offset + y_j, str(k+1), color='black', fontsize=10, ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.1", facecolor=color, alpha=0.8, edgecolor=color, linewidth=1))
            
            # Add column labels
            col_width = W + gap
            ax.text(W//2, -10, f"Lens {cam_i_1indexed:02d}", ha='center', va='bottom',
                    fontsize=8, fontweight='normal', color='black')
            ax.text(col_width + W//2, -10, f"Lens {cam_j_1indexed:02d}", ha='center', va='bottom',
                    fontsize=8, fontweight='normal', color='black')
            
            # Add row labels
            row_height = H + gap
            ax.text(-10, H//2, "RGB", ha='right', va='center',
                    fontsize=8, fontweight='normal', color='black', rotation=90)
            ax.text(-10, row_height + H//2, "Depth", ha='right', va='center',
                    fontsize=8, fontweight='normal', color='black', rotation=90)
            
            # Title with raw match count
            plt.suptitle(
                f"Match Verification: {pair_key} @ Step {step}, Time {frame_time_id}\n"
                f"Raw matches: {M} | Showing top {len(pts_i_vis)} numbered | Confidence Mean: {conf_vis.mean():.3f}",
                fontsize=10, fontweight='normal', y=0.98
            )
            
            # Add colorbar for confidence
            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable
            norm = Normalize(vmin=confidence.min(), vmax=confidence.max())
            sm = ScalarMappable(cmap='RdYlGn', norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02, shrink=0.8)
            cbar.set_label('Match Confidence', rotation=270, labelpad=15)
            
            # Save
            save_path = f"{save_dir}/match_verify_step{step:06d}_pair{pair_key.replace('->', '_')}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            
            print(f"  Saved: {save_path}")
        
        print(f"{'='*80}\n")
    
    @torch.no_grad()
    def save_checkpoint(self, step, global_tic=None):
        """
        Save training checkpoint and optionally stats
        
        Args:
            step: Current training step
            prefix: Prefix for filename (e.g., "final_", "")
        """
        if step != self.cfg.max_steps:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            stats = {
                "step": step,
                "memory_allocated": mem,
                "ellipse_time": time.time() - global_tic,
                "num_GS": len(self.splats["means"]) 
            }
            stats_filename = f"{self.stats_dir}/train_{step:04d}_rank{self.world_rank}.json"
            with open(stats_filename, "w") as f:
                json.dump(stats, f)
        
        # Save model checkpoint
        data = {"step": step, "splats": self.splats.state_dict()}
        
        # Add deformation module if enabled
        if cfg.deform_opt and self._is_deform_phase:
            if self.world_size > 1:
                data["deform_module"] = self.deformation.module.state_dict()
            else:
                data["deform_module"] = self.deformation.state_dict()
        
        if step == self.cfg.max_steps:
            step = "final"
        ckpt_filename = f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
        torch.save(data, ckpt_filename)
        
        print(f"Checkpoint saved: {ckpt_filename}")

    @torch.no_grad()
    def save_ply(self, step):
        """
        Save Gaussian splats as PLY file
        
        Args:
            step: Current training step
            prefix: Prefix for filename (e.g., "final_", "clean_")
            sh_degree_to_use: SH degree to use (if None, use default)
        """
        # Prepare appearance data
        sh0 = self.splats["sh0"]
        shN = self.splats["shN"]
        
        # Prepare geometric data
        means = self.splats["means"]
        scales = self.splats["scales"]
        quats = self.splats["quats"]
        opacities = self.splats["opacities"]
        
        # Determine filename
        if step == self.cfg.max_steps:
            step = "final"
        ply_filename = f"{self.ply_dir}/point_cloud_{step}.ply"
        
        # Export PLY
        export_splats(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            format="ply",
            save_to=ply_filename,
        )
        
        print(f"PLY saved: {ply_filename} ({len(means)} Gaussians)")
    
    @torch.no_grad()
    def save_gauss(self, times: torch.Tensor, final_params: Dict[str, torch.Tensor], step: int = None):
        """
        Save time-specific deformed canonical Gaussians, auxiliary Gaussians, and combined Gaussians as PLY files.
        
        Args:
            times: Tensor of time values, [B].
            final_params: Final Gaussian parameters.
            step: Current training step (for filename; defaults to 'final' if None).
        """
        if step is None or step == self.cfg.max_steps:
            step_str = "final"
        else:
            step_str = f"{step:06d}"
        
        # Canonical count (for splitting combined into canonical + aux)
        N = len(self.splats["means"])
        
        # Prepare directories
        cano_dir = f"{self.ply_dir}/cano_deformed"
        os.makedirs(cano_dir, exist_ok=True)
            
        # Extract parts
        # Deformed Canonical (first N Gaussians)
        canonical_means = final_params.means[:N]
        canonical_scales = final_params.scales[:N]
        canonical_quats = final_params.rotations[:N]
        canonical_opacities = final_params.opacities[:N]
        canonical_colors = final_params.colors[:N]  # [N, K, 3]
        
        # Split SH0 and SHN for export
        canonical_sh0 = canonical_colors[:, 0:1]  # [N, 3]
        canonical_shN = canonical_colors[:, 1:]   # [N, K-1, 3]

        t = int(times[0] * self.max_time_id)

        cano_filename = f"{cano_dir}/step_{step_str}_time_{t}.ply"
        export_splats(
            means=canonical_means,
            scales=canonical_scales,
            quats=canonical_quats,
            opacities=canonical_opacities,
            sh0=canonical_sh0,
            shN=canonical_shN,
            format="ply",
            save_to=cano_filename,
        )
        print(f"  Canonical deformed saved: {cano_filename} ({N} Gaussians)")
    
    @torch.no_grad()
    def vis_depth(self, step, gt_rgb, render_rgb, rendered_depth, metric_depth, valid_mask, save_dir):
        """
        Visualize RGB images with rendered depth and monocular depth side-by-side.
        Focus on RANKING consistency rather than absolute differences.
        """
        os.makedirs(save_dir, exist_ok=True)

        # --- Prepare Inputs ---
        gt_rgb_np = gt_rgb.detach().cpu().numpy()                 # [B, H, W, 3]
        render_rgb_np = render_rgb.detach().cpu().numpy()         # [B, H, W, 3]
        rendered_depth_np = rendered_depth.detach().cpu().numpy() # [B, H, W]
        metric_depth_np = metric_depth.detach().cpu().numpy()     # [B, H, W]
        
        if valid_mask is not None:
            valid_mask_np = valid_mask.detach().cpu().numpy()   # [B, H, W]
        else:
            valid_mask_np = np.ones_like(rendered_depth_np)

        # --- Visualization per image ---
        B = gt_rgb_np.shape[0]
        total_maps = 4  # GT RGB, Rendered RGB, Mono Depth, Rendered Depth
        n_cols = 2
        n_rows = (total_maps + n_cols - 1) // n_cols
        
        for b in range(B):
            gt_rgb_img = gt_rgb_np[b]                 # [H, W, 3]
            render_rgb_img = render_rgb_np[b]         # [H, W, 3]
            rendered_depth_img = rendered_depth_np[b] # [H, W]
            metric_depth_img = metric_depth_np[b]         # [H, W]
            valid_mask_img = valid_mask_np[b]         # [H, W]
            
            # Apply mask to depths for visualization
            rendered_depth_masked = np.where(valid_mask_img > 0.5, rendered_depth_img, np.nan)
            metric_depth_masked = np.where(valid_mask_img > 0.5, metric_depth_img, np.nan)

            # Create subplot grid
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axs = axs.flatten()

            # 1. GT RGB
            axs[0].imshow(gt_rgb_img)
            axs[0].set_title("Ground Truth RGB")
            axs[0].axis('off')

            # 2. Rendered RGB
            axs[1].imshow(render_rgb_img)
            axs[1].set_title("Rendered RGB")
            axs[1].axis('off')

            # 3. Monocular Depth (normalized) - show original values
            im2 = axs[2].imshow(metric_depth_masked, cmap='jet')
            valid_metric = metric_depth_masked[~np.isnan(metric_depth_masked)]
            if len(valid_metric) > 0:
                mono_stats = f"min={valid_metric.min():.3f}, max={valid_metric.max():.3f}"
            else:
                mono_stats = "no valid depth"
            axs[2].set_title(f"Pre-extracted Depth (Metric)\n{mono_stats}")
            axs[2].axis('off')
            fig.colorbar(im2, ax=axs[2], shrink=0.7)

            # 4. Rendered Depth (metric) - show original values
            im3 = axs[3].imshow(rendered_depth_masked, cmap='jet')
            valid_rendered = rendered_depth_masked[~np.isnan(rendered_depth_masked)]
            if len(valid_rendered) > 0:
                depth_stats = f"min={valid_rendered.min():.2f}m, max={valid_rendered.max():.2f}m"
            else:
                depth_stats = "no valid depth"
            axs[3].set_title(f"Rendered Depth (Metric)\n{depth_stats}")
            axs[3].axis('off')
            fig.colorbar(im3, ax=axs[3], shrink=0.7)

            # Hide unused subplots
            for j in range(total_maps, len(axs)):
                axs[j].axis('off')

            plt.tight_layout()
            save_path = f"{save_dir}/depth_vis_{step}_lens{b+1:02d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved depth visualization: {save_path}")
    
    @torch.no_grad()
    def log_info(self, step, deform_params, means2d, gt_rgb, render_rgb, soft_mask, dyn_mask, H, W, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        # --- Prepare Inputs ---
        dynamicness = torch.sigmoid(self.splats["dynamicness"]).detach().cpu().numpy()    # [N]
        if deform_params.dynamic_offset is not None:
            all_dynamicness = torch.sigmoid(self.splats["dynamicness"] + deform_params.dynamic_offset).detach().cpu().numpy()
            dyn_offset = deform_params.dynamic_offset.detach().cpu().numpy()
        else:
            all_dynamicness = dynamicness
            dyn_offset = np.zeros_like(dynamicness)
        mean_mag = deform_params.means_offset.norm(dim=-1).detach().cpu().numpy()         # [N]
        color_mag = deform_params.colors_offset.norm(dim=(-2, -1)).detach().cpu().numpy() # [N]
        scales_offset = deform_params.scales_offset.norm(dim=-1).detach().cpu().numpy()     # [N]
        opacity_mag = deform_params.opacities_offset.abs().detach().cpu().numpy().flatten() # [N]
        quat = deform_params.rotations_offset       # [N, 4], unit quaternions
        w = quat[..., 0].clamp(-1 + 1e-6, 1- 1e-6)  # Clamp the 'w' component to avoid domain errors with acos
        angle_rad = 2.0 * torch.acos(torch.abs(w))  # [N], in radians
        angle_deg = torch.rad2deg(angle_rad).cpu().numpy() # [N], in degrees

        means2d_np = means2d.detach().cpu().numpy()       # [B, N, 2]
        gt_rgb_np = gt_rgb.detach().cpu().numpy()         # [B, H, W, 3]
        render_rgb_np = render_rgb.detach().cpu().numpy() # [B, H, W, 3]
        dyn_mask_np = dyn_mask.detach().cpu().numpy()     # [B, H, W]
        soft_mask_np = soft_mask.detach().cpu().numpy()   # [B, H, W]

        # --------- Global Histogram and Scatter --------
        save_hist(
            data=dynamicness, 
            title=f"Canonical Dynamicness Distribution @ step {step}", 
            xlabel="Dynamic Score", 
            filename=f"hist_canonical_dynamicness_{step}.png",
            save_dir=save_dir
        )
        save_hist(
            data=all_dynamicness, 
            title=f"Combined Dynamicness Distribution @ step {step}", 
            xlabel="Dynamic Score", 
            filename=f"hist_combined_dynamicness_{step}.png",
            save_dir=save_dir
        )
        save_hist(
            data=dyn_offset, 
            title=f"Delta Dynamicness Distribution @ step {step}", 
            xlabel="Dynamic Score", 
            filename=f"hist_delta_dynamicness_{step}.png",
            save_dir=save_dir
        )

        save_scatter(
            x=all_dynamicness, y=mean_mag, 
            xlabel="Combined Dynamic Score", 
            ylabel="Means Magnitude", 
            filename=f"scatter_dynamic_offset_{step}.png",
            save_dir=save_dir, step=step
        )

        # --------- Visualization per image ---------
        B = means2d_np.shape[0]
        for b in range(B):
            u, v = means2d_np[b, :, 0], means2d_np[b, :, 1]
            gt_rgb_img = gt_rgb_np[b]
            render_rgb_img = render_rgb_np[b]
            dyn_mask_img = dyn_mask_np[b]
            soft_mask_img = soft_mask_np[b]

            # Prepare all maps
            maps = {
                "Canonical Dynamicness": (scatter_map(H, W, u, v, dynamicness), 'inferno'),
                "Delta Dynamicness": (scatter_map(H, W, u, v, dyn_offset), 'inferno'),
                "Combined Dynamicness": (scatter_map(H, W, u, v, all_dynamicness), 'inferno'),
                "Mean Magnitude": (scatter_map(H, W, u, v, mean_mag), 'plasma'),
                "Scale Magnitude": (scatter_map(H, W, u, v, scales_offset), 'plasma'),
                "Rotation Magnitude": (scatter_map(H, W, u, v, angle_deg), 'plasma'),
                "Opacity Magnitude": (scatter_map(H, W, u, v, opacity_mag), 'bone'),
                "SH Coefficient Magnitude": (scatter_map(H, W, u, v, color_mag), 'viridis'),
            }

            total_maps = len(maps) + 1  # +1 for RGB
            n_cols = min(4, total_maps)
            n_rows = (total_maps + n_cols - 1) // n_cols
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            axs = axs.flatten()

            axs[0].imshow(gt_rgb_img)
            axs[0].set_title("Ground Truth")
            axs[0].axis('off')

            axs[1].imshow(render_rgb_img)
            axs[1].set_title("Rendered Image")
            axs[1].axis('off')

            im2 = axs[2].imshow(dyn_mask_img, cmap='hot')
            axs[2].set_title("Per-frame Binary Mask")
            axs[2].axis('off')
            fig.colorbar(im2, ax=axs[2], shrink=0.7)

            im3 = axs[3].imshow(soft_mask_img, cmap='hot')
            axs[3].set_title("Per-view Soft Mask")
            axs[3].axis('off')
            fig.colorbar(im3, ax=axs[3], shrink=0.7)

            for i, (name, (img, cmap)) in enumerate(maps.items(), start=4):
                im = axs[i].imshow(img, cmap=cmap)
                axs[i].set_title(f"{name}\nmin={img.min():.3f}, max={img.max():.3f}")
                axs[i].axis('off')
                fig.colorbar(im, ax=axs[i], shrink=0.7)

            for j in range(i + 1, len(axs)):
                axs[j].axis('off')

            plt.tight_layout()
            plt.savefig(f"{save_dir}/composite_vis_{step}_lens{b+1}.png")
            plt.close()
    
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
    def save_depth(self, gt_rgb, render_rgb, rendered_depth, valid_mask, save_dir, filename):
        """
        Visualize RGB images with rendered depth and monocular depth side-by-side.
        Focus on RANKING consistency rather than absolute differences.
        """
        os.makedirs(save_dir, exist_ok=True)

        # --- Prepare Inputs ---
        gt_rgb_np = gt_rgb.detach().cpu().numpy()                 # [H, W, 3]
        render_rgb_np = render_rgb.detach().cpu().numpy()         # [H, W, 3]
        rendered_depth_np = rendered_depth.detach().cpu().numpy() # [H, W]
        
        if valid_mask is not None:
            valid_mask_np = valid_mask.detach().cpu().numpy()   # [H, W]
        else:
            valid_mask_np = np.ones_like(rendered_depth_np)

        # --- Visualization per image ---
        total_maps = 3  # GT RGB, Rendered RGB, Rendered Depth
        n_cols = 3
        n_rows = (total_maps + n_cols - 1) // n_cols
        
        # Apply mask to depths for visualization
        rendered_depth_masked = np.where(valid_mask_np > 0.5, rendered_depth_np, np.nan)
        
        # Normalize BOTH depths to [0, 1] for fair visual comparison
        valid_both = valid_mask_np > 0.5
        rendered_norm = np.full_like(rendered_depth_np, np.nan)
        
        if valid_both.sum() > 0:
            # Normalize rendered depth to [0, 1]
            valid_rendered = rendered_depth_np[valid_both]
            r_min, r_max = valid_rendered.min(), valid_rendered.max()
            if r_max > r_min:
                rendered_norm[valid_both] = (rendered_depth_np[valid_both] - r_min) / (r_max - r_min)
            else:
                rendered_norm[valid_both] = 0.5

        # Create subplot grid
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axs = axs.flatten()

        # 1. GT RGB
        axs[0].imshow(gt_rgb_np)
        axs[0].set_title("Ground Truth RGB")
        axs[0].axis('off')

        # 2. Rendered RGB
        axs[1].imshow(render_rgb_np)
        axs[1].set_title("Rendered RGB")
        axs[1].axis('off')

        # 4. Rendered Depth (metric) - show original values
        im2 = axs[2].imshow(rendered_depth_masked, cmap='jet')
        valid_rendered = rendered_depth_masked[~np.isnan(rendered_depth_masked)]
        if len(valid_rendered) > 0:
            depth_stats = f"min={valid_rendered.min():.2f}m, max={valid_rendered.max():.2f}m"
        else:
            depth_stats = "no valid depth"
        axs[2].set_title(f"Rendered Depth (Metric)\n{depth_stats}")
        axs[2].axis('off')
        fig.colorbar(im2, ax=axs[2], shrink=0.7)

        # Hide unused subplots
        for j in range(total_maps, len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{filename}")
        plt.close()

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        if stage == "train":
            valloader = torch.utils.data.DataLoader(
                self.trainset,
                batch_sampler=CamBatchSampler(self.trainset),
                num_workers=4,
                shuffle=False,
                persistent_workers=True,
                pin_memory=True,
            )
        else:
            valloader = torch.utils.data.DataLoader(
                self.valset,
                batch_sampler=CamBatchSampler(self.valset),
                num_workers=4,
                shuffle=False,
                persistent_workers=True,
                pin_memory=True,
            )
        
        ellipse_time = 0
        total_images = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            # Track total images processed
            batch_size = data["image"].shape[0]  # Get actual batch size from data
            total_images += batch_size

            camtoworlds = data["camtoworld"].to(device)  # tensor: [6, 4, 4]
            Ks = data["K"].to(device)
            radial_coeffs = data["poly_coeffs"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            times = data["time_id"].to(device) if "time_id" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            renders, _, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                radial_coeffs=radial_coeffs,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                masks=masks,
                deform_opt=cfg.deform_opt,
                times=times
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = renders[..., :3].clamp(0.0, 1.0)
            if world_rank == 0:
                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p)) # for all pixels (not exclude the invalid regions)
                metrics["msssim"].append(self.msssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))

        if world_rank == 0:
            total_time = ellipse_time  # Store total time before averaging
            ellipse_time /= len(valloader)  # len(vallloader) is the batch length

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "fps": total_images / total_time if total_time > 0 else 0.0
                }
            )
            print(f"\nPSNR: {stats['psnr']:.3f}, MSSSIM: {stats['msssim']:.4f}, LPIPS: {stats['lpips']:.3f}, FPS: {stats['fps']:.2f}")
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()
        
            # Store final FPS for training time file
            if step == self.cfg.max_steps - 1 and stage == "val":
                self.final_fps = stats["fps"]
    
    @torch.no_grad()
    def render_all(self, use_rig_mask: bool = False):
        """Render all frames for both train and test sets"""
        print("Rendering all fisheye frames...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        os.makedirs(self.render_dir, exist_ok=True)

        # The final inference must in the deform_phase
        self._is_deform_phase = True

        # Load rig range masks once
        rig_range_masks = {}
        if use_rig_mask:
            rig_range_dir = os.path.join(os.path.dirname(cfg.data_dir), "robot_range")
            print("rig_range_dir:", rig_range_dir)
            # Get sorted mask files
            rig_range_path = Path(rig_range_dir)
            if rig_range_path.exists():
                mask_files = sorted(
                    rig_range_path.glob("lens*.npy"),
                    key=lambda x: int(x.stem[4:])  # Extract number from lens01'
                )
                if not mask_files:
                    print(f"No camera masks found in {rig_range_dir}")
                for mask_file in mask_files:
                    # Extract camera ID from filename (e.g., 'lens01.npy' -> 1)
                    cam_id = int(os.path.basename(mask_file)[4:6])
                    rig_range_masks[cam_id] = torch.from_numpy(
                        np.load(mask_file)
                    ).float().unsqueeze(0).to(device)  # [1, H, W]
        
        def render_loader(dataloader, split: str):
            """Render dataloader with specific split name"""
            pbar = tqdm.tqdm(dataloader, desc=f"Rendering {split} set")
            for i, data in enumerate(pbar):
                camtoworlds = data["camtoworld"].to(device)
                Ks = data["K"].to(device)
                radial_coeffs = data["poly_coeffs"].to(device)
                pixels = data["image"].to(device) / 255.0  # [B, H, W, 3]
                masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
                times = data["time_id"].to(device) if "time_id" in data else None # [B]
                camera_id = int(data["camera_id"][0].item()) if "camera_id" in data else 0 # [B]
                image_name = data["image_name"][0]
                height, width = pixels.shape[1:3]
                
                # Render
                renders, _, _, _, _ = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    radial_coeffs=radial_coeffs,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    masks=masks,
                    deform_opt=cfg.deform_opt,
                    times=times,
                    render_mode="RGB+ED"
                )  # [1, H, W, 3]

                # Rendered colors
                colors = renders[..., :3].clamp(0.0, 1.0)

                # Apply rig range mask and fisheye circular mask
                combined_mask = None
                if camera_id in rig_range_masks:
                    rig_mask = rig_range_masks[camera_id]
                    if masks is not None:
                        combined_mask = masks * rig_mask
                    else:
                        combined_mask = rig_mask
                else:
                    combined_mask = masks if masks is not None else None

                # -------- Dynamic-focused regions (EXCLUDE robot_range) --------
                dyn_roi = None
                if data.get("dynamic_masks", None) is not None:
                    dyn_roi = data["dynamic_masks"].to(device).float()  # [B,H,W]

                if dyn_roi is not None:
                    if combined_mask is not None:
                        dyn_roi = dyn_roi * combined_mask

                    valid_px = int(dyn_roi.sum().item())
                    if valid_px > 0:
                        # Extract bounding boxes from GT dynamic mask (rig-excluded)
                        dyn_mask_2d = dyn_roi.squeeze(0)  # [H, W]
                        bboxes = extract_dyn_bboxes(dyn_mask_2d, min_area=200, merge_close=True, bbox_dilat=8)

                        if len(bboxes) > 0:
                            if world_rank == 0:
                                patch_save_dir = os.path.join(self.render_dir, split, "bbox", f"lens0{camera_id}")
                                patch_path = os.path.join(patch_save_dir, "patches")
                                bbox_path = os.path.join(patch_save_dir, "bboxes")
                                os.makedirs(patch_save_dir, exist_ok=True)
                                os.makedirs(patch_path, exist_ok=True)
                                os.makedirs(bbox_path, exist_ok=True)
                                # Save bbox visualization on the original image
                                gt_with_bboxes = pixels[0].cpu().numpy().copy()  # [H, W, 3]
                                # Draw all bboxes
                                for j, (bx1, by1, bx2, by2) in enumerate(bboxes):
                                    bx1c = max(0, bx1); by1c = max(0, by1)
                                    bx2c = min(width, bx2); by2c = min(height, by2)
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
                                    if text_y < height - 10 and text_x < width - 10:
                                        gt_with_bboxes[text_y:text_y+8, text_x:text_x+8, :] = [1.0, 1.0, 0.0]  # yellow square as number indicator
                                # Save bbox overlay
                                bbox_overlay_path = os.path.join(bbox_path, f"{os.path.splitext(os.path.basename(image_name))[0]}.png")
                                imageio.imwrite(bbox_overlay_path, (np.clip(gt_with_bboxes, 0, 1) * 255).astype(np.uint8))
                            for i, (x1, y1, x2, y2) in enumerate(bboxes):
                                # Clamp (safety)
                                x1c = max(0, x1); y1c = max(0, y1)
                                x2c = min(width, x2); y2c = min(height, y2)
                                if x2c - x1c <= 180 and y2c - y1c <= 180:
                                    continue  # too small for stable metrics
                                # Extract patches [1, 3, h, w]
                                gt_patch = pixels[0, y1c:y2c, x1c:x2c].permute(2, 0, 1).unsqueeze(0)
                                pred_patch = colors[0, y1c:y2c, x1c:x2c].permute(2, 0, 1).unsqueeze(0)

                                # Save original patches 
                                if world_rank == 0:
                                    # Convert to numpy for saving [H, W, 3]
                                    gt_patch_np = gt_patch[0].permute(1, 2, 0).cpu().numpy()
                                    pred_patch_np = pred_patch[0].permute(1, 2, 0).cpu().numpy()
                                    
                                    # Create side-by-side comparison with gap
                                    gap_width = 4  # pixels of gap between patches
                                    gap_color = [1.0, 1.0, 1.0]  # white gap
                                    
                                    patch_h, patch_w = gt_patch_np.shape[:2]
                                    gap_strip = np.full((patch_h, gap_width, 3), gap_color, dtype=gt_patch_np.dtype)
                                    
                                    # Concatenate: GT + gap + rendered
                                    patch_comparison = np.concatenate([gt_patch_np, gap_strip, pred_patch_np], axis=1)
                                    
                                    # Save individual patches and comparison
                                    base_name = f"{os.path.splitext(os.path.basename(image_name))[0]}_bbox{(i+1):02d}"
                                    
                                    # Save side-by-side comparison
                                    comp_patch_path = os.path.join(patch_path, f"{base_name}.png")
                                    imageio.imwrite(comp_patch_path, (np.clip(patch_comparison, 0, 1) * 255).astype(np.uint8))

                # Save rendered images
                if world_rank == 0:
                    # Create split and type-specific directories
                    gt_dir = os.path.join(self.render_dir, split, "gt_final", f"lens0{camera_id}")
                    render_dir = os.path.join(self.render_dir, split, "renders_final", f"lens0{camera_id}")
                    os.makedirs(gt_dir, exist_ok=True)
                    os.makedirs(render_dir, exist_ok=True)
                    # Save GT image
                    if masks is not None:
                        # Expand mask to match image dimensions [1,H,W] -> [1,H,W,3]
                        mask_3c = masks.unsqueeze(-1).expand(-1, -1, -1, 3)
                        pixels = pixels * mask_3c  # Mask GT
                    gt_img = (pixels.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                    gt_path = os.path.join(gt_dir, f"{os.path.basename(image_name)}")
                    imageio.imwrite(gt_path, gt_img)
                    # Save rendered image
                    render_img = (colors.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                    render_path = os.path.join(render_dir, f"{os.path.basename(image_name)}")
                    imageio.imwrite(render_path, render_img)
        
        # Setup data loaders
        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=1, shuffle=False, num_workers=0)
        testloader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=0)

        # Rendering frames for train and test sets
        render_loader(trainloader, "train")
        render_loader(testloader, "test")
    
    @torch.no_grad()
    def evaluate_depth(self):
        """
        Evaluate rendered depth against GT metric depths.
        Computes absolute depth metrics (abs_rel, sq_rel, rmse, etc.) on valid pixels.
        Saves depth maps and per-frame metrics for both train and test splits.
        """
        print("\n" + "="*60)
        print("Starting Depth Evaluation...")
        print("="*60)
        
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        
        # Must be in deform phase for final inference
        self._is_deform_phase = True
        
        def compute_depth_metrics(gt_depth, pred_depth, valid_mask):
            """
            Compute standard depth metrics on valid pixels.
            
            Args:
                gt_depth: [H, W] ground truth metric depth
                pred_depth: [H, W] rendered depth
                valid_mask: [H, W] binary mask (1.0 = valid)
            
            Returns:
                dict of metrics
            """
            # Filter valid pixels
            valid = valid_mask > 0.5
            if valid.sum() == 0:
                return None
            
            gt_valid = gt_depth[valid]
            pred_valid = pred_depth[valid]
            
            # Absolute metrics
            abs_diff = torch.abs(gt_valid - pred_valid)
            mae = abs_diff.mean()
            abs_rel = (abs_diff / (gt_valid + 1e-6)).mean()
            sq_rel = ((abs_diff ** 2) / (gt_valid + 1e-6)).mean()
            rmse = torch.sqrt((abs_diff ** 2).mean())
            rmse_log = torch.sqrt(((torch.log(gt_valid + 1e-6) - torch.log(pred_valid + 1e-6)) ** 2).mean())
            
            # Threshold accuracy (? < 1.25, 1.25^2, 1.25^3)
            thresh = torch.max((gt_valid / (pred_valid + 1e-6)), (pred_valid / (gt_valid + 1e-6)))
            a1 = (thresh < 1.25).float().mean()
            a2 = (thresh < 1.25 ** 2).float().mean()
            a3 = (thresh < 1.25 ** 3).float().mean()
            
            return {
                'mae': mae.item(),
                'abs_rel': abs_rel.item(),
                'sq_rel': sq_rel.item(),
                'rmse': rmse.item(),
                'rmse_log': rmse_log.item(),
                'a1': a1.item(),
                'a2': a2.item(),
                'a3': a3.item(),
            }
        
        def evaluate_split(dataloader, split: str):
            """Evaluate depth for a specific split"""
            print(f"\nEvaluating {split} split...")
            
            metrics_dict = defaultdict(lambda: defaultdict(list))  # {lens_id: {metric: [values]}}
            frame_metrics = {}
            
            pbar = tqdm.tqdm(dataloader, desc=f"Processing {split}")
            for i, data in enumerate(pbar):
                camtoworlds = data["camtoworld"].to(device)
                Ks = data["K"].to(device)
                radial_coeffs = data["poly_coeffs"].to(device)
                masks = data["mask"].to(device) if "mask" in data else None
                times = data["time_id"].to(device) if "time_id" in data else None
                camera_id = int(data["camera_id"][0].item()) if "camera_id" in data else 0
                image_name = data["image_name"][0]
                height, width = data["image"].shape[1:3]
                
                # Load GT metric depth and eroded mask
                metric_depths = data.get("metric_depths", None)
                eroded_masks = data.get("eroded_mask", None)
                
                if metric_depths is None or eroded_masks is None:
                    print(f"[Warning] Skipping {image_name}: missing metric_depth or eroded_mask")
                    continue
                
                metric_depths = metric_depths.to(device)  # [B, H, W]
                eroded_masks = eroded_masks.to(device)    # [B, H, W]
                
                # Render depth
                renders, _, _, _, _ = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    radial_coeffs=radial_coeffs,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    masks=masks,
                    deform_opt=cfg.deform_opt,
                    times=times,
                    render_mode="RGB+ED"
                )
                
                # Extract depth [B, H, W]
                rendered_depth = renders[..., 3].squeeze(0)  # [H, W]
                gt_depth = metric_depths.squeeze(0)          # [H, W]
                valid_mask = eroded_masks.squeeze(0)         # [H, W]
                
                # Compute metrics
                depth_metrics = compute_depth_metrics(gt_depth, rendered_depth, valid_mask)
                
                if depth_metrics is None:
                    print(f"[Warning] No valid pixels for {image_name}")
                    continue
                
                # Store metrics
                for metric, value in depth_metrics.items():
                    metrics_dict[camera_id][metric].append(value)
                
                frame_metrics[f"{split}_{image_name}"] = depth_metrics
                
                # # Save depth maps
                # if world_rank == 0:
                #     # Create directory structure
                #     gt_depth_dir = os.path.join(self.render_dir, split, "gt_depth", f"lens{camera_id:02d}")
                #     render_depth_dir = os.path.join(self.render_dir, split, "rendered_depth", f"lens{camera_id:02d}")
                #     os.makedirs(gt_depth_dir, exist_ok=True)
                #     os.makedirs(render_depth_dir, exist_ok=True)
                    
                #     # Convert to numpy
                #     gt_depth_np = gt_depth.cpu().numpy()
                #     rendered_depth_np = rendered_depth.cpu().numpy()
                #     # Use eroded_mask for visualization (same mask used for metrics)
                #     vis_mask_np = valid_mask.cpu().numpy()
                    
                #     # Apply mask: invalid regions = nan (will be white)
                #     gt_depth_masked = np.where(vis_mask_np > 0.5, gt_depth_np, np.nan)
                #     rendered_depth_masked = np.where(vis_mask_np > 0.5, rendered_depth_np, np.nan)
                    
                #     # Apply colormap directly (nan pixels will be white)
                #     cmap = plt.cm.jet
                #     cmap.set_bad(color='white')  # Set invalid (nan) pixels to white
                    
                #     # Create figure for GT depth (no axes, no colorbar)
                #     fig_gt = plt.figure(frameon=False)
                #     fig_gt.set_size_inches(gt_depth_np.shape[1]/100, gt_depth_np.shape[0]/100)
                #     ax_gt = plt.Axes(fig_gt, [0., 0., 1., 1.])
                #     ax_gt.set_axis_off()
                #     fig_gt.add_axes(ax_gt)
                #     ax_gt.imshow(gt_depth_masked, cmap=cmap, aspect='auto')
                    
                #     # Save GT depth
                #     frame_name = Path(image_name).stem
                #     gt_path = os.path.join(gt_depth_dir, f"{frame_name}.png")
                #     fig_gt.savefig(gt_path, dpi=100, bbox_inches='tight', pad_inches=0)
                #     plt.close(fig_gt)
                    
                #     # Create figure for rendered depth (no axes, no colorbar)
                #     fig_render = plt.figure(frameon=False)
                #     fig_render.set_size_inches(rendered_depth_np.shape[1]/100, rendered_depth_np.shape[0]/100)
                #     ax_render = plt.Axes(fig_render, [0., 0., 1., 1.])
                #     ax_render.set_axis_off()
                #     fig_render.add_axes(ax_render)
                #     ax_render.imshow(rendered_depth_masked, cmap=cmap, aspect='auto')
                    
                #     # Save rendered depth
                #     render_path = os.path.join(render_depth_dir, f"{frame_name}.png")
                #     fig_render.savefig(render_path, dpi=100, bbox_inches='tight', pad_inches=0)
                #     plt.close(fig_render)

                # Save depth maps
                if world_rank == 0:
                    # Create directory structure
                    gt_depth_dir = os.path.join(self.render_dir, split, "gt_depth", f"lens{camera_id:02d}")
                    render_depth_dir = os.path.join(self.render_dir, split, "rendered_depth", f"lens{camera_id:02d}")
                    os.makedirs(gt_depth_dir, exist_ok=True)
                    os.makedirs(render_depth_dir, exist_ok=True)
                    
                    # Convert to numpy
                    gt_depth_np = gt_depth.cpu().numpy()
                    rendered_depth_np = rendered_depth.cpu().numpy()
                    # Use eroded_mask for visualization (same mask used for metrics)
                    vis_mask_np = valid_mask.cpu().numpy()
                    
                    # Apply mask: invalid regions = nan (will be white)
                    gt_depth_masked = np.where(vis_mask_np > 0.5, gt_depth_np, np.nan)
                    rendered_depth_masked = np.where(vis_mask_np > 0.5, rendered_depth_np, np.nan)
                    
                    # Apply colormap directly (nan pixels will be white)
                    cmap = plt.cm.jet
                    cmap.set_bad(color='white')  # Set invalid (nan) pixels to white
                    
                    # Save GT depth with colorbar (auto-scale for each image independently)
                    frame_name = Path(image_name).stem
                    gt_path = os.path.join(gt_depth_dir, f"{frame_name}.png")
                    
                    # Save depth maps
                if world_rank == 0:
                    # Create directory structure
                    gt_depth_dir = os.path.join(self.render_dir, split, "gt_depth", f"lens{camera_id:02d}")
                    render_depth_dir = os.path.join(self.render_dir, split, "rendered_depth", f"lens{camera_id:02d}")
                    os.makedirs(gt_depth_dir, exist_ok=True)
                    os.makedirs(render_depth_dir, exist_ok=True)
                    
                    # Convert to numpy
                    gt_depth_np = gt_depth.cpu().numpy()
                    rendered_depth_np = rendered_depth.cpu().numpy()
                    # Use eroded_mask for visualization (same mask used for metrics)
                    vis_mask_np = valid_mask.cpu().numpy()
                    
                    # Apply mask: invalid regions = nan (will be white)
                    gt_depth_masked = np.where(vis_mask_np > 0.5, gt_depth_np, np.nan)
                    rendered_depth_masked = np.where(vis_mask_np > 0.5, rendered_depth_np, np.nan)
                    
                    # Apply colormap directly (nan pixels will be white)
                    cmap = plt.cm.jet
                    cmap.set_bad(color='white')  # Set invalid (nan) pixels to white
                    
                    # Save GT depth with colorbar (auto-scale for each image independently)
                    frame_name = Path(image_name).stem
                    gt_path = os.path.join(gt_depth_dir, f"{frame_name}.png")
                    
                    # Use shrink parameter to control colorbar height
                    H, W = gt_depth_np.shape
                    fig_gt, ax_gt = plt.subplots(figsize=(W/100, H/100), dpi=100)
                    ax_gt.set_axis_off()
                    im_gt = ax_gt.imshow(gt_depth_masked, cmap=cmap, aspect='auto')  # Auto-scale like original
                    
                    cbar_gt = plt.colorbar(im_gt, ax=ax_gt, fraction=0.05, pad=0.005, shrink=0.7)
                    cbar_gt.set_label('Depth (m)', rotation=270, labelpad=10, fontsize=10)
                    cbar_gt.ax.tick_params(labelsize=10)
                    
                    fig_gt.savefig(gt_path, dpi=100, bbox_inches='tight', pad_inches=0)
                    plt.close(fig_gt)
                    
                    # Save rendered depth with colorbar (auto-scale independently)
                    render_path = os.path.join(render_depth_dir, f"{frame_name}.png")
                    
                    fig_render, ax_render = plt.subplots(figsize=(W/100, H/100), dpi=100)
                    ax_render.set_axis_off()
                    im_render = ax_render.imshow(rendered_depth_masked, cmap=cmap, aspect='auto')  # Auto-scale like original
                    
                    cbar_render = plt.colorbar(im_render, ax=ax_render, fraction=0.05, pad=0.005, shrink=0.7)
                    cbar_render.set_label('Depth (m)', rotation=270, labelpad=10, fontsize=10)
                    cbar_render.ax.tick_params(labelsize=10)
                    
                    fig_render.savefig(render_path, dpi=100, bbox_inches='tight', pad_inches=0)
                    plt.close(fig_render)
            
            return metrics_dict, frame_metrics
        
        # Setup dataloaders
        trainloader = torch.utils.data.DataLoader(
            self.trainset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=0
        )
        testloader = torch.utils.data.DataLoader(
            self.valset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=0
        )
        
        # Evaluate both splits
        train_metrics, train_frame_metrics = evaluate_split(trainloader, "train")
        test_metrics, test_frame_metrics = evaluate_split(testloader, "test")
        
        if world_rank == 0:
            from utils import calculate_stats, save_depth_stats_to_csv
            
            # Combine frame metrics
            all_frame_metrics = {**train_frame_metrics, **test_frame_metrics}
            
            # Save per-frame metrics
            frame_metrics_path = os.path.join(self.stats_dir, "depth_frame_metrics.json")
            with open(frame_metrics_path, "w") as f:
                json.dump({"per_frame": all_frame_metrics}, f, indent=2)
            print(f"\n Saved per-frame depth metrics to: {frame_metrics_path}")
            
            # Compute overall statistics (mean ± std format)
            overall_stats = {
                split: {
                    metric: calculate_stats([
                        v for cam_id, cam_metrics in metrics.items()
                        for m, values in cam_metrics.items()
                        if m == metric
                        for v in values
                    ])
                    for metric in ['mae', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
                }
                for split, metrics in [("train", train_metrics), ("test", test_metrics)]
            }
            
            # Compute per-lens statistics (mean ± std format)
            per_lens_stats = {
                split: {
                    f"lens{int(cam_id):02d}": {
                        metric: calculate_stats(values)
                        for metric, values in cam_metrics.items()
                    }
                    for cam_id, cam_metrics in metrics.items()
                }
                for split, metrics in [("train", train_metrics), ("test", test_metrics)]
            }
            
            # Save overall statistics (TXT)
            overall_txt_path = os.path.join(self.stats_dir, "depth_overall_stats.txt")
            with open(overall_txt_path, "w", encoding='utf-8') as f:
                for split, split_stats in overall_stats.items():
                    f.write(f"\n{split.upper()} Set Statistics:\n")
                    for metric, stats in split_stats.items():
                        f.write(f"{metric}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})\n")
            print(f" Saved overall depth stats to: {overall_txt_path}")
            
            # Save overall statistics (CSV) - using the formatted function
            overall_csv_path = os.path.join(self.stats_dir, "depth_overall_stats.csv")
            save_depth_stats_to_csv(overall_stats, overall_csv_path)
            
            # Save per-lens statistics (TXT)
            lens_txt_path = os.path.join(self.stats_dir, "depth_lens_stats.txt")
            with open(lens_txt_path, "w", encoding='utf-8') as f:
                for split, lenses in per_lens_stats.items():
                    f.write(f"\n{split.upper()} Set Per-Lens Statistics:\n")
                    for lens, metrics in lenses.items():
                        f.write(f"\n{lens}:\n")
                        for metric, stats in metrics.items():
                            f.write(f"  {metric}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})\n")
            print(f" Saved per-lens depth stats to: {lens_txt_path}")
            
            # Also save JSON summary for compatibility
            depth_summary = {
                "overall": overall_stats,
                "per_lens": per_lens_stats
            }
            
            summary_path = os.path.join(self.stats_dir, "depth_metrics_summary.json")
            with open(summary_path, "w") as f:
                json.dump(depth_summary, f, indent=2)
            print(f" Saved depth metrics summary (JSON) to: {summary_path}")
            
            # Print summary
            print("\n" + "="*60)
            print("DEPTH EVALUATION SUMMARY")
            print("="*60)
            for split in ["train", "test"]:
                print(f"\n{split.upper()} Split:")
                stats = overall_stats.get(split, {})
                if stats:
                    for metric_name in ['mae', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']:
                        metric_stats = stats.get(metric_name, {})
                        if metric_stats:
                            print(f"  {metric_name:9s}: {metric_stats['mean']:.4f} ± {metric_stats['std']:.4f}")
            print("="*60 + "\n")
    
    @torch.no_grad()
    def _viewer_render_fn(self, camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device) / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


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

        # Load the current step
        step = ckpts[0]["step"]

        # Load splats parameters
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        
        # Load deform module parameters
        if "deform_module" in ckpts[0]:
            if world_size > 1:
                runner.deformation.module.load_state_dict(ckpts[0]["deform_module"])
            else:
                runner.deformation.load_state_dict(ckpts[0]["deform_module"])

        if hasattr(cfg, 'eval_depth') and cfg.eval_depth:
            # Evaluate depth maps
            print("\nRunning depth evaluation from checkpoint...")
            runner.evaluate_depth()
        else:
            # Resume training (original behavior)
            runner.train(start_step=step+1)
        # runner.enhanced_eval(use_rig_mask=True)
        
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python trainer_update.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python trainer_update.py default --steps_scaler 0.25

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
