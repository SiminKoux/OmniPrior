"""
Architecture:
- AdaptiveMonoRankingLoss: Single-view depth ranking with adaptive decay
- DynamicAwareGuidanceLoss: Guidance for dynamicness optimization
- MetricDepthLoss: Supervise metric depth
- DepthsmoothLoss: Smoothness loss on depth maps
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional

class DynamicAwareGuidanceLoss:
    def __init__(self, config: Dict):
        self.config = config

    def dynamic_guidance_loss(
        self,
        dynamicness,                 # [N,] dynamicness probability in logit space
        info: dict,                  # {"means2d":[B, N, 2], "radii":[B, N, 2]}
        valid_mask: torch.Tensor,    # [B, H, W] in {0,1}
        dynamic_masks: torch.Tensor, # [B, H, W] in {0,1}
    ):
        """
        Basic guidance loss using BCE Loss in logit space.
        
        Args:
            dynamicness: [N,] predicted dynamicness scores (logits)
            info: dictionary with rendering info (means2d, radii)
            dynamic_masks: [B, H, W] ground truth binary masks
            valid_masks: [B, H, W] valid region masks
        Returns:
            guidance_loss: total guidance loss
            loss_static: static region loss
            loss_dynamic: dynamic region loss
        """
        device = dynamic_masks.device
        Bv, H, W = dynamic_masks.shape
        means2d = info["means2d"]  # [B, N, 2]
        radii = info["radii"]      # [B, N, 2]

        assert means2d.shape[0] == Bv, "Batch mismatch between means2d and masks"
        N = means2d.shape[1]
        
        # Integer pixel indices for grouping / GT lookup
        x = means2d[..., 0].long().clamp_(0, W - 1)           # [B, N]
        y = means2d[..., 1].long().clamp_(0, H - 1)           # [B, N]
        cam_ids = torch.arange(Bv, device=device).unsqueeze(1).expand(Bv, N)  # [B, N]

        # Visibility
        if radii.ndim == 3:
            valid_vis = torch.sqrt((radii ** 2).sum(dim=-1)) > 0 # [B, N] bool
        else:
            valid_vis = radii > 0  # [B, N] bool
        # Validity
        if valid_mask is not None:
            valid_depth_float = valid_mask[cam_ids, y, x]  # [B, N] float (0.0 or 1.0)
            valid_depth = valid_depth_float > 0.5          # [B, N] bool
        else:
            valid_depth = torch.ones(B, N, dtype=torch.bool, device=device)  # All valid if no mask, [B, N] bool
        # Combined valid mask
        valid_comb = valid_vis & valid_depth     # [B, N], bool

        # Per-instance GT and scores
        # Indexing produces [B, N], then mask->flatten to [M_vis]
        gt_full = dynamic_masks[cam_ids, y, x]               # [B, N]
        score_full = dynamicness.unsqueeze(0).expand(Bv, N)  # [B, N]

        # Filter valid instances
        gt_vec = gt_full[valid_comb]        # [M_vis]
        score_vec = score_full[valid_comb]  # [M_vis]

        # Numerical guards (light touch)
        score_vec = torch.where(torch.isfinite(score_vec), score_vec, torch.zeros_like(score_vec))

        # Split by GT binary masks
        static_mask  = (gt_vec == 0.0)
        dynamic_mask = ~static_mask

        # Initialize losses
        loss_static  = torch.tensor(0.0, device=device)
        loss_dynamic = torch.tensor(0.0, device=device)

        # BCE losses for static and dynamic regions
        if static_mask.any():
            loss_static = F.binary_cross_entropy_with_logits(
                input=score_vec[static_mask], 
                target=gt_vec[static_mask].float(), 
                reduction='mean')
        if dynamic_mask.any():
            loss_dynamic = F.binary_cross_entropy_with_logits(
                input=score_vec[dynamic_mask], 
                target=gt_vec[dynamic_mask].float(),
                reduction='mean'
            )
        guidance_loss = loss_static + loss_dynamic
        return guidance_loss, loss_static, loss_dynamic 
    
    def __call__(
        self,
        dynamicness: torch.Tensor,
        info: Dict,
        dynamic_masks: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Complete guidance loss with depth fusion.
        Args:
            dynamicness: [N] predicted dynamicness scores
            info: dictionary with rendering info (means2d, radii, etc)
            dynamic_masks: [B, H, W] ground truth binary masks
            valid_masks: [B, H, W] valid region masks
            step: current training step for scheduling

        Returns:
            loss_dict: individual components for logging
        """
        guidance_loss, loss_static, loss_dynamic = self.dynamic_guidance_loss(
            dynamicness, info, valid_mask, dynamic_masks
        )

        # Package for logging
        loss_dict = {
            "guidance": guidance_loss,
            "guidance_static": loss_static,
            "guidance_dynamic": loss_dynamic,
        }
        
        return loss_dict

class AdaptiveMonoRankingLoss(nn.Module):
    """
    Single-view depth ranking loss with adaptive weighting.
    
    Key features:
    - Stratified sampling (hard/medium/easy pairs)
    - Distance caps to avoid edge artifacts
    - Adaptive weight decay after geometry stabilizes
    - All pairs stay local to avoid fisheye distortion and edge artifacts
    - Reduces supervision in late training to avoid color conflicts
    
    Scheduling (30k iterations):
        0-1k:    Off (pre-ranking, let RGB converge)
        1k-5k:   Ramp up 0->1.0 (coarse geometry initialization)
        5k-15k:  Full weight 1.0 (depth refinement)
        15k-25k: Gradual decay 1.0->0.2 (reduce after geometry stabilizes)
        25k-30k: Minimal 0.2 (avoid color conflicts in late training)
    """
    def __init__(
        self,
        margin: float = 0.1,       # Margin for depth ranking
        sample_pairs: int = 1024,  # Number of pairs to sample
        min_distance: int = 10,    # Minimum distance for valid pairs
        pre_rank_end: int = 1000,  # No ranking before this
        ramp_up_end: int = 5000,   # Ramp 0->1
        peak_end: int = 15000,     # Full weight
        decay_end: int = 30000,    # Decay 1.0->0.5
        min_weight: float = 0.5,   # Minimal weight in late training
    ):
        super().__init__()
        
        # Loss parameters
        self.margin = margin
        self.sample_pairs = sample_pairs
        self.min_distance = min_distance
        
        # Scheduling parameters
        self.pre_rank_end = pre_rank_end
        self.ramp_up_end = ramp_up_end
        self.peak_end = peak_end
        self.decay_end = decay_end
        self.min_weight = min_weight

        # Visualization cache
        self._vis_cache: Optional[Dict] = None
        
    def _compute_weight(self, step: int) -> float:
        """Compute adaptive weight based on training stage."""
        
        # Stage 1: Pre-ranking (no supervision)
        if step < self.pre_rank_end:
            return 0.0
        
        # Stage 2: Ramp up (coarse geometry)
        if step < self.ramp_up_end:
            progress = (step - self.pre_rank_end) / (self.ramp_up_end - self.pre_rank_end)
            return progress
        
        # Stage 3: Peak (full supervision)
        if step < self.peak_end:
            return 1.0
        
        # Stage 4: Decay (reduce after geometry stabilizes)
        if step < self.decay_end:
            progress = (step - self.peak_end) / (self.decay_end - self.peak_end)
            # Smooth decay: 1.0 -> min_weight
            return 1.0 - (1.0 - self.min_weight) * progress
        
        # Stage 5: Minimal (late training)
        return self.min_weight
    
    def _sampling(self, valid_coords, mono_depth_view, n_pairs, device):
        """
        Stratified sampling with distance caps to avoid long-distance pairs.
        
        Strategy:
        - 40% hard pairs (small depth diff, short distance)
        - 30% medium pairs (medium depth diff, medium distance)
        - 30% easy pairs (large depth diff, moderate distance)
        
        Key features:
        - Distance caps for each difficulty level (no spanning entire image)
        - Weight easy pairs by depth difference (not distance) to avoid edge bias
        - All pairs stay local to avoid fisheye distortion and edge artifacts
        """
        n_valid = len(valid_coords[0])
        
        if n_valid < 2:
            return torch.empty((0, 2), dtype=torch.long, device=device)
        
        y_coords, x_coords = valid_coords
        depths = mono_depth_view[y_coords, x_coords]  # [n_valid]
        
        # Get image dimensions for adaptive distance cap
        H, W = mono_depth_view.shape
        max_reasonable_dist = min(H, W) * 0.5  # Allow up to 50% of image size (was 30%)
        
        # Sample more candidates
        n_candidates = min(n_pairs * 10, 100000)  # More candidates for diverse sampling
        
        # Generate random pairs
        idx1 = torch.randint(0, n_valid, (n_candidates,), device=device)
        idx2 = torch.randint(0, n_valid, (n_candidates,), device=device)
        
        # Filter self-pairs
        valid_pair_mask = idx1 != idx2
        idx1 = idx1[valid_pair_mask]
        idx2 = idx2[valid_pair_mask]
        
        if len(idx1) == 0:
            return torch.empty((0, 2), dtype=torch.long, device=device)
        
        # Compute metrics
        dy = torch.abs(y_coords[idx1] - y_coords[idx2]).float()
        dx = torch.abs(x_coords[idx1] - x_coords[idx2]).float()
        spatial_dist = torch.sqrt(dy ** 2 + dx ** 2)
        depth_diff = torch.abs(depths[idx1] - depths[idx2])
        
        # Global distance constraint (avoid extreme edge-to-edge spans)
        global_dist_mask = spatial_dist < max_reasonable_dist
        
        # Stratified sampling by difficulty
        n_hard = n_pairs * 4 // 10    # 40% hard
        n_medium = n_pairs * 3 // 10  # 30% medium
        n_easy = n_pairs - n_hard - n_medium  # 30% easy
        
        selected_indices = []
        
        # 1. Hard pairs: Small depth diff (< 0.1), SHORT distance
        # Keep these pairs local for precision
        hard_mask = (
            (depth_diff < 0.1) & 
            (depth_diff > 0.01) & 
            (spatial_dist >= self.min_distance) &
            (spatial_dist < self.min_distance * 8) &
            global_dist_mask
        )  # 1% - 10%
        if hard_mask.sum() > 0:
            hard_indices = torch.where(hard_mask)[0]
            if len(hard_indices) > n_hard:
                # Random sample from hard pairs
                hard_selected = hard_indices[torch.randperm(len(hard_indices), device=device)[:n_hard]]
            else:
                hard_selected = hard_indices
            selected_indices.append(hard_selected)
        
        # 2. Medium pairs: Medium depth diff (0.1-0.3), MEDIUM distance
        medium_mask = (
            (depth_diff >= 0.1) & 
            (depth_diff < 0.3) & 
            (spatial_dist >= self.min_distance * 2) &
            (spatial_dist < self.min_distance * 15) &
            global_dist_mask
        ) # 10% - 30%
        # medium_mask = (
        #     (depth_diff >= 0.15) & 
        #     (depth_diff < 0.35) & 
        #     (spatial_dist >= self.min_distance * 2) &
        #     (spatial_dist < self.min_distance * 15) &
        #     global_dist_mask
        # ) # 15% - 35%
        if medium_mask.sum() > 0:
            medium_indices = torch.where(medium_mask)[0]
            if len(medium_indices) > n_medium:
                # Weight by depth difference for diversity
                weights_medium = depth_diff[medium_indices]
                if weights_medium.sum() > 0:
                    weights_medium = weights_medium / weights_medium.sum()
                    medium_selected = medium_indices[torch.multinomial(weights_medium, n_medium, replacement=False)]
                else:
                    medium_selected = medium_indices[torch.randperm(len(medium_indices), device=device)[:n_medium]]
            else:
                medium_selected = medium_indices
            selected_indices.append(medium_selected)
        
        # 3. Easy pairs: Large depth diff (>= 0.3), LONGER distance (but still capped)
        # Allow longer distances here but avoid extreme spans
        easy_mask = (
            (depth_diff >= 0.3) &
            (spatial_dist >= self.min_distance * 3) &
            (spatial_dist < max_reasonable_dist * 0.9) &
            global_dist_mask
        ) # 30% - 100%
        # easy_mask = (
        #     (depth_diff >= 0.35) & 
        #     (depth_diff < 0.7) &
        #     (spatial_dist >= self.min_distance * 3) &
        #     (spatial_dist < max_reasonable_dist * 0.9) &
        #     global_dist_mask
        # ) # 35% - 70%
        if easy_mask.sum() > 0:
            easy_indices = torch.where(easy_mask)[0]
            if len(easy_indices) > n_easy:
                # Weight by depth difference for diversity
                # This prefers large depth gaps, not necessarily far-apart pixels
                weights_easy = depth_diff[easy_indices]
                if weights_easy.sum() > 0:
                    weights_easy = weights_easy / weights_easy.sum()
                    easy_selected = easy_indices[torch.multinomial(weights_easy, n_easy, replacement=False)]
                else:
                    easy_selected = easy_indices[torch.randperm(len(easy_indices), device=device)[:n_easy]]
            else:
                easy_selected = easy_indices
            selected_indices.append(easy_selected)
        
        # Combine all selected pairs
        if len(selected_indices) == 0:
            # Fallback: simple random sampling with moderate distance constraints
            basic_mask = (
                (spatial_dist >= self.min_distance) & 
                (spatial_dist < max_reasonable_dist * 0.7) &
                (depth_diff > 0.01) &
                global_dist_mask
            )
            if basic_mask.sum() > 0:
                basic_indices = torch.where(basic_mask)[0]
                n_select = min(n_pairs, len(basic_indices))
                selected = basic_indices[torch.randperm(len(basic_indices), device=device)[:n_select]]
                pair_indices = torch.stack([idx1[selected], idx2[selected]], dim=1)
                return pair_indices
            else:
                return torch.empty((0, 2), dtype=torch.long, device=device)
        
        combined_indices = torch.cat(selected_indices)
        
        # Shuffle for training stability
        combined_indices = combined_indices[torch.randperm(len(combined_indices), device=device)]
        
        # Build pair indices
        pair_indices = torch.stack([idx1[combined_indices], idx2[combined_indices]], dim=1)
        
        return pair_indices

    def forward(
        self,
        step: int,
        rendered_depth: torch.Tensor,  # [B, H, W], Metric Depth
        mono_depth: torch.Tensor,      # [B, H, W], Normalized Depth
        valid_mask: Optional[torch.Tensor] = None, # [B, H, W]
        collect_vis: bool = False      # Whether to collect visualization data
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute adaptive single-view ranking loss.
        
        Returns:
            loss: Scalar loss
            loss_dict: Dictionary with loss breakdown
        """
        device = rendered_depth.device
        
        # Get adaptive weight
        weight = self._compute_weight(step)
        
        # If weight is 0, return zero loss immediately
        if weight < 1e-6:
            return {
                'loss': torch.tensor(0.0, device=device),
                'weight': 0.0
            }
        
        B, H, W = rendered_depth.shape
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid_views = 0

        vis_store = {} if collect_vis else None

        rendered_norm = rendered_depth
        
        # Loop over each lens in batch
        for b in range(B):
            if valid_mask is not None:
                mask_b = valid_mask[b] > 0.5
            else:
                mask_b = torch.ones(H, W, dtype=torch.bool, device=device)
            
            if mask_b.sum() < 20:
                continue
            
            valid_coords = torch.where(mask_b)
            n_valid = len(valid_coords[0])
            
            if n_valid < 2:
                continue
            
            # Sample pairs for this lens
            n_pairs = min(self.sample_pairs, n_valid * (n_valid - 1) // 2)
            pair_indices = self._sampling(
                valid_coords=valid_coords, 
                mono_depth_view=mono_depth[b], 
                n_pairs=n_pairs, 
                device=device
            )
            
            if len(pair_indices) == 0:
                continue
            
            # Get coordinates
            y_coords = valid_coords[0][pair_indices]  # [n_pairs, 2]
            x_coords = valid_coords[1][pair_indices]  # [n_pairs, 2]
            
            # Get depth values
            rendered_pairs = rendered_norm[b, y_coords, x_coords] # [n_pairs, 2]
            mono_pairs = mono_depth[b, y_coords, x_coords]        # [n_pairs, 2]
            
            # Compute differences
            rendered_diff = rendered_pairs[:, 0] - rendered_pairs[:, 1]
            mono_diff = mono_pairs[:, 0] - mono_pairs[:, 1]
            
            # Hinge loss: penalize ranking violations
            ranking_violation = torch.relu(
                self.margin - torch.sign(mono_diff) * rendered_diff
            )
            
            base_loss = ranking_violation.mean()
            
            # Apply adaptive weight
            total_loss = total_loss + base_loss
            valid_views += 1

            # Collect visualization data
            if collect_vis:
                agreement = (torch.sign(rendered_diff) == torch.sign(mono_diff))
                vis_store[b] = {
                    'y_coords': y_coords.detach().cpu(),     # [n_pairs, 2]
                    'x_coords': x_coords.detach().cpu(),     # [n_pairs, 2]
                    'agreement': agreement.detach().cpu(),   # [n_pairs]
                    'rendered_depth': rendered_depth[b].detach().cpu(),
                    'mono_depth': mono_depth[b].detach().cpu(),
                    'valid_mask': mask_b.detach().cpu()
                }

        # Average across valid cameras
        final_loss = total_loss / max(valid_views, 1)

        # Store visualization cache
        self._vis_cache = vis_store if collect_vis else None

        return {
            'loss': final_loss,
            'weight': weight
        }

    @torch.no_grad()
    def vis_sampled_pairs(
        self,
        save_path: str,
        step: int,
        max_vis_pairs: int = 50,
    ):
        """
        Visualize sampled pixel pairs using CACHED data.
        
        Green lines: Rankings agree
        Red lines: Rankings disagree
        """
        if self._vis_cache is None or len(self._vis_cache) == 0:
            print("[vis_sampled_pairs] No cached data. Call forward(..., collect_vis=True).")
            return
        
        from pathlib import Path
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Visualize each lens
        for lens, data in self._vis_cache.items():
            y_coords = data['y_coords'].numpy()         # [n_pairs, 2]
            x_coords = data['x_coords'].numpy()         # [n_pairs, 2]
            agreement = data['agreement'].numpy()       # [n_pairs]
            rendered_np = data['rendered_depth'].numpy()
            mono_np = data['mono_depth'].numpy()
            valid = data['valid_mask'].numpy()
            
            # Prepare visualization (set invalid to NaN)
            rendered_vis = np.full_like(rendered_np, np.nan)
            mono_vis = np.full_like(mono_np, np.nan)
            rendered_vis[valid] = rendered_np[valid]
            mono_vis[valid] = mono_np[valid]
            
            # Limit visualization pairs
            n_vis = min(max_vis_pairs, len(agreement))
            
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            # Plot rendered depth
            im1 = axes[0].imshow(rendered_vis, cmap='jet')
            axes[0].set_title(f'Rendered Depth (Lens {lens+1})')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
            
            # Plot mono depth
            im2 = axes[1].imshow(mono_vis, cmap='jet')
            axes[1].set_title(f'Monocular Depth (Lens {lens+1})')
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

            # Draw lines for pairs
            for i in range(n_vis):
                y1, y2 = y_coords[i]
                x1, x2 = x_coords[i]
                color = 'green' if agreement[i] else 'red'
                alpha = 0.6 if agreement[i] else 0.9
                linewidth = 0.5 if agreement[i] else 1.0
                
                for ax in axes:
                    ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=linewidth)
                    ax.scatter([x1, x2], [y1, y2], color=color, s=5, alpha=alpha)
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='green', lw=2, 
                      label=f'Agree ({agreement.sum()}/{len(agreement)})'),
                Line2D([0], [0], color='red', lw=2, 
                      label=f'Disagree ({(~agreement).sum()}/{len(agreement)})')
            ]
            axes[0].legend(handles=legend_elements, loc='upper right')

            plt.tight_layout()
            save_file = Path(save_path) / f"step_{step:06d}_lens{lens+1}.png"
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[Ranking Vis] Saved: {save_file}")
            print(f"  Agreement: {agreement.sum()}/{len(agreement)} "
                  f"({100*agreement.sum()/len(agreement):.1f}%)")
    
    @torch.no_grad()
    def vis_sampled_pairs_ori(
        self, 
        rendered_depth: torch.Tensor,  # [H, W]
        mono_depth: torch.Tensor,      # [H, W]
        valid_mask: torch.Tensor,      # [H, W]
        save_path: str,
        step: int,
        lens: int,
        max_vis_pairs: int = 50
    ):
        """
        Visualize sampled pixel pairs on depth maps.
        
        Green lines: Rankings agree (sign(rendered_diff) == sign(mono_diff))
        Red lines: Rankings disagree (violation)
        """
        device = rendered_depth.device
        
        # Sample pairs
        mask_b = valid_mask > 0.5
        if mask_b.sum() < 20:
            return
        
        valid_coords = torch.where(mask_b)
        n_valid = len(valid_coords[0])
        if n_valid < 2:
            return
        
        n_pairs = min(self.sample_pairs, n_valid * (n_valid - 1) // 2)
        pair_indices = self._sampling(valid_coords, mono_depth, n_pairs, device)
        
        if len(pair_indices) == 0:
            return
        
        # Get coordinates
        y_coords = valid_coords[0][pair_indices]  # [n_pairs, 2]
        x_coords = valid_coords[1][pair_indices]  # [n_pairs, 2]
        
        # Get depth values
        rendered_pairs = rendered_depth[y_coords, x_coords]  # [n_pairs, 2]
        mono_pairs = mono_depth[y_coords, x_coords]          # [n_pairs, 2]
        
        # Compute agreement
        rendered_diff = rendered_pairs[:, 0] - rendered_pairs[:, 1]
        mono_diff = mono_pairs[:, 0] - mono_pairs[:, 1]
        agreement = (torch.sign(rendered_diff) == torch.sign(mono_diff))
        
        # Move to CPU for visualization
        y_coords = y_coords.cpu().numpy()
        x_coords = x_coords.cpu().numpy()
        agreement = agreement.cpu().numpy()
        rendered_np = rendered_depth.cpu().numpy()
        mono_np = mono_depth.cpu().numpy()
        rendered_vis = np.full_like(rendered_np, np.nan)
        mono_vis = np.full_like(mono_np, np.nan)
        valid = mask_b.cpu().numpy()
        rendered_vis[valid] = rendered_np[valid]
        mono_vis[valid] = mono_np[valid]
        
        # Limit visualization pairs
        n_vis = min(max_vis_pairs, len(agreement))
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot rendered depth
        im1 = axes[0].imshow(rendered_vis, cmap='jet')
        axes[0].set_title('Rendered Depth with Sampled Pairs')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Plot mono depth
        im2 = axes[1].imshow(mono_vis, cmap='jet')
        axes[1].set_title('Monocular Depth with Sampled Pairs')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Draw lines for pairs
        for i in range(n_vis):
            y1, y2 = y_coords[i]
            x1, x2 = x_coords[i]
            color = 'green' if agreement[i] else 'red'
            alpha = 0.6 if agreement[i] else 0.9
            linewidth = 0.5 if agreement[i] else 1.0
            
            for ax in axes:
                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=linewidth)
                ax.scatter([x1, x2], [y1, y2], color=color, s=5, alpha=alpha)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label=f'Agree ({agreement.sum()}/{len(agreement)})'),
            Line2D([0], [0], color='red', lw=2, label=f'Disagree ({(~agreement).sum()}/{len(agreement)})')
        ]
        axes[0].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        Path(save_path).mkdir(parents=True, exist_ok=True)
        save_file = Path(save_path) / f"step_{step:06d}_lens{lens+1}.png"
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved intra-view pair visualization: {save_path}")
        print(f"  Agreement: {agreement.sum()}/{len(agreement)} ({100*agreement.sum()/len(agreement):.1f}%)")
    
    @torch.no_grad()
    def vis_sampled_pairs_more(
        self,
        rendered_depth: torch.Tensor,  # [H, W]
        mono_depth: torch.Tensor,      # [H, W]
        valid_mask: torch.Tensor,      # [H, W]
        save_path: str,
        step: int,
        max_vis_pairs: int = 50,
    ):
        """
        Visualize sampled depth ranking pairs.
        
        Shows:
        - Rendered depth map
        - Monocular depth map
        - Sampled pairs (color-coded by agreement)
        - Hard/medium/easy pair distribution
        
        Args:
            rendered_depth: [H, W] rendered depth
            mono_depth: [H, W] monocular depth
            valid_mask: [H, W] valid region mask
            save_path: Directory to save visualization
            step: Current training step
            max_vis_pairs: Maximum number of pairs to visualize
        """
        device = rendered_depth.device
        H, W = rendered_depth.shape
        
        # Get adaptive weight
        weight = self.get_adaptive_weight(step)
        
        if weight < 1e-6:
            return  # Skip if loss is inactive
        
        # Sample pairs (same as forward)
        mask_b = valid_mask > 0.5
        if mask_b.sum() < 20:
            return
        
        valid_coords = torch.where(mask_b)
        n_valid = len(valid_coords[0])
        if n_valid < 2:
            return
        
        n_pairs = min(self.sample_pairs, n_valid * (n_valid - 1) // 2)
        pair_indices = self._sampling(valid_coords, mono_depth, n_pairs, device)
        
        if len(pair_indices) == 0:
            return
        
        # Limit visualization
        n_vis = min(max_vis_pairs, len(pair_indices))
        pair_indices = pair_indices[:n_vis]
        
        # Get coordinates
        y_coords = valid_coords[0][pair_indices].cpu().numpy()  # [n_vis, 2]
        x_coords = valid_coords[1][pair_indices].cpu().numpy()

        # Get depth values
        rendered_pairs = rendered_depth[
            torch.from_numpy(y_coords[:, 0]), 
            torch.from_numpy(x_coords[:, 0])
        ].cpu().numpy()
        mono_pairs = mono_depth[
            torch.from_numpy(y_coords[:, 0]),
            torch.from_numpy(x_coords[:, 0])
        ].cpu().numpy()
        
        rendered_pairs_2 = rendered_depth[
            torch.from_numpy(y_coords[:, 1]),
            torch.from_numpy(x_coords[:, 1])
        ].cpu().numpy()
        mono_pairs_2 = mono_depth[
            torch.from_numpy(y_coords[:, 1]),
            torch.from_numpy(x_coords[:, 1])
        ].cpu().numpy()
        
        # Compute agreement
        rendered_diff = rendered_pairs - rendered_pairs_2
        mono_diff = mono_pairs - mono_pairs_2
        agreement = (np.sign(rendered_diff) == np.sign(mono_diff))
        
        # Compute pair distances and categorize
        dy = y_coords[:, 0] - y_coords[:, 1]
        dx = x_coords[:, 0] - x_coords[:, 1]
        pixel_dist = np.sqrt(dx**2 + dy**2)
        
        hard_dist_cap = self.min_distance * 8
        medium_dist_cap = self.min_distance * 15
        
        pair_types = np.where(
            pixel_dist < hard_dist_cap, 
            'hard',
            np.where(pixel_dist < medium_dist_cap, 'medium', 'easy')
        )
        # Create visualization
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Convert to numpy for plotting
        rendered_np = rendered_depth.cpu().numpy()
        mono_np = mono_depth.cpu().numpy()
        
        # ===== Plot 1: Rendered depth with pairs =====
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(rendered_np, cmap='turbo', interpolation='nearest')
        ax1.set_title(f'Rendered Depth\nStep {step}, Weight {weight:.3f}', fontsize=12)
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Draw pairs on rendered depth
        for i in range(n_vis):
            color = 'lime' if agreement[i] else 'red'
            alpha = 0.6 if agreement[i] else 0.8
            linewidth = 1.5 if pair_types[i] == 'hard' else (1.2 if pair_types[i] == 'medium' else 1.0)
            
            # Draw line connecting pair
            ax1.plot([x_coords[i, 0], x_coords[i, 1]], 
                    [y_coords[i, 0], y_coords[i, 1]],
                    color=color, alpha=alpha, linewidth=linewidth)
            
            # Draw points
            ax1.scatter(x_coords[i, 0], y_coords[i, 0], 
                       c=color, s=30, alpha=alpha, edgecolors='white', linewidths=0.5, zorder=5)
            ax1.scatter(x_coords[i, 1], y_coords[i, 1], 
                       c=color, s=30, alpha=alpha, edgecolors='white', linewidths=0.5, zorder=5)
        
        # ===== Plot 2: Monocular depth with pairs =====
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(mono_np, cmap='jet', interpolation='nearest')
        ax2.set_title('Monocular Depth (Ground Truth)', fontsize=12)
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Draw pairs on mono depth
        for i in range(n_vis):
            color = 'lime' if agreement[i] else 'red'
            alpha = 0.6 if agreement[i] else 0.8
            linewidth = 1.5 if pair_types[i] == 'hard' else (1.2 if pair_types[i] == 'medium' else 1.0)
            
            ax2.plot([x_coords[i, 0], x_coords[i, 1]], 
                    [y_coords[i, 0], y_coords[i, 1]],
                    color=color, alpha=alpha, linewidth=linewidth)
            ax2.scatter(x_coords[i, 0], y_coords[i, 0], 
                       c=color, s=30, alpha=alpha, edgecolors='white', linewidths=0.5, zorder=5)
            ax2.scatter(x_coords[i, 1], y_coords[i, 1], 
                       c=color, s=30, alpha=alpha, edgecolors='white', linewidths=0.5, zorder=5)
        
        # ===== Plot 3: Agreement statistics =====
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Pie chart for agreement
        agree_count = agreement.sum()
        disagree_count = len(agreement) - agree_count
        
        colors_pie = ['lime', 'red']
        explode = (0.05, 0.05)
        ax3.pie([agree_count, disagree_count], 
               labels=[f'Agree\n{agree_count}/{len(agreement)}', 
                      f'Disagree\n{disagree_count}/{len(agreement)}'],
               colors=colors_pie, autopct='%1.1f%%', startangle=90, explode=explode,
               textprops={'fontsize': 11, 'weight': 'bold'})
        ax3.set_title('Ranking Agreement', fontsize=12)
        
        # ===== Plot 4: Pair type distribution =====
        ax4 = fig.add_subplot(gs[1, 0])
        
        unique_types, type_counts = np.unique(pair_types, return_counts=True)
        colors_bar = {'hard': 'orange', 'medium': 'steelblue', 'easy': 'green'}
        bar_colors = [colors_bar.get(t, 'gray') for t in unique_types]
        
        ax4.bar(unique_types, type_counts, color=bar_colors, alpha=0.7, edgecolor='black')
        ax4.set_title('Pair Type Distribution', fontsize=12)
        ax4.set_ylabel('Count', fontsize=10)
        ax4.grid(axis='y', alpha=0.3)
        
        # Add counts on bars
        for i, (t, c) in enumerate(zip(unique_types, type_counts)):
            ax4.text(i, c + 0.5, str(c), ha='center', va='bottom', fontsize=10, weight='bold')
        
        # ===== Plot 5: Distance vs depth difference scatter =====
        ax5 = fig.add_subplot(gs[1, 1])
        
        depth_diff = np.abs(mono_diff)
        
        scatter_colors = ['lime' if a else 'red' for a in agreement]
        ax5.scatter(pixel_dist, depth_diff, c=scatter_colors, alpha=0.5, s=40, edgecolors='black', linewidths=0.5)
        ax5.set_xlabel('Pixel Distance', fontsize=10)
        ax5.set_ylabel('Mono Depth Difference', fontsize=10)
        ax5.set_title('Distance vs Depth Difference', fontsize=12)
        ax5.grid(True, alpha=0.3)
        
        # Add distance caps
        ax5.axvline(hard_dist_cap, color='orange', linestyle='--', alpha=0.5, label=f'Hard cap ({hard_dist_cap})')
        ax5.axvline(medium_dist_cap, color='steelblue', linestyle='--', alpha=0.5, label=f'Medium cap ({medium_dist_cap})')
        ax5.legend(fontsize=8, loc='upper right')
        
        # ===== Plot 6: Agreement by pair type =====
        ax6 = fig.add_subplot(gs[1, 2])
        
        type_order = ['hard', 'medium', 'easy']
        agreement_by_type = {t: [] for t in type_order}
        
        for i, pt in enumerate(pair_types):
            agreement_by_type[pt].append(agreement[i])
        
        type_agree_rates = []
        type_labels = []
        for t in type_order:
            if t in agreement_by_type and len(agreement_by_type[t]) > 0:
                rate = 100.0 * np.mean(agreement_by_type[t])
                type_agree_rates.append(rate)
                type_labels.append(f'{t.capitalize()}\n({len(agreement_by_type[t])} pairs)')
            else:
                type_agree_rates.append(0)
                type_labels.append(f'{t.capitalize()}\n(0 pairs)')
        
        bars = ax6.bar(range(len(type_order)), type_agree_rates, 
                      color=[colors_bar[t] for t in type_order], alpha=0.7, edgecolor='black')
        ax6.set_xticks(range(len(type_order)))
        ax6.set_xticklabels(type_labels, fontsize=9)
        ax6.set_ylabel('Agreement Rate (%)', fontsize=10)
        ax6.set_title('Agreement Rate by Pair Type', fontsize=12)
        ax6.set_ylim(0, 105)
        ax6.grid(axis='y', alpha=0.3)
        
        # Add percentage on bars
        for i, (bar, rate) in enumerate(zip(bars, type_agree_rates)):
            if rate > 0:
                ax6.text(i, rate + 2, f'{rate:.1f}%', ha='center', va='bottom', fontsize=9, weight='bold')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', 
                   markersize=8, label='Agree', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, label='Disagree', markeredgecolor='black'),
            Line2D([0], [0], color='orange', linewidth=2, label='Hard pairs'),
            Line2D([0], [0], color='steelblue', linewidth=2, label='Medium pairs'),
            Line2D([0], [0], color='green', linewidth=2, label='Easy pairs'),
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=5, 
                  bbox_to_anchor=(0.5, 0.98), fontsize=10, framealpha=0.9)
        
        # Save
        Path(save_path).mkdir(parents=True, exist_ok=True)
        save_file = Path(save_path) / f"intra_pairs_step_{step:06d}.png"
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Intra-View Viz] Saved to {save_file}")
        print(f"  Total pairs: {len(agreement)}")
        print(f"  Agreement: {100.0 * agree_count / len(agreement):.1f}%")
        print(f"  Hard: {np.sum(pair_types == 'hard')} ({100.0 * np.mean(agreement[pair_types == 'hard']) if np.any(pair_types == 'hard') else 0:.1f}% agree)")
        print(f"  Medium: {np.sum(pair_types == 'medium')} ({100.0 * np.mean(agreement[pair_types == 'medium']) if np.any(pair_types == 'medium') else 0:.1f}% agree)")
        print(f"  Easy: {np.sum(pair_types == 'easy')} ({100.0 * np.mean(agreement[pair_types == 'easy']) if np.any(pair_types == 'easy') else 0:.1f}% agree)")

class MetricDepthLoss(nn.Module):
    """
    Metric depth regularization loss for aligning rendered depth with ground truth metric depth.
    
    Features:
    - Multiple loss types: L1 (default), L2, Huber, or scale-invariant (log-space)
    - Automatic scheduling: warmup -> full weight -> decay
    - NaN-safe: only supervises pixels with valid metric depth GT
    - Mask-aware: respects fisheye masks and invalid regions
    
    Loss types:
        'l1':   Mean absolute error |rendered - metric|
                -> Precise, robust to outliers, good for smooth depth
        
        'l2':   Mean squared error (rendered - metric)^2
                -> Precise but sensitive to outliers
        
        'huber':Smooth L1 (L2 for small errors, L1 for large)
                -> Precise + robust, best for noisy GT
        
        'si_log': Scale-invariant log-space (Eigen et al. 2014)
                  log_diff = log(rendered) - log(metric)
                  loss = mean(log_diff²) - 0.5 * mean(log_diff)^2
                  -> For scale-ambiguous scenarios
    
    **Recommendation for metric-scale depths:**
    Use 'l1' (default) or 'huber' for precise and smooth alignment.
    Use 'si_log' only if scale ambiguity exists.
    
    Scheduling (example: 5k-30k steps):
        5k-7k:   Warmup (weight: 0.0 -> 1.0)
        7k-25k:  Full supervision (weight: 1.0)
        25k-30k: Decay (weight: 1.0 -> 0.0)
        30k+:    Off (weight: 0.0)
    
    Args:
        loss_type: Type of loss ('l1', 'l2', 'huber', 'si_log')
        huber_delta: Delta parameter for Huber loss (default: 0.5 meters)
        start_step: Step to start regularization (default: 5000)
        end_step: Step to end regularization (default: 30000)
        warmup_ratio: Fraction of active steps for warmup (default: 0.25)
        decay_ratio: Fraction of active steps for decay (default: 0.25)
    """
    
    def __init__(
        self,
        loss_type: str = 'l1',
        huber_delta: float = 0.5,
        start_step: int = 5000,
        end_step: int = 30000,
        warmup_ratio: float = 0.25,
        decay_ratio: float = 0.25,
    ):
        super().__init__()
        
        assert loss_type in ['l1', 'l2', 'huber', 'si_log'], \
            f"loss_type must be one of ['l1', 'l2', 'huber', 'si_log'], got {loss_type}"
        
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        
        self.start_step = start_step
        self.end_step = end_step
        
        # Compute warmup and decay boundaries
        total_steps = end_step - start_step
        self.warmup_steps = min(2000, int(total_steps * warmup_ratio))
        self.decay_start = end_step - min(5000, int(total_steps * decay_ratio))
        
    def _compute_weight(self, step: int) -> float:
        """
        Compute scheduling weight based on current step.
        
        Returns:
            weight in [0.0, 1.0]
        """
        # Before start: off
        if step < self.start_step:
            return 0.0
        
        # After end: off
        if step >= self.end_step:
            return 0.0
        
        # Warmup phase: linear ramp 0.0 -> 1.0
        warmup_end = self.start_step + self.warmup_steps
        if step < warmup_end:
            progress = (step - self.start_step) / self.warmup_steps
            return progress
        
        # Decay phase: linear ramp 1.0 -> 0.0
        if step > self.decay_start:
            progress = (self.end_step - step) / (self.end_step - self.decay_start)
            return progress
        
        # Full weight phase
        return 1.0
    
    def forward(
        self,
        rendered_depth: torch.Tensor,    # [B, H, W] rendered depth
        metric_depth: torch.Tensor,      # [B, H, W] GT metric depth (may contain NaN)
        valid_mask: Optional[torch.Tensor] = None,  # [B, H, W] additional valid mask
        step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute metric depth regularization loss.
        
        Args:
            rendered_depth: [B, H, W] rendered depth from Gaussian Splatting
            metric_depth: [B, H, W] ground truth metric depth (may contain NaN for invalid regions)
            valid_mask: [B, H, W] optional additional valid mask (e.g., fisheye mask)
            step: current training step for scheduling
            
        Returns:
            loss_dict: Dictionary containing:
                - 'loss': weighted loss (ready to add to total loss)
                - 'raw_loss': unweighted scale-invariant loss
                - 'weight': current schedule weight
                - 'valid_ratio': fraction of pixels with valid GT
        """
        # Compute schedule weight
        weight = self._compute_weight(step)
        
        # If weight is 0, return zero loss immediately
        if weight == 0.0:
            return {
                'loss': torch.tensor(0.0, device=rendered_depth.device),
                'raw_loss': torch.tensor(0.0, device=rendered_depth.device),
                'weight': 0.0,
                'valid_ratio': 0.0,
            }
        
        # Create valid mask: where we have valid metric depth GT
        valid_metric_mask = ~torch.isnan(metric_depth)  # [B, H, W]
        
        # Combine with additional mask if provided
        if valid_mask is not None:
            valid_metric_mask = valid_metric_mask & (valid_mask > 0.5)
        
        # Check if we have any valid pixels
        num_valid = valid_metric_mask.sum()
        if num_valid == 0:
            return {
                'loss': torch.tensor(0.0, device=rendered_depth.device),
                'raw_loss': torch.tensor(0.0, device=rendered_depth.device),
                'weight': weight,
                'valid_ratio': 0.0,
            }
        
        # Extract valid depths
        rendered_depth_valid = rendered_depth[valid_metric_mask]  # [N_valid]
        metric_depth_valid = metric_depth[valid_metric_mask]      # [N_valid]
        
        # Compute loss based on loss_type
        if self.loss_type == 'l1':
            # L1 loss: Mean absolute error
            # Best for: Precise alignment, robust to outliers, smooth depth
            raw_loss = torch.mean(torch.abs(rendered_depth_valid - metric_depth_valid))
            
        elif self.loss_type == 'l2':
            # L2 loss: Mean squared error
            # Best for: Precise alignment, but sensitive to outliers
            raw_loss = torch.mean((rendered_depth_valid - metric_depth_valid) ** 2)
            
        elif self.loss_type == 'huber':
            # Huber loss: Smooth L1 (L2 for small errors, L1 for large)
            # Best for: Precise + robust, ideal for noisy GT
            diff = rendered_depth_valid - metric_depth_valid
            abs_diff = torch.abs(diff)
            # Huber: 0.5 * x^2 if |x| <= delta, else delta * (|x| - 0.5 * delta)
            huber_mask = abs_diff <= self.huber_delta
            raw_loss = torch.where(
                huber_mask,
                0.5 * diff ** 2,
                self.huber_delta * (abs_diff - 0.5 * self.huber_delta)
            ).mean()
            
        elif self.loss_type == 'si_log':
            # Scale-invariant log-space loss (Eigen et al. 2014)
            # Best for: Scale-ambiguous scenarios (not recommended for metric-scale data)
            log_rendered = torch.log(rendered_depth_valid + 1e-6)
            log_metric = torch.log(metric_depth_valid + 1e-6)
            log_diff = log_rendered - log_metric
            # Scale-invariant: E[d^2] - λ * E[d]^2 where λ=0.5
            raw_loss = torch.mean(log_diff ** 2) - 0.5 * torch.mean(log_diff) ** 2
        
        # Apply weight
        weighted_loss = weight * raw_loss
        
        # Compute statistics
        total_pixels = valid_metric_mask.numel()
        valid_ratio = float(num_valid) / total_pixels
        
        return {
            'loss': weighted_loss,
            'raw_loss': raw_loss,
            'weight': weight,
            'valid_ratio': valid_ratio,
        }

class DepthSmoothLoss(nn.Module):
    """
    Edge-aware depth smoothness regularization for 3D Gaussian Splatting.
    
    **Problem**:
        GS rendered depth may lack smooth structure due to discrete Gaussian splats.
        Monocular/metric depth maps have inherently smooth structure.
    
    **Solution**:
        Encourage locally smooth depth while preserving edges.
        Uses image gradients to detect edges (don't smooth across object boundaries).
    
    Loss formulation:
        L_smooth = mean(|∂_x D| * exp(-λ*|∂_x I|)) + mean(|∂_y D| * exp(-λ*|∂_y I|))
    
    where:
        - ∂_x D, ∂_y D: Depth gradients (what we want to minimize)
        - ∂_x I, ∂_y I: Image gradients (edge detector)
        - exp(-λ*|∂_x I|): Edge-aware weight (low at edges, high in smooth regions)
        - λ: Edge sensitivity (default: 10.0)
    
    **Key features**:
    1. Edge-aware: Preserves depth discontinuities at object boundaries
    2. Anisotropic: Different smoothness in x/y based on image structure
    3. Weak constraint: Acts as regularizer, not hard supervision
    
    Scheduling (example: 15k-30k steps):
        0-15k:   OFF (let depth stabilize from other losses)
        15k-20k: Ramp up 0 -> max_weight
        20k-30k: Full weight (encourage smoothness)
    
    Args:
        start_step: Step to start regularization (default: 15000)
        end_step: Step to end regularization (default: 30000)
        max_weight: Maximum weight (default: 0.01, weak constraint)
        edge_lambda: Edge sensitivity parameter (default: 10.0)
        use_edge_aware: Enable edge-aware weighting (default: True)
    """
    
    def __init__(
        self,
        start_step: int = 15000,
        end_step: int = 30000,
        max_weight: float = 0.01,
        edge_lambda: float = 10.0,
        use_edge_aware: bool = True,
    ):
        super().__init__()
        
        self.start_step = start_step
        self.end_step = end_step
        self.max_weight = max_weight
        self.edge_lambda = edge_lambda
        self.use_edge_aware = use_edge_aware
    
    def _compute_weight(self, step: int) -> float:
        """Compute scheduling weight based on current step."""
        # Before start: off
        if step < self.start_step:
            return 0.0
        
        # After end: keep at max
        if step >= self.end_step:
            return self.max_weight
        
        # Linear ramp up: 0.0 -> max_weight
        progress = (step - self.start_step) / (self.end_step - self.start_step)
        return self.max_weight * progress
    
    def forward(
        self,
        rendered_depth: torch.Tensor,  # [B, H, W] rendered depth
        rgb: Optional[torch.Tensor] = None,  # [B, H, W, 3] for edge detection
        valid_mask: Optional[torch.Tensor] = None,  # [B, H, W]
        step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute edge-aware depth smoothness loss.
        
        Args:
            rendered_depth: [B, H, W] rendered depth from Gaussian Splatting
            rgb: [B, H, W, 3] RGB image for edge detection (optional)
            valid_mask: [B, H, W] valid region mask (optional)
            step: current training step for scheduling
            
        Returns:
            loss_dict: Dictionary containing:
                - 'loss': weighted smoothness loss
                - 'raw_loss': unweighted smoothness loss
                - 'weight': current schedule weight
        """
        # Compute schedule weight
        weight = self._compute_weight(step)
        
        # If weight is 0, return zero loss immediately
        if weight < 1e-6:
            return {
                'loss': torch.tensor(0.0, device=rendered_depth.device),
                'raw_loss': torch.tensor(0.0, device=rendered_depth.device),
                'weight': 0.0,
            }
        
        device = rendered_depth.device
        
        # Compute depth gradients
        # Horizontal: [B, H, W-1]
        grad_x = torch.abs(rendered_depth[:, :, 1:] - rendered_depth[:, :, :-1])
        # Vertical: [B, H-1, W]
        grad_y = torch.abs(rendered_depth[:, 1:, :] - rendered_depth[:, :-1, :])
        
        if self.use_edge_aware and rgb is not None:
            # Compute image gradients for edge detection
            # Convert RGB to grayscale: [B, H, W, 3] → [B, H, W]
            gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            
            # Image gradients
            img_grad_x = torch.abs(gray[:, :, 1:] - gray[:, :, :-1])  # [B, H, W-1]
            img_grad_y = torch.abs(gray[:, 1:, :] - gray[:, :-1, :])  # [B, H-1, W]
            
            # Edge-aware weights: exp(-λ * |∇I|)
            # High gradient (edge) -> low weight (preserve discontinuity)
            # Low gradient (smooth) -> high weight (encourage smoothness)
            weight_x = torch.exp(-self.edge_lambda * img_grad_x)  # [B, H, W-1]
            weight_y = torch.exp(-self.edge_lambda * img_grad_y)  # [B, H-1, W]
            
            # Weighted smoothness
            smooth_x = grad_x * weight_x
            smooth_y = grad_y * weight_y
        else:
            # Isotropic smoothness (no edge awareness)
            smooth_x = grad_x
            smooth_y = grad_y
        
        # Apply valid mask if provided
        if valid_mask is not None:
            # Mask for gradients (both pixels must be valid)
            mask_x = valid_mask[:, :, 1:] & valid_mask[:, :, :-1]  # [B, H, W-1]
            mask_y = valid_mask[:, 1:, :] & valid_mask[:, :-1, :]  # [B, H-1, W]
            
            # Masked mean
            if mask_x.sum() > 0:
                smooth_x_loss = smooth_x[mask_x].mean()
            else:
                smooth_x_loss = torch.tensor(0.0, device=device)
            
            if mask_y.sum() > 0:
                smooth_y_loss = smooth_y[mask_y].mean()
            else:
                smooth_y_loss = torch.tensor(0.0, device=device)
        else:
            # Global mean
            smooth_x_loss = smooth_x.mean()
            smooth_y_loss = smooth_y.mean()
        
        # Total smoothness loss
        raw_loss = smooth_x_loss + smooth_y_loss
        weighted_loss = weight * raw_loss
        
        return {
            'loss': weighted_loss,
            'raw_loss': raw_loss,
            'weight': weight,
        }
