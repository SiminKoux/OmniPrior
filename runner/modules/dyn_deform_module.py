import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from utils import batch_quat_mul
from .feature_query import HexPlaneField

@dataclass
class DeformedParams:
    means: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor

@dataclass
class DeformParams:
    means_offset: torch.Tensor     # [N, 3]
    scales_offset: torch.Tensor    # [N, 3]
    rotations_offset: torch.Tensor # [N, 4]
    opacities_offset: torch.Tensor # [N, 1]
    colors_offset: torch.Tensor    # [N, 16, 3]
    dynamic_offset: torch.Tensor   # [N, 1]

def poc_fre(x, n):
    if n.numel() == 0:  # n is empty
        return x  # return input unchanged if no positional encoding is needed
    x_embed = (x.unsqueeze(-1) * n).flatten(-2)
    x_sin = x_embed.sin()
    x_cos = x_embed.cos()
    x_embed = torch.cat([x, x_sin, x_cos], dim=-1)
    return x_embed  


class ST_Deform(nn.Module):
    def __init__(self, n_depth=8, n_width=256, n_in=27, args=None):
        super(ST_Deform, self).__init__()
        # Set the number of layers and width of the MLPs
        self.n_depth = n_depth
        self.n_width = n_width
        self.n_in = n_in

        # Initialize the Hex- feature plane
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        self.ratio = 0

        self.enable_ddyn = getattr(args, 'enable_ddyn', False)

        # Initialize feature decoder mlps
        self.create_net()
        
        # Organize the networks
        self.nets = {
            'feature': self.feature_out,
            'pos': self.pos_deform,
            'sca': self.sca_deform,
            'rot': self.rot_deform,
            'opa': self.opa_deform,
            'app': self.app_deform
            }
    
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    
    def set_aabb(self, xyz_max, xyz_min):
        self.grid.set_aabb(xyz_max, xyz_min)
    
    def _create_feature_encoder(self):
        if self.n_depth < 1:
            raise ValueError("feature encoder (MLP) must be at least including one layer (feature_dim to n_width).")
        layers = [nn.Linear(self.grid.feature_dim, self.n_width)]
        for _ in range(self.n_depth - 1):
            layers.extend([nn.ReLU(), nn.Linear(self.n_width, self.n_width)])
        return nn.Sequential(*layers)
    
    def _create_mlp(self, out_dim):
        layers = [nn.ReLU(), nn.Linear(self.n_width, self.n_width)]
        layers.extend([nn.ReLU(), nn.Linear(self.n_width, out_dim)])
        return nn.Sequential(*layers)
    
    def create_net(self):
        # Initalize Motion Decoder
        self.feature_out = self._create_feature_encoder()
        self.pos_deform = self._create_mlp(3)
        self.sca_deform = self._create_mlp(3)
        self.opa_deform = self._create_mlp(1)
        self.rot_deform = self._create_mlp(4)
        self.app_deform = self._create_mlp(16*3)

        if self.enable_ddyn:
            print("Learning Dynamicness Offsets!")
            self.dyn_deform = self._create_mlp(1)

    def freeze_mlps(self):
        for param_name in ['feature', 'pos', 'sca', 'rot', 'opa', 'app']:
            for param in self.nets[param_name].parameters():
                param.requires_grad = False
        if self.enable_ddyn:
            for param in self.dyn_deform.parameters():
                param.requires_grad = False

    def unfreeze_mlps(self):
        for param_name in ['feature', 'pos', 'sca', 'rot', 'opa', 'app']:
            for param in self.nets[param_name].parameters():
                param.requires_grad = True  
        if self.enable_ddyn:
            for param in self.dyn_deform.parameters():
                param.requires_grad = True

    def get_mlp_params(self):
        params_list = []
        for name, param in self.named_parameters():
            if "grid" not in name:
                params_list.append(param)
        return params_list
    
    def get_grid_params(self):
        params_list = []
        for name, param in self.named_parameters():
            if "grid" in name:
                params_list.append(param)
        return params_list
    
    def query_feature(self, means_embed: torch.Tensor, times: torch.Tensor):
        """
        Queries the scene or motion grid features 
            and processes it through the tiny MLPs (merge all features).

        Args:
            means_embed: A tensor containing the embedded Gaussian positions with shape (N, D), 
                           where N is the number of points and D is the embedding dimension.
            times: A tensor containing the times with shape (N, B).
        
        Returns:
            torch.Tensor: The processed hidden features after passing through the encoder (feature planes + MLP).
        """
        grid_feature = self.grid(means_embed[:, :3], times[:, :1])
        hexplane_hidden = self.feature_out(grid_feature)
        return hexplane_hidden
    
    def _apply_deform(self, 
                      orig_param: torch.Tensor, 
                      offset: torch.Tensor, 
                      gate: torch.Tensor, 
                      is_rotation: bool = False,
                      is_sh: bool = False):
        """
        Modulates a potential offset by the dynamicness score and applies it.
        
        Args:
            orig_param: The original Gaussian parameter (e.g., means, scales).
            offset: The raw offset predicted by the motion MLP.
            gate: The gating signal to modulate the offset.
            is_rotation: Flag to handle quaternion multiplication.
            is_sh: Flag to handle spherical harmonics broadcasting.

        Returns:
            The deformed parameter and the final applied offset.
        """
        result = orig_param.clone()

        if is_rotation:
            # Normalize the predicted quaternion
            q = F.normalize(offset, dim=-1, eps=1e-12) # [N, 4]
            qI = torch.zeros_like(q) # identity quaternion
            qI[..., 0] = 1.0  # [1, 0, 0, 0]
            # Soft interpolation
            mix = torch.lerp(qI, q, gate)  # [N, 4]
            final_offset = F.normalize(mix, dim=-1, eps=1e-12)  # [N, 4]
            out = batch_quat_mul(result, final_offset)  # [N, 4]   
        elif is_sh:
            # For SH, gate shape is [N, 1], offset is [N, 16, 3]
            # We need to unsqueeze gate to [N, 1, 1] for broadcasting
            final_offset = gate.view(-1, 1, 1) * offset  # [N, 16, 3]
            out = result + final_offset  # [N, 16, 3]
        else:
            # For other attributes (means, scales, opacities): simple linear modulation
            final_offset = gate * offset  # [N, 3] or [N, 1]
            out = result + final_offset  # [N, 3] or [N, 1]
        
        return out, final_offset
    
    @property
    def get_empty_ratio(self):
        return self.ratio

    def forward(self, 
                means_embed: torch.Tensor, 
                scales_embed: torch.Tensor, 
                rotations_embed: torch.Tensor, 
                opacity: torch.Tensor, 
                app: torch.Tensor, 
                times: torch.Tensor,
                cano_dyn: torch.Tensor = None):
        """
        Args:
            means_embed: Tensor, [N, 3].
            scales_embed: Tensor, [N, 3]. 
            rotations_embed: Tensor, [N, 4].
            opacity: Tensor, [N], opacity.unsqueeze(-1) -> [N, 1]. 
            app: Tensor, [N, k, 3], where k=16 in default.
            times: Tensor, [N, 6], all elements are the same.
            cano_dyne: Tensor, [N, 1], canoncial dynamicness for modulation.

        Returns:
            deform_params: DeformParams, containing offsets and dynamicness.
            deformed_params: DeformedParams, containing deformed Gaussian parameters.
        """
        
        # Get all Potential Offsets from Motion Stream
        motion_hidden = self.query_feature(means_embed=means_embed, times=times) # [N, ...]

        if opacity.dim() == 1:
            opacity = opacity.unsqueeze(-1) # [N, 1]
        
        # Potential offsets for each attribute
        dx = self.nets['pos'](motion_hidden)
        ds = self.nets['sca'](motion_hidden)
        do = self.nets['opa'](motion_hidden)
        dr = self.nets['rot'](motion_hidden)
        dcolors = self.nets['app'](motion_hidden)

        if self.enable_ddyn:
            ddyn = self.dyn_deform(motion_hidden)  # [N, 1]
            gate = torch.sigmoid(cano_dyn + ddyn)  # [N, 1]
        else:
            ddyn = None
            gate = torch.sigmoid(cano_dyn)

        # Apply modulated deformation for each attribute
        means, final_dx = self._apply_deform(
            orig_param=means_embed[:, :3], 
            offset=dx, 
            gate=gate
        )
        scales, final_ds = self._apply_deform(
            orig_param=scales_embed[:, :3], 
            offset=ds, 
            gate=gate
        )
        rotations, final_dr = self._apply_deform(
            orig_param=rotations_embed[:, :4], 
            offset=dr, 
            gate=gate, 
            is_rotation=True
        )
        opacities, final_do = self._apply_deform(
            orig_param=opacity, 
            offset=do, 
            gate=gate
        )
        dcolors = dcolors.reshape(app.shape)
        colors, final_dcolors = self._apply_deform(
            orig_param=app, 
            offset=dcolors, 
            gate=gate,
            is_sh=True
        )

        # Package and return results
        deform_params = DeformParams(
            means_offset=final_dx,
            scales_offset=final_ds,
            rotations_offset=final_dr,
            opacities_offset=final_do.squeeze(-1) if final_do is not None else None,
            colors_offset=final_dcolors,
            dynamic_offset=ddyn.squeeze(-1) if ddyn is not None else None
        )

        deformed_params = DeformedParams(
            means=means,
            scales=scales,
            rotations=rotations,
            opacities=opacities.squeeze(-1),
            colors=colors
        )

        return deform_params, deformed_params


def initilize_weights(m):
    if isinstance(m, nn.Linear):
        # default mode='fan_in', and another choice is 'fan_out'
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def diff1(x, dim: int):
    # helper: shape-preserving narrow difference
    return x.narrow(dim, 1, x.size(dim)-1) - x.narrow(dim, 0, x.size(dim)-1)

def charbonnier(x: torch.Tensor, eps: float):
    return torch.sqrt(x * x + eps)

def _reduce_penalty(pen, mode: str, eps: float, delta: float):
    if mode == 'l2':
        return (pen * pen).mean()
    elif mode == 'huber':
        return F.huber_loss(
            input=pen, 
            target=torch.zeros_like(pen), 
            reduction='mean', 
            delta=delta)
    else: # charbonnier
        return charbonnier(pen, eps).mean()

def compute_plane_tv(fea_plane: torch.Tensor, tv_mode: str, eps: float, delta: float):
    # fea_plane: [B, C, H, W]
    dx = diff1(fea_plane, -2)  # H
    dy = diff1(fea_plane, -1)  # W
    tv = _reduce_penalty(dx, tv_mode, eps, delta) + _reduce_penalty(dy, tv_mode, eps, delta)
    return tv

def compute_plane_smooth(fea_plane: torch.Tensor, tv_mode: str, eps: float, delta : float, time_dim: int):
    # second-order along time dimension (for tx/ty/tz planes)
    if fea_plane.size(time_dim) < 3:
        return fea_plane.new_tensor(0.0)
    dt1 = diff1(fea_plane, time_dim)
    dt2 = diff1(dt1, time_dim)
    sm = _reduce_penalty(dt2, tv_mode, eps, delta)
    return sm

def sh_spectral_reg(sh_coeffs: torch.Tensor, max_l: int = 3, lam_base: float = 0.01):
    """
    sh_coeffs: [..., n_coeff], order-packed as sum_{l=0..L} (2l+1)
    """
    if lam_base <= 0:
        return torch.tensor(0., device=sh_coeffs.device, dtype=sh_coeffs.dtype)
    reg, offset = 0.0, 0
    for l in range(max_l + 1):
        count = 2 * l + 1
        coeff_l = sh_coeffs[..., offset:offset + count]
        lam = lam_base * (l * l)
        if l == 0:
            lam *= 0.1  # relax DC
        reg = reg + lam * (coeff_l * coeff_l).mean()
        offset += count
    return reg


class DeformOptModule(nn.Module):
    def __init__(self, args):
        super(DeformOptModule, self).__init__()
        # Setups for the ST_Deform module
        net_width = args.net_width
        net_depth = args.net_depth
        posebase_pe = args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        
        # Initialize the ST_Deform module
        self.st_deform = ST_Deform(n_depth=net_depth, 
                                   n_width=net_width, 
                                   n_in=3+(3*posebase_pe)*2,
                                   args=args)
        
        self.register_buffer("pose_poc", torch.FloatTensor([(2**i) for i in range(posebase_pe)]))
        self.register_buffer("scale_rotation_poc", torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.apply(initilize_weights)

        self.s_tv_reg = args.s_tv_reg
        self.st_tv_reg = args.st_tv_reg
        self.st_l1_reg = args.st_l1_reg

        self.tv_mode = getattr(args, 'tv_mode', 'charbonnier')   # 'charbonnier' / 'huber' / 'l2'
        self.tv_eps = float(getattr(args, 'tv_eps', 1e-5))
        self.tv_huber_delta = float(getattr(args, 'tv_huber_delta', 0.01))
        self.time_dim = int(getattr(args, 'time_dim', -2))
        self.sh_reg_lambda = float(getattr(args, 'sh_reg_lambda', 0.01))

    @property
    def get_aabb(self):
        return self.st_deform.get_aabb
    
    @property
    def get_empty_ratio(self):
        return self.st_deform.get_empty_ratio
    
    def get_mlp_params(self):
        return self.st_deform.get_mlp_params()
    
    def get_grid_params(self):
        return self.st_deform.get_grid_params()
    
    def forward(self, point, scale=None, rotation=None, opacity=None, app=None, times_sel=None, cano_dyn=None):
        point_embed = poc_fre(point, self.pose_poc)
        scales_embed = poc_fre(scale, self.scale_rotation_poc)
        rotations_embed = poc_fre(rotation, self.scale_rotation_poc)
        offsets, deformed_params = self.st_deform(means_embed=point_embed, 
                                                  scales_embed=scales_embed, 
                                                  rotations_embed=rotations_embed, 
                                                  opacity=opacity, 
                                                  app=app, 
                                                  times=times_sel,
                                                  cano_dyn=cano_dyn)
        return offsets, deformed_params

    def st_deform_loss(self):
        '''
        Robust regularization for multi-scale HexPlane
        - Spatial plane (xy/xz/yz): First-order difference dx/dy (Charbonnier/Huber/L2), sum -> /count -> *w_spatial
        - Temporal plane (tx/ty/tz): Second-order difference dt2 along time_dim, sum -> /count -> *w_time2
        - Three weighted terms: s_tv_reg, st_tv_reg, st_l1_reg
        '''
        multires_grids = self.st_deform.grid.grids  # 6 * [1, rank * F_dim, reso, reso]
        
        # Initialize as tensors instead of integers
        device = multires_grids[0][0].device  # Get device from first grid
        s_tv_loss = torch.tensor(0.0, device=device)   # smoothness(tv) loss for spatial planes
        st_tv_loss = torch.tensor(0.0, device=device)  # smoothness(tv) loss for spatiotemporal planes
        st_l1_loss = torch.tensor(0.0, device=device)  # l1 loss for spatiotemporal planes
        
        for grids in multires_grids:
            if len(grids) == 6:
                for s_grid_id in [0, 1, 3]:  # xy, xz, yz
                    # First-order differences along both spatial dimensions appropriate for spatial regularizations
                    plane = grids[s_grid_id]
                    s_tv_loss += compute_plane_tv(
                        fea_plane=plane,
                        tv_mode=self.tv_mode,
                        eps = self.tv_eps,
                        delta=self.tv_huber_delta
                    )

                for st_grid_id in [2, 4, 5]: # tx, ty, tz
                    # Second-order differences along time dimension better for temporal smoothness
                    plane = grids[st_grid_id]
                    st_tv_loss += compute_plane_smooth(
                        fea_plane=plane,
                        tv_mode=self.tv_mode,
                        eps=self.tv_eps,
                        delta=self.tv_huber_delta,
                        time_dim=self.time_dim
                    )

                    # L1 anchored to 1: prevent excessive deformation in the temporal domain
                    st_l1_loss = st_l1_loss + torch.abs(1.0 - plane).mean()
        
        total = (s_tv_loss  * self.s_tv_reg) + \
            (st_tv_loss * self.st_tv_reg) + \
            (st_l1_loss * self.st_l1_reg)

        return total, s_tv_loss.detach(), st_tv_loss.detach(), st_l1_loss.detach()

    def shs_reg(self, color_offsets: torch.Tensor, max_l: int = 3):
        """
        Regularization for spherical harmonics coefficients.
        Args:
            color_offsets: Tensor of shape [..., n_coeff, 3], 
                           where n_coeff is the number of SH coefficients.
            max_l: Maximum spherical harmonics order.
        Returns:
            Regularization term for SH coefficients.
        """
        _, _, C = color_offsets.shape
        reg = torch.tensor(0.0, device=color_offsets.device, dtype=color_offsets.dtype)
        for c in range(C):
            # Each channel is regularized separately
            sh_c = color_offsets[..., :, c]  # [N,16]
            reg = reg + sh_spectral_reg(
                sh_coeffs = sh_c,
                max_l = max_l,
                lam_base = self.sh_reg_lambda
            )
        # Average over each channel
        return reg / float(C)