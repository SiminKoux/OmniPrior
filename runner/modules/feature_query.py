import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import itertools
from typing import Optional, Collection, Iterable

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True):
    """
    Args:
        grid: [B, feature_dim, reso*], default is [1, 16, reso*]
        coords: [n, grid_dim], default is [n, 2]
    """
    grid_dim = coords.shape[-1]
    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add one
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)  # add batch dimension, [1, n, grid_dim]
    
    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f'Grid-sample was called with {grid_dim}D data but is only implemented for 2D and 3D data.')
    
    # Adjust coords shape to match grid
    # coords.shape[0] = [B], [1]*1 = [1], list(coords.shape[1:]) = [n, 2]
    # the final shape is [B, 1, n, 2], where the default B = 1
    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(grid,   # [1, feature_dim, reso*]
                          coords, # [1, 1, n, grid_dim]
                          align_corners=align_corners,
                          mode='bilinear',
                          padding_mode='border')  # [B, feature_dim, 1, n]
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [n, feature_dim]
    return interp  

def interpolate_ms_features(pts: torch.Tensor, ms_grids: Collection[Iterable[nn.Module]], 
                            grid_dim: int, concat_features: bool, num_levels: Optional[int]):
    # when pts is [N, 3], grid_dim is 2, coo_combs = [(0, 1), (0, 2), (1, 2)]
    # when pts is [N, 4], grid_dim is 2, coo_combs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    coo_combs = list(itertools.combinations(range(pts.shape[-1]), grid_dim))
    
    if num_levels is None:
        num_levels = len(ms_grids)  # length of multires list, e.g. [1, 2, 4, 8]
    
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for multires_id, grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for coo_index, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[coo_index].shape[1]  # shape of grid[coo_index]: 1, out_dim, *reso
            interp_out_plane = (grid_sample_wrapper(grid[coo_index], pts[..., coo_comb]).view(-1, feature_dim))
            # compute product over planes
            interp_space = interp_space * interp_out_plane   # [N, feature_dim]
            
        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp += interp_space
    
    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    
    return multi_scale_interp
            
def normalize_aabb(pts: torch.Tensor, aabb: torch.Tensor):
    # Normalize points to AABB
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0

def init_grid_param(config, a: float = 0.1, b: float = 0.5):
    grid_nd = config['grid_dim']   # default: 2, creates 2D planes/grids
    in_dim = config['input_dim']   # default: 4, [0, 1, 2, 3] = [x, y, z, t]
    out_dim = config['output_dim'] # default: 16, feature dimension of the grid
    reso = config['resolution']
    
    assert in_dim == len(reso), 'Resolution must have same number of elements as input-dimension'
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim, 'Grid-dimensions must be less than or equal to input-dimensions'
    # Create all feature planes: [xy, xz, xt, yz, yt, zt]
    # [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    
    grid_coefs = nn.ParameterList()
    for coo_index, coo_comb in enumerate(coo_combs):
        # Initialize grid coefficients, 2D planes shape: [1, out_dim, reso[1], reso[0]]
        # coo_comb[::-1]: Reverses resolution order for proper grid alignment
        # For example, if coo_comb = (0, 3), reso = [64, 64, 64, 25], grid_coef shape = [1, out_dim, 25, 64]
        grid_coef = nn.Parameter(torch.empty([1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]))
        ''' Initialize grid coefficients
            *** Considerations ***:
            1. For spatiotemporal planes, more stable learning is required;
                since temporal changes should be gradual, 
                also act as "no change/deformation" baseline for temporal effects.
            2. For spatial planes, more expressive learning is required.
                Providing diverse initial features by uiform random initialization, 
                helps in capturing spatial details early. '''
        if has_time_planes and 3 in coo_comb: # 3 is the index of the time dimension
            # For spatiotempral planes (xt, yt, zt) - stable temporal learning
            nn.init.ones_(grid_coef)  # all elements are initialized to 1.0
        else:
            # For spatial planes (xy, xz, yz) - maintain spatial expressiveness
            nn.init.uniform_(grid_coef, a=a, b=b) # uniform random initialization in [a, b]
        grid_coefs.append(grid_coef)
    return grid_coefs

class HexPlaneField(nn.Module):
    def __init__(self, bounds, planeconfig, multires):
        super().__init__()
        aabb = torch.tensor([[bounds, bounds, bounds], [-bounds, -bounds, -bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config = [planeconfig]
        self.multires = multires
        self.concat_features = True
        # Use for separating spatial and spatiotemporal
        self.s_plane_indices = [0, 1, 3]  # xy, xz, yz
        self.st_plane_indices = [2, 4, 5] # xt, yt, zt

        ### Init the planes ###
        self.grids = nn.ModuleList()
        self.feature_dim = 0
        for res in self.multires:
            # initialize the coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: mutlo-res only on the spatial planes
            config['resolution'] = [r * res for r in config['resolution'][:3]] + config['resolution'][3:]
            gp = init_grid_param(config)
            # shape[1] is out-dimension - Concatenate over feature len for each scale
            if self.concat_features:
                self.feature_dim += gp[-1].shape[1]
            else:
                self.feature_dim = gp[-1].shape[1]
            self.grids.append(gp)
        print(f'Initialized HexPlaneField with {self.feature_dim} features')
        print(f'Initialized HexPlaneField with model grids: {self.grids}')

    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    
    def set_aabb(self, xyz_max, xyz_min):
        """Set the axis-aligned bounding box for the feature planes.
    
        Args:
            xyz_max: Maximum bounds per dimension [x_max, y_max, z_max]
            xyz_min: Minimum bounds per dimension [x_min, y_min, z_min]
        """
        if not isinstance(xyz_max, np.ndarray) or not isinstance(xyz_min, np.ndarray):
            raise TypeError("xyz_max and xyz_min must be numpy arrays")
        scene_aabb = np.stack([xyz_max, xyz_min])
        aabb_tensor = torch.from_numpy(scene_aabb).to(dtype=torch.float32)
        if hasattr(self, 'aabb') and self.aabb is not None:
            # Maintain device placement if AABB already exists
            aabb_tensor = aabb_tensor.to(self.aabb.device)
        self.aabb = nn.Parameter(aabb_tensor, requires_grad=False)
        print(f'Voxel Plane: aabb set to {self.aabb}')
    
    def freeze_planes(self):
        for grid in self.grids:
            for plane in grid:
                plane.requires_grad = False
    
    def unfreeze_planes(self):
        for grid in self.grids:
            for plane in grid:
                plane.requires_grad = True
    
    def freeze_spatial_planes(self):
        for grid in self.grids:
            for i in self.s_plane_indices:
                grid[i].requires_grad = False
    
    def freeze_spatiotemporal_planes(self):
        for grid in self.grids:
            for i in self.st_plane_indices:
                grid[i].requires_grad = False

    def unfreeze_spatial_planes(self):
        for grid in self.grids:
            for i in self.s_plane_indices:
                grid[i].requires_grad = True
    
    def unfreeze_spatiotemporal_planes(self):
        for grid in self.grids:
            for i in self.st_plane_indices:
                grid[i].requires_grad = True
    
    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """
        Get density features for the given points
        Args:
            pts: [N, 3], global coordinates to query
            timestamps: [N, 1], timestamps to query
        Returns:
            features: [N, feature_dim], density features
        """
        pts = normalize_aabb(pts, self.aabb)
        if timestamps is not None:
            pts = torch.cat((pts, timestamps), dim=-1)
        pts = pts.reshape(-1, pts.shape[-1]) # [N, 3] or [N, 4]
        features = interpolate_ms_features(pts, 
                                           ms_grids=self.grids, 
                                           grid_dim=self.grid_config[0]["grid_dim"], 
                                           concat_features=self.concat_features, 
                                           num_levels=None)
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)
        return features

    def forward(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        features = self.get_density(pts, timestamps)
        return features