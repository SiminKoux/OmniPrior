import math
import torch
from typing import Dict, Tuple

from utils import knn, rgb_to_sh
from dataloader.dataset import Parser
from gsplat.optimizers import SelectiveAdam

def initialize_dynamicness(N, init_static_prob=0.75, noise=0.5):
    """
    Initialize dynamicness for each point in the GS.
    
    Args:
        N: Number of points
        init_static_prob: Initial static probability (0.75 means 75% of points tend to be static)
        noise: with noise, increases diversity
    """
    # Compute the corresponding logit value: 
    # logit(0.25) ≈ -1.10 (dynamic probability 25%)
    static_logits = torch.logit(torch.tensor(1.0 - init_static_prob))

    # Add noise to introduce diversity
    dynamic_logits = torch.randn(N) * noise + static_logits
    
    return dynamic_logits

def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "metric",
    init_num_pts: int = 10_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr=1.6e-4,
    scales_lr=5e-3,
    opacities_lr=5e-2,
    quats_lr=1e-3,
    sh0_lr=2.5e-3,
    shN_lr=2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "metric" or init_type == "original":
        print("Initialization based on pre-defined point cloud.")
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        print("Randomly Initialization.")
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm, random, or pcd")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]
    dynamicness = initialize_dynamicness(N, init_static_prob=0.75, noise=0.5)  # [N,]
    # dynamicness = initialize_dynamicness(N, init_static_prob=0.95, noise=0.2)  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
        ("dynamicness", torch.nn.Parameter(dynamicness.float()), 1e-3)
    ]

    # color is SH coefficients.
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
    colors[:, 0, :] = rgb_to_sh(rgbs)
    params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
    params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    
    return splats, optimizers