import random
from typing import Optional, Tuple

import torch


def initialize_centers(
    X: torch.Tensor, n_clusters: int, seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize cluster centers (internal method)

    Args:
        X (torch.Tensor): input data
        n_clusters (int): number of clusters
        seed (int): random seed

    Returns:
        torch.Tensor: initial cluster centers
        torch.Tensor: indices of initial cluster centers
    """
    num_samples = len(X)
    if seed:
        torch.manual_seed(seed)

    indices = torch.randperm(num_samples)[:n_clusters]
    cluster_centers = X[indices]
    return cluster_centers, indices


def _revive_empty_clusters(
    X: torch.Tensor,
    centers: torch.Tensor,
    cls_assignments: torch.Tensor,
    x_assignments: torch.Tensor,
    n_clusters: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Revive empty clusters (internal method). This is used to handle non-assigned
    centers (scatter_mean may not always fill every center).

    We must be a little careful here, because a typical "random" strategy will
    be to initialise the new cluster centers with random data points. However,
    this can lead to an edge-case where the new cluster centers are initialised
    with points that are identical to existing cluster centers.

    Args:
        X (torch.Tensor): input data
        centers (torch.Tensor): cluster centers
        cls_assignments (torch.Tensor): closest cluster center of each data point
        x_assignments (torch.Tensor): closest index of each cluster center
        n_clusters (int): number of clusters (optional)
        seed (int): random seed (optional)

    Returns:
        torch.Tensor: revived cluster centers
    """
    n_x = X.shape[0]
    if n_clusters is not None:
        if n_clusters > centers.shape[0]:
            init_extra_centers = torch.zeros(
                n_clusters - centers.shape[0],
                *centers.shape[1:],
                device=centers.device,
                dtype=centers.dtype,
            )
            centers = torch.concat((centers, init_extra_centers), dim=0)
    else:
        n_clusters = centers.shape[0]

    cls_ixs = torch.unique(cls_assignments)
    dead_cluster_ixs = list(set(range(n_clusters)) - set(cls_ixs.cpu().tolist()))

    if dead_cluster_ixs:
        k_new = len(dead_cluster_ixs)
        x_ixs_used = x_assignments.cpu().tolist()
        if seed:
            random.seed(seed)
        cls_ix_new_candidates = set(random.sample(range(n_x), n_clusters))
        cls_ix_new = list(cls_ix_new_candidates - set(x_ixs_used))[:k_new]
        new_centers_init = X[cls_ix_new]
        centers[dead_cluster_ixs] = new_centers_init

    return centers
