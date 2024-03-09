from typing import Optional, Tuple, Callable
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


def sanitize_centers(
    X: torch.Tensor,
    centers: torch.Tensor,
    distance_function: Callable,
    n_clusters: int,
    snap_to_data: bool = False,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Sanitize cluster centers (internal method). This checks a few attributes, and is
    used to avoid empty clusters.

    Args:
        X (torch.Tensor): input data
        centers (torch.Tensor): cluster centers
        distance_function (Callable): distance function
        n_clusters (int): number of clusters,
        snap_to_data (bool): whether to move each cluster centers to the nearest data point.
            This is useful to avoid empty clusters.
        seed (int): random seed

    Returns:
        torch.Tensor: sanitized cluster centers
    """
    num_extra = n_clusters - centers.shape[0]
    if num_extra < 0:
        raise ValueError("Number of `cluster_centers` is greater than `n_clusters`")

    distance_matrix = distance_function(X, centers)
    indices = torch.argmin(distance_matrix, dim=0)

    if num_extra > 0:
        _, indices_addl = initialize_centers(X, len(indices) + num_extra, seed=seed)
        indices_addl = list(set(indices_addl.tolist()) - set(indices.tolist()))[:num_extra]
        indices = torch.cat([indices, torch.LongTensor(indices_addl)])

    centers = X[indices]
    return centers


def _revive_dead_clusters(X: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """
    Revive dead clusters (internal method). This is used to handle non-assigned
    centers (scatter_mean may not always fill every center), default to 0.

    Args:
        X (torch.Tensor): input data
        centers (torch.Tensor): cluster centers

    Returns:
        torch.Tensor: revived cluster centers
    """
    new_centers_init = X[torch.randperm(len(X))[: len(centers)]]
    mask_unasgnd = centers == 0
    centers[mask_unasgnd] = new_centers_init[mask_unasgnd]
    return centers
