"""Pairwise distance functions for KMeans clustering."""

from functools import partial
from typing import Callable, List, Tuple

import torch


def get_distance_function(distance_name: str):
    """
    Return pairwise distance function from name. Currently
    supports euclidean/cosine, all other choices will raise
    a NotImplementedError.
    """
    if distance_name.lower() == "euclidean":
        return partial(pairwise_lp, p=2)
    elif distance_name.lower() == "cosine":
        return pairwise_cosine
    else:
        raise NotImplementedError(f"{distance_name=}")


def pairwise_lp(A: torch.Tensor, B: torch.Tensor, p: int = 2):
    """Compute pairwise LP distances between two sets of points."""

    # add batch dimension (assumes not present)
    A = A.unsqueeze(dim=0)  # 1 * N * M
    B = B.unsqueeze(dim=0)  # 1 * K * M

    # dis = torch.sqrt(A**2 + B**2 - 2 * A @ B.T)
    dis = torch.cdist(A, B, p=p)
    return dis.squeeze(0)


def pairwise_cosine(A: torch.Tensor, B: torch.Tensor):
    """Compute the pairwise cosine distance between two sets of points."""

    # A: N*M,  B: K*M
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    # return N*K matrix for pairwise distance
    cosine_dis = 1 - A_normalized @ B_normalized.T

    return cosine_dis


def closest_indices(
    X: torch.Tensor, centers: torch.Tensor, distance_function: Callable, dims: Tuple
) -> List[torch.Tensor]:
    """Find the closest centers for each data point in X"""
    dist_matrix = distance_function(X, centers)
    out = []
    for dim in dims:
        assignments = torch.argmin(dist_matrix, dim=dim, keepdim=False)
        out.append(assignments)
    return out
