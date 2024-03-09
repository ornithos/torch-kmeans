""" Pairwise distance functions for KMeans clustering. """

import torch


def get_distance_function(distance_name: str):
    """
    Return pairwise distance function from name. Currently
    supports euclidean/cosine, all other choices will raise
    a NotImplementedError.
    """
    if distance_name.lower() == "euclidean":
        return pairwise_euclidean
    elif distance_name.lower() == "cosine":
        return pairwise_cosine
    else:
        raise NotImplementedError(f"{distance_name=}")


def pairwise_euclidean(A: torch.Tensor, B: torch.Tensor):
    # A: N*M,  B: K*M

    # N*1
    A = A.unsqueeze(dim=1).norm(dim=-1)

    # 1*K
    B = B.unsqueeze(dim=0).norm(dim=-1)

    dis = A + B - 2 * A @ B.T
    return dis


def pairwise_cosine(A: torch.Tensor, B: torch.Tensor):
    # A: N_1*M,  B: K*M
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    # return N*K matrix for pairwise distance
    cosine_dis = 1 - A_normalized @ B_normalized.T

    return cosine_dis
