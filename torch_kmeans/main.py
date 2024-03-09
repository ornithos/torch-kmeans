import logging
from typing import Optional, Tuple

from tqdm import tqdm
import torch
from torch import nn
from torch_scatter import scatter_mean

from distances import get_distance_function
from initialization import initialize_centers, sanitize_centers


class KMeans(nn.Module):
    """
    PyTorch-based KMeans model

    Args:
        n_clusters (int): Number of clusters
        distance (str): Distance metric to use. Currently supports "euclidean" and "cosine"
        tol (float): Tolerance for convergence
        max_iter (int): Maximum number of iterations
        init (str): Initialization strategy. Currently supports "random", but intend to support
            "kmeans++" in the future
        use_tqdm (bool): Use tqdm progress bar

    Attributes:
        cluster_centers (torch.Tensor): Cluster centers
        assignments (torch.Tensor): Cluster assignments

    Methods:
        fit(X: torch.Tensor): Fit the model to the data
        predict(X: torch.Tensor): Predict cluster assignments for new data
    """

    def __init__(
        self,
        n_clusters: int,
        distance: str = "euclidean",
        tol: float = 1e-4,
        max_iter: int = 100,
        init: str = "random",
        use_tqdm: bool = True,
    ):
        """
        KMeans clustering module initialization. See class docstring for details.
        """
        super(KMeans, self).__init__()

        assert init == "random", "Only random initialization is supported at this time"
        """TODO: Implement kmeans++ initialization"""

        self._use_tqdm = use_tqdm
        self._logger = logging.getLogger("kmeans")
        self._tqdm_meter = tqdm(desc="[kmeans]") if use_tqdm else None
        self.distance_function = get_distance_function(distance)
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.init_strategy = init

        self._centers = None

    def _data_checks(self, X, cluster_centers: Optional[torch.Tensor], n_clusters: int) -> None:
        """
        Perform some basic checking on the torch data types

        Args:
            X (torch.Tensor): input data
            cluster_centers (torch.Tensor): cluster centers
            n_clusters (int): number of clusters
        """
        assert isinstance(X, torch.Tensor)

        if not X.dtype in (torch.float32, torch.float16):
            self._logger.warning(
                f"X is of dtype {X.dtype=} which is not recommended. You may have better performance using torch.float32."
            )

        if cluster_centers is not None:
            assert isinstance(cluster_centers, torch.Tensor)
            assert cluster_centers.device == X.device
            assert cluster_centers.dtype == X.dtype

        assert isinstance(n_clusters, int)
        assert n_clusters > 0
        assert n_clusters <= len(X)

    def fit(
        self, X: torch.Tensor, centers: Optional[torch.Tensor] = None, seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit cluster centers to the data according to Lloyd's algorithm (KMeans)

        Args:
            X (torch.Tensor): input data
            centers (torch.Tensor): initial cluster centers
            seed (int): random seed

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: cluster assignments, cluster centers
        """
        self._data_checks(X, centers, self.n_clusters)

        # initialize
        if centers is None:
            centers, _ = initialize_centers(X, self.n_clusters, seed=seed)
        elif self._centers is None:
            centers = sanitize_centers(
                X, centers, self.distance_function, self.n_clusters, seed=seed, snap_to_data=True
            )
        else:
            centers = self._centers

        for i in range(self.max_iter):
            assignments = self._find_closest_centers(X, centers)
            new_centers = scatter_mean(X.T, assignments).T
            """scatter_mean is the magic for calculating the new_centers. See `torch_scatter`"""

            # handling non-assigned centers (scatter_mean may not always fill every center), default to 0
            new_centers_init = X[torch.randperm(len(X))[: self.n_clusters]]
            mask_unasgnd = new_centers == 0  # should we worry about FP error here?
            new_centers[mask_unasgnd] = new_centers_init[mask_unasgnd]

            # what is the shift?
            center_shift = (new_centers - centers).norm()
            centers = new_centers
            iteration += 1

            # update tqdm meter
            if self._use_tqdm:
                self._tqdm_meter.set_postfix(  # type: ignore  - tqdm_meter is available iff use_tqdm is True
                    iteration=f"{iteration}",
                    center_shift=f"{center_shift ** 2:0.6f}",
                    tol=f"{self.tol:0.6f}",
                )
                # tqdm_meter.update()
            if center_shift**2 < self.tol:
                break

        self._centers = centers
        return assignments, centers

    def _find_closest_centers(self, X: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Find the closest centers for each data point in X"""
        dist_matrix = self.distance_function(X, centers)
        assignments = torch.argmin(dist_matrix, dim=1).squeeze()
        return assignments

    def predict(self, X: torch.Tensor, centers: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Find the closest centers for each data point in X

        Args:
            X (torch.Tensor): input data
            centers (torch.Tensor): cluster centers (optional - if not provided, uses the centers
                from the fit method)

        Returns:
            torch.Tensor: cluster assignments
        """
        centers = centers if centers is not None else self._centers
        assert centers is not None, "No cluster centers found. Please fit the model first."

        self._data_checks(X, centers, self.n_clusters)
        assignments = self._find_closest_centers(X, centers)

        return assignments
