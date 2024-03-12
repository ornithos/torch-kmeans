import logging
import random
from typing import Literal, Optional, Union

import torch
from torch import nn
from torch_scatter import scatter_mean
from tqdm import tqdm

from .distances import closest_indices, get_distance_function
from .initialization import _revive_empty_clusters, initialize_centers
from .utils import KmeansResult


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
        n_init: int = 10,
        init: str = "random",
        use_tqdm: bool = True,
    ):
        """
        KMeans clustering module initialization. See class docstring for details.
        """
        super(KMeans, self).__init__()

        assert max_iter > 0
        assert init == "random", "Only random initialization is supported at this time"
        """TODO: Implement kmeans++ initialization"""

        self._use_tqdm = use_tqdm
        self._tqdm_meter = None
        self._logger = logging.getLogger("kmeans")
        self.distance_function = get_distance_function(distance)
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.init_strategy = init
        self.n_init = n_init

        self._centers = None

    def _tqdm_service(
        self,
        operation: Union[Literal["start"], Literal["stop"]],
        c_init: Optional[int] = None,
    ) -> None:
        if self._use_tqdm:
            if operation == "start":
                c_init = c_init if c_init is not None else 0
                self._tqdm_meter = tqdm(desc=f"[kmeans ({c_init:d}/{self.n_init})]")
            elif operation == "stop":
                self._tqdm_meter.disable = True  # type: ignore[attr-defined]
                """mypy: tqdm_meter exists <=iff=> use_tqdm"""
                self._tqdm_meter = None

    def _data_checks(self, X, centers: Optional[torch.Tensor], seed: Optional[int]) -> None:
        """
        Perform some basic checking on the torch data types

        Args:
            X (torch.Tensor): input data
            centers (torch.Tensor): cluster centers
            seed (int): random seed
        """
        assert isinstance(X, torch.Tensor)
        assert X.ndimension() == 2

        if X.dtype not in (torch.float32, torch.float16):
            self._logger.warning(
                f"X is of dtype {X.dtype=} which is not recommended. You may have better performance using torch.float32."
            )

        assert isinstance(self.n_clusters, int)
        assert self.n_clusters > 0
        assert self.n_clusters <= len(X)

        if centers is not None:
            assert isinstance(centers, torch.Tensor)
            assert centers.device == X.device
            assert centers.dtype == X.dtype
            assert centers.ndimension() == 2

            if self.n_clusters < centers.shape[0]:
                raise ValueError("Number of `centers` is greater than `n_clusters`")

            if self.n_init > 1:
                self._logger.warning(
                    "Multiple restarts may be less useful when `init_centers` is provided. "
                    "Consider setting `n_init=1`."
                )

    def fit_transform(
        self,
        X: torch.Tensor,
        init_centers: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> KmeansResult:
        """
        Fit cluster centers to the data according to Lloyd's algorithm (KMeans)

        Args:
            X (torch.Tensor): input data
            centers (torch.Tensor): initial cluster centers
            seed (int): random seed

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: cluster assignments, cluster centers
        """
        self._data_checks(X, init_centers, seed)

        if seed is not None:
            random.seed(seed)

        seeds = [random.randint(0, 2**32) for _ in range(self.n_init)]

        # Main loop
        best_result = KmeansResult(torch.zeros(0), torch.zeros(0), 0, False)
        for restart_ix, c_seed in enumerate(seeds):
            # initialize
            if init_centers is None:
                centers = initialize_centers(X, self.n_clusters, seed=c_seed)[0]
            else:
                cls_assignments, x_indices = closest_indices(
                    X, init_centers, self.distance_function, dims=(1, 0)
                )
                centers = _revive_empty_clusters(
                    X,
                    init_centers,
                    cls_assignments,
                    x_indices,
                    n_clusters=self.n_clusters,
                    seed=c_seed,
                )

            torch.manual_seed(c_seed)
            km_result = self._kmeans_internal(X, centers, restart_ix)

            if km_result < best_result:
                best_result = km_result

        self._centers = best_result.centers
        return km_result

    def fit(
        self,
        X: torch.Tensor,
        centers: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Alias for `fit_transform` with no values returned."""
        self.fit_transform(X, centers, seed=seed)

    def _kmeans_internal(
        self, X: torch.Tensor, centers: torch.Tensor, restart_ix: int
    ) -> KmeansResult:
        """
        Internal kmeans method for fitting the model. This is the main algorithmic loop, which is
        called potentially multiple times from the `fit` method.

        Args:
            X (torch.Tensor): input data
            centers (torch.Tensor): initial cluster centers
            restart_ix (int): outer loop index for multiple restarts

        Returns:
            KmeansResult: result of the kmeans iterations
        """
        self._tqdm_service("start", restart_ix)
        has_converged = False
        for i in range(self.max_iter):
            with torch.no_grad():
                cls_assignments, x_indices = closest_indices(
                    X, centers, self.distance_function, dims=(1, 0)
                )
                new_centers = scatter_mean(X.T, cls_assignments).T
                """scatter_mean is the magic for calculating the new_centers. See `torch_scatter`"""
                new_centers = _revive_empty_clusters(X, new_centers, cls_assignments, x_indices)

                # what is the "loss"? - i.e. the residual centers movement
                centers_residual_norm = (new_centers - centers).norm().item()  # Frobenius
                centers = new_centers

            # update tqdm meter
            if self._use_tqdm:
                self._tqdm_meter.set_postfix(  # type: ignore[attr-defined]
                    iteration=f"{i}",
                    center_shift=f"{centers_residual_norm ** 2:0.6f}",
                    tol=f"{self.tol:0.6f}",
                )
                """mypy: tqdm_meter exists <=iff=> use_tqdm"""

            # Loop early exit
            if centers_residual_norm**2 < self.tol:
                has_converged = True
                break

        self._tqdm_service("stop")
        km_result = KmeansResult(centers, cls_assignments, i, has_converged)
        km_result.calculate_inertia(X)
        return km_result

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
        (cls_assignments,) = closest_indices(X, centers, self.distance_function, dims=(1,))

        return cls_assignments
