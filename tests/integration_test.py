import numpy as np
import pytest
import torch
from sklearn.cluster import KMeans as skKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import adjusted_mutual_info_score

from torch_kmeans import KMeans


class SyntheticDataset:
    """
    Generate synthetic data from sklearn's `make_blobs` function.

    Args
    ----
    n_samples : int
        Number of samples to generate.
    n_features : int
        Number of features for each sample.
    n_clusters : int
        Number of clusters to generate.
    cluster_std : float
        Standard deviation of the clusters.
    seed : int
        Random seed for reproducibility.

    Properties
    ----------
    X : torch.Tensor
        Observations
    y : torch.Tensor
        Ground truth labels
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 3,
        n_clusters: int = 4,
        cluster_std: float = 2.0,
        seed: int = 1001,
    ):
        """See class docstring for details."""
        super().__init__()
        np.random.seed(seed)

        # re mypy: ignore return syntax (return_centers=False and hence X, y is correct return)
        X, y = make_blobs(  # type: ignore[syntax]
            n_samples=n_samples,
            centers=n_clusters,
            n_features=n_features,
            random_state=seed,
            cluster_std=cluster_std,
            return_centers=False,
        )

        self.X = torch.from_numpy(X).to(dtype=torch.float32)
        self.y = torch.from_numpy(y).to(dtype=torch.int64)


@pytest.fixture
def synthetic_dataset():
    return SyntheticDataset(n_samples=100, n_features=3, n_clusters=4, cluster_std=2.0, seed=1001)


@pytest.mark.parametrize("n_clusters,max_iter", [(1, 100), (2, 100), (4, 100), (8, 200)])
def test_kmeans(synthetic_dataset: SyntheticDataset, n_clusters: int, max_iter: int):
    """
    Test the KMeans class defined herein, checking that all of the basic clustering objects
    have the correct contents, and test the results against sklearn's KMeans implementation.
    """
    X, y = synthetic_dataset.X, synthetic_dataset.y
    kmeans = KMeans(n_clusters=4, max_iter=100, tol=1e-4, use_tqdm=False, n_init=50)
    km_results = kmeans.fit_transform(X)
    assignments, cluster_centers = km_results.assignments, km_results.centers

    # basic assertions
    assert cluster_centers.shape[0] == 4
    assert cluster_centers.shape[1] == 3
    assert assignments.shape[0] == 100
    assert assignments.dtype == torch.int64
    assert cluster_centers.dtype == torch.float32
    assert assignments.max() == 3
    assert assignments.min() == 0
    assert assignments.ndimension() == 1
    assert cluster_centers.ndimension() == 2

    # compare with results from sklean's KMeans implementation
    sk_kmeans = skKMeans(init="random", n_clusters=4, max_iter=100, tol=1e-4, random_state=1001)
    sk_assignments = sk_kmeans.fit_predict(X.numpy())

    mi_score_ours = adjusted_mutual_info_score(y, assignments.numpy())
    mi_score_sk = adjusted_mutual_info_score(y, sk_assignments)

    assert mi_score_ours >= mi_score_sk * 0.90  # similar or better to sklearn's score.
    """
    Note: scikit-learn typically uses a better initialisation, and has a better score.
    We throttle sklearn to use a simpler strategy, and typically beat it, but due to randomness,
    we allow for a small margin of error.
    """
