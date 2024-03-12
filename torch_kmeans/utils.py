from dataclasses import dataclass, field

import torch


@dataclass(order=True)
class KmeansResult:
    """
    Result of K-means algorithm.

    Objects are compared solely on the calculated interia.

    Args:
        centers (torch.Tensor): cluster centers
        assignments (torch.Tensor): cluster assignments
        interia (float): sum of squared distances of samples to their closest cluster center
        n_iter (int): number of iterations run
        converged (bool): whether the algorithm converged
    """

    centers: torch.Tensor = field(compare=False)
    assignments: torch.Tensor = field(compare=False)
    n_iter: int = field(compare=False)
    converged: bool = field(compare=False)
    interia: float = field(compare=True, default=float("inf"))

    def calculate_inertia(self, X: torch.Tensor) -> None:
        """
        Calculate the inertia of the model.
        While this is potentially slow (loop), it is memory efficient.

        Args:
            X (torch.Tensor): input data
            y (torch.Tensor): cluster assignments
            centers (torch.Tensor): cluster centers

        Returns:
            torch.Tensor: inertia
        """
        inertia = 0
        n_clusters = self.centers.shape[0]
        for i in range(n_clusters):
            inertia += ((X[self.assignments == i] - self.centers[i]) ** 2).sum().item()
        self.inertia = inertia
