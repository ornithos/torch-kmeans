# Torch K-means

A dead simple implementation of k-means clustering that uses PyTorch. In contrast to `scikit-learn`'s `KMeans` class, we can take advantage of fast CUDA kernels and PyTorch tooling such as distributed dataloaders, and multi-GPU capabilities.

### Background
This projected started life from [`subhadarship/kmeans_pytorch`](https://github.com/subhadarship/kmeans_pytorch) which appears to be largely abandoned. This fixes memory problems for large matrices when computing pairwise distance functions, which used to require (for matrices of size `n x d` and `m x d`) an intermediate matrix of size `n x m x d`.

We also take advantage of `torch_scatter`, which provides a very fast and efficient way to compute the cluster means.

### Basic Usage
The package uses a similar interface to `scikit-learn`; see the following example.

```python
import torch
import torch_kmeans

X = torch.randn(200,3)

km = torch_kmeans.KMeans(distance="cosine", n_clusters=4)

result = km.fit_transform(X, seed=101)
assignments, centers = result.assignments, result.centers
```

The `result` object holds the cluster centers, assignments of each datapoint, and the [inertia](https://scikit-learn.org/stable/modules/clustering.html#k-means) of the resulting clustering.

### Goals
The following are immediate goals for this project:

* Fixing the `pip` dependency bug in GitHub Actions.
  * The `torch_scatter` package cannot find `torch` during the build step, even though `torch` is explicitly installed first. It's perhaps related to [pytorch_sparse/issues/#156](https://github.com/rusty1s/pytorch_sparse/issues/156), although this is yet to be determined. I cannot reproduce this locally.
* Adding new features:
  - kmeans++ initialization
  - callbacks/hooks to extract the evolving cluster distribution
  - benchmarks
  - multi-GPU training
* Create more reliable tests (avoid rng optimisation)
  - Comparing to sklearn's implementation requires use of kmeans++ within this project.
