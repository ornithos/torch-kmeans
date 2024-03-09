# Torch K-means

A dead simple implementation of k-means clustering that uses PyTorch. In contrast to `scikit-learn`'s `KMeans` class, we can take advantage of PyTorch tooling such as optimizers, backprop and GPU capabilities.

This projected started life from [`subhadarship/kmeans_pytorch`](https://github.com/subhadarship/kmeans_pytorch) which appears to be largely abandoned. For example, there's a PR which was opened almost a year ago with no interaction from the maintainer. Given I found a number of engineering and computational problems, I decided to perform a do-over. The most crucial problem was the poor implementation of the pairwise distance functions, requiring (for matrices of size `n x d` and `m x d`) an intermediate matrix of size `n x m x d` which is infeasible for large matrices.

The following are immediate goals for this project:

* Creating a `scikit-learn` style API
* Implementing tests
* Implementing CI
* Adding new features:
  - stochastic gradient descent
  - multi-GPU training

For example, on a 3090 with 24GB RAM I am unable to cluster a 10^6 x 1024 matrix into 1024 clusters. I intend to fix this.
