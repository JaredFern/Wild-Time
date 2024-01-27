import numpy as np
import torch

from scipy import optimize
from scipy.sparse.linalg import LinearOperator, eigsh
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def compute_hvp(network, loss_fn, dataloader, vector, max_samples=1000, device="cuda"):
    """Compute a Hessian-vector product."""
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataloader.dataset)
    hvp = torch.zeros(p, dtype=torch.float, device=device)
    vector = vector.to(device)
    num_samples = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        loss = loss_fn(network(X), y) / n

        grads = torch.autograd.grad(
            loss, inputs=network.parameters(), create_graph=True
        )
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [
            g.contiguous()
            for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)
        ]
        hvp += parameters_to_vector(grads)
        num_samples += X.shape[0]
        if num_samples >= max_samples:
            break
    return hvp


def lanczos(matrix_vector, dim, neigs, device="cuda"):
    """Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products)."""

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).to(device)
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return (
        torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(),
        torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float(),
    )


def get_hessian_eigenvalues(
    network,
    loss_fn,
    dataloader,
    neigs=6,
    device="cuda",
    max_samples=1000,
    physical_batch_size=1000,
):
    """Compute the leading Hessian eigenvalues via Lanczos power iterations.Adapted from: https://github.com/locuslab/edge-of-stability/blob/github/src/utilities.py"""
    hvp_delta = (
        lambda delta: compute_hvp(
            network, loss_fn, dataloader, delta, max_samples=max_samples, device=device
        )
        .detach()
        .cpu()
    )
    nparams = len(parameters_to_vector((network.parameters())).cpu())
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs, device=device)
    torch.cuda.empty_cache()
    return evals


def sharpness(model, criterion_fn, A=1, epsilon=1e-3, p=0, bounds=None):
    """Computes sharpness metric according to https://arxiv.org/abs/1609.04836.

    Args:
        model: Model on which to compute sharpness
        criterion_fn: Function that takes in a model and returns the loss
            value and gradients on the appropriate data that will be used in
            the loss maximization done in the sharpness calculation.
        A: Projection matrix that defines the subspace in which the loss
            maximization will be done. If A=1, no projection will be done.
        epsilon: Defines the size of the neighborhood that will be used in the
            loss maximization.
        p: The dimension of the random projection subspace in which maximization
            will be done. If 0, assumed to be the full parameter space.
    """
    run_fn = create_run_model(model, A, criterion_fn)
    if bounds is None:
        bounds = compute_bounds(model, A, epsilon)
    dim = flatten_parameters(model).shape[0] if p == 0 else p

    # Find the maximum loss in the neighborhood of the minima
    y = optimize.minimize(
        lambda x: run_fn(x),
        np.zeros(dim),
        method="L-BFGS-B",
        bounds=bounds,
        jac=True,
        options={"maxiter": 10},
    ).x.astype(np.float32)

    model_copy = copy.deepcopy(model)
    if A is 1:
        flat_diffs = y
    else:
        flat_diffs = A @ y
    apply_diffs(model_copy, flat_diffs)
    maximum = criterion_fn(model_copy)["loss"]
    loss_value = criterion_fn(model)["loss"]
    sharpness = 100 * (maximum - loss_value) / (1 + loss_value)
    return sharpness