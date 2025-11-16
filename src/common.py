import numpy as np


rng = np.random.default_rng(42)

Array = np.ndarray

def normalize(v: Array) -> Array:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def sim(q: Array, k: Array) -> float:
    nq, nk = np.linalg.norm(q), np.linalg.norm(k)
    return float(q @ k) if (nq == 0 or nk == 0) else float((q @ k) / (nq * nk))


def inclusion(v1: Array, v2: Array) -> float:
    """ Checks how much of v1 is included in v2 """
    return float(np.min([v1, v2], axis=0).sum() / v1.sum())


def random_rotation_matrix(dim: int) -> Array:
    # Create a random matrix
    A = rng.normal(size=(dim, dim))
    # QR decomposition
    Q, R = np.linalg.qr(A)
    # Ensure a proper rotation (determinant = 1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q # type: ignore


def random_basis(dim: int) -> Array:
    return np.eye(dim)[rng.integers(0, dim)] # type: ignore


def random_norm_vector(dim: int) -> Array:
    return normalize(np.abs(rng.standard_normal(dim)))


def random_norm_sparse_vector(dim: int, sparsity: float) -> Array:
    vec = np.abs(rng.standard_normal(dim))
    mask = rng.binomial(1, sparsity, dim)
    return normalize(vec * mask)


def permute(v: Array, perm: Array) -> Array:
    return v[perm] # type: ignore


def random_permutation(dim: int) -> Array:
    return rng.permutation(dim) # type: ignore


def mix_two_vectors(base: Array, update: Array, speed: float, eps: float) -> Array:
    """Small, gated mixing to avoid jumps; same rule for every node."""
    nr = np.linalg.norm(update)
    if nr < eps:
        return base * speed # type: ignore
    update_normalized = update / nr
    return normalize((1 - speed) * base + speed * update_normalized) # type: ignore
