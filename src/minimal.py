from typing import List, Tuple
import numpy as np

from src.common import inclusion, mix_two_vectors, permute, random_norm_sparse_vector, random_permutation

Array = np.ndarray


class Func:
    def __init__(self, name: str, dim: int, sparsity: float) -> None:
        self.name = name
        self.perm = random_permutation(dim)
        self.key = random_norm_sparse_vector(dim, sparsity)

    def __call__(self, q: Array) -> Array:
        return permute(q, self.perm) # type: ignore
    
    def __repr__(self) -> str:
        return f"Func(name={self.name})"


def route_threshold(current: Func, q: Array, funcs: List[Func], tau: float) -> List[Tuple[str, float]]:
    res = sorted([
        (f.name, inclusion(q, f.key)) for f in funcs if f is not current and inclusion(q, f.key) >= tau
    ], key=lambda x: x[1], reverse=True)
    return res


def fire_and_learn(f: Func, q: Array, r: Array, beta: float, gamma: float, eps: float) -> Tuple[Array, Array]:
    f.key = mix_two_vectors(f.key, r, speed=beta, eps=eps)
    q_next = f(q)
    r_next = gamma * r + q_next
    return q_next, r_next


def wait(r: Array, steps: int, gamma: float) -> Array:
    for _ in range(steps):
        r = gamma * r
    return r


def main() -> None: 
    dim = 2048
    sparsity = 0.01

    start_q = np.eye(dim)[0]

    X1  = Func("X1", dim=dim, sparsity=sparsity)
    X2  = Func("X2", dim=dim, sparsity=sparsity)
    MID = Func("MID", dim=dim, sparsity=sparsity)
    Y1  = Func("Y1", dim=dim, sparsity=sparsity)
    Y2  = Func("Y2", dim=dim, sparsity=sparsity)

    funcs = [X1, X2, MID, Y1, Y2]

    # hyperparams
    tau   = 0.35     # routing threshold
    beta  = 0.25     # key mixing rate (small)
    gamma = 0.8      # context decay
    eps   = 1e-3     # update gate for tiny context

    # training: learn both X1->MID->Y1 and X2->MID->Y2
    r = np.zeros(dim)
    for _ in range(5):
        # Scenario X1->MID->Y1
        q = start_q
        q, r = fire_and_learn(X1, q, r, beta, gamma, eps)
        r = wait(r, steps=2, gamma=gamma)
        q, r = fire_and_learn(MID, q, r, beta, gamma, eps)
        r = wait(r, steps=2, gamma=gamma)
        q, r = fire_and_learn(Y1, q, r, beta, gamma, eps)
        # long gap between scenarios
        r = wait(r, steps=300, gamma=gamma)

        # Scenario X2->MID->Y2
        q = start_q
        q, r = fire_and_learn(X2, q, r, beta, gamma, eps) 
        r = wait(r, steps=2, gamma=gamma)
        q, r = fire_and_learn(MID, q, r, beta, gamma, eps)
        r = wait(r, steps=2, gamma=gamma)
        q, r = fire_and_learn(Y2, q, r, beta, gamma, eps)
        # long gap between scenarios
        r = wait(r, steps=300, gamma=gamma)

    # ------- test 1: X1->MID->Y1 -------
    q = start_q
    q = X1(q); print("After X1:", route_threshold(X1, q, funcs, tau))
    q = MID(q); print("After X1->MID:", route_threshold(MID, q, funcs, tau))

    # ------- test 2: X2->MID->Y2 -------
    q = start_q
    q = X2(q); print("After X2:", route_threshold(X2, q, funcs, tau))
    q = MID(q); print("After X2->MID:", route_threshold(MID, q, funcs, tau))


# =================== demo: simultaneous learning ===================
if __name__ == "__main__":
    main()