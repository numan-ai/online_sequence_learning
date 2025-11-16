from typing import List, Tuple
import numpy as np

from src.common import inclusion, mix_two_vectors, permute, random_norm_sparse_vector, random_permutation

Array = np.ndarray


class Concept:
    """ Node of a graph """
    def __init__(self, name: str, dim: int, sparsity: float) -> None:
        self.name = name
        self.perm = random_permutation(dim)
        self.key = random_norm_sparse_vector(dim, sparsity)

    def __call__(self, q: Array) -> Array:
        return permute(q, self.perm) # type: ignore
    
    def __repr__(self) -> str:
        return f"Func(name={self.name})"


def route_threshold(current: Concept, path: Array, concepts: List[Concept], threshold: float) -> List[Tuple[str, float]]:
    """ Selects next concepts to fire based on the current path """
    res = sorted([
        (f.name, inclusion(path, f.key)) for f in concepts if f is not current and inclusion(path, f.key) >= threshold
    ], key=lambda x: x[1], reverse=True)
    return res


def fire_and_learn(concept: Concept, q: Array, r: Array, beta: float, gamma: float, eps: float) -> Tuple[Array, Array]:
    """ Fires a concept and updates its key to match the context a little more """
    concept.key = mix_two_vectors(concept.key, r, speed=beta, eps=eps)
    q_next = concept(q)
    r_next = gamma * r + q_next
    return q_next, r_next


def wait(r: Array, steps: int, gamma: float) -> Array:
    """ Decays the context """
    for _ in range(steps):
        r = gamma * r
    return r


def main() -> None: 
    dim = 2048
    sparsity = 0.01

    init_path_vector = np.eye(dim)[0]

    EnemyLeft  = Concept("EnemyLeft", dim=dim, sparsity=sparsity)
    EnemyRight  = Concept("EnemyRight", dim=dim, sparsity=sparsity)
    Pain = Concept("Pain", dim=dim, sparsity=sparsity)
    GoRight  = Concept("GoRight", dim=dim, sparsity=sparsity)
    GoLeft  = Concept("GoLeft", dim=dim, sparsity=sparsity)

    funcs = [EnemyLeft, EnemyRight, Pain, GoRight, GoLeft]

    # hyperparams
    firing_threshold   = 0.35     # routing threshold
    learning_rate  = 0.25     # key mixing rate
    context_decay = 0.8      # context decay
    eps   = 1e-3     # update gate for tiny context

    # training: learn both EnemyLeft->Pain->GoRight and EnemyRight->Pain->GoLeft
    context = np.zeros(dim)
    for _ in range(5):
        # Scenario EnemyLeft->Pain->GoRight
        path_vector = init_path_vector
        path_vector, context = fire_and_learn(EnemyLeft, path_vector, context, learning_rate, context_decay, eps)
        context = wait(context, steps=2, gamma=context_decay)  # decays context
        path_vector, context = fire_and_learn(Pain, path_vector, context, learning_rate, context_decay, eps)
        context = wait(context, steps=2, gamma=context_decay)
        path_vector, context = fire_and_learn(GoRight, path_vector, context, learning_rate, context_decay, eps)
        # long gap between scenarios
        context = wait(context, steps=300, gamma=context_decay)

        # Scenario EnemyRight->Pain->GoLeft
        path_vector = init_path_vector
        path_vector, context = fire_and_learn(EnemyRight, path_vector, context, learning_rate, context_decay, eps) 
        context = wait(context, steps=2, gamma=context_decay)
        path_vector, context = fire_and_learn(Pain, path_vector, context, learning_rate, context_decay, eps)
        context = wait(context, steps=2, gamma=context_decay)
        path_vector, context = fire_and_learn(GoLeft, path_vector, context, learning_rate, context_decay, eps)
        # long gap between scenarios
        context = wait(context, steps=300, gamma=context_decay)

    # ------- test 1: EnemyLeft->Pain->GoRight -------
    path_vector = init_path_vector
    path_vector = EnemyLeft(path_vector); print("After EnemyLeft:", route_threshold(EnemyLeft, path_vector, funcs, firing_threshold))
    path_vector = Pain(path_vector); print("After EnemyLeft->Pain:", route_threshold(Pain, path_vector, funcs, firing_threshold))

    # ------- test 2: EnemyRight->Pain->GoLeft -------
    path_vector = init_path_vector
    path_vector = EnemyRight(path_vector); print("After EnemyRight:", route_threshold(EnemyRight, path_vector, funcs, firing_threshold))
    path_vector = Pain(path_vector); print("After EnemyRight->Pain:", route_threshold(Pain, path_vector, funcs, firing_threshold))


# =================== demo: simultaneous learning ===================
if __name__ == "__main__":
    main()