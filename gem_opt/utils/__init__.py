from collections.abc import Hashable

import numpy as np
from numpy.random import RandomState


def fake_cost_fn(
    vi: Hashable, vj: Hashable, sampler: RandomState, use_const: float = None
) -> float:
    """
    Mock-up `cost_fn` function to produce costs of moving from entiry `Vi` to entity `Vj`.

    Parameters
    ----------
    vi : Hashable
        Entiry to move from
    vj : Hashable
        Entiry to move to
    sampler : RandomState
        What to use for generating random costs
    use_const : float, optional
        Whether to use constant costs instead of random samples,
        by default None

    Returns
    -------
    float
        Generated cost
    """
    if vi == vj:  # assume there is no cost to move within the entity
        return 0
    return use_const if use_const is not None else sampler.rand()


def create_fake_adjacency_matrix(n_nodes: int = 5, dtype: str = "int32") -> np.ndarray:
    """
    Create fake adjacency matrix for graph with `n_nodes`.

    Assumes adjacency only to preceding and the following node,
    as well as self-loop by default.

    Parameters
    ----------
    n_nodes : int, optional
        How many nodes to create, by default 5
    dtype : str, optional
        What dtype to use, by default "int32"

    Returns
    -------
    adj_arr: np.ndarray
        Adjacency matrix as numpy array
    """
    adj_arr = np.zeros((n_nodes, n_nodes))
    # add adjacency only to preceding and the following node, as well as self-loop by default
    for k in [-1, 0, 1]:
        adj_arr += np.eye(n_nodes, k=k, dtype=dtype)
    return adj_arr
