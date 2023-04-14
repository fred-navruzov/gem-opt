from typing import Callable

import networkx as nx


def construct_cost_graph(
    graph: nx.Graph, cost_fn: Callable, cost_field_name: str = "weight"
) -> nx.Graph:
    """
    Add cost data to existing graph as an edge attribute.

    .. note::
        Adding negative costs is ommitted.
        Thus, edge will not receive any key-value update

    Parameters
    ----------
    graph : nx.Graph
        Graph with edges to add costs to.
        Can be nx.DiGraph as well.
    cost_fn : Callable
        A callable object, that expects in- and out- nodes of an edge
        and produces cost for a given edge.
    cost_field_name : str, optional
        What attribute to save costs to, by default "weight".

    Returns
    -------
    nx.Graph
        Graph with added nodes
    """
    for vi, vj in graph.edges:
        cost = cost_fn(vi, vj)
        if cost >= 0:
            graph[vi][vj][cost_field_name] = cost
    return graph
