import networkx as nx
import pytest

from gem_opt.const import DEFAULT_PARAMS
from gem_opt.utils.graph_utils import construct_cost_graph


@pytest.fixture()
def graph():
    g = nx.DiGraph()
    g.add_edges_from([(1, 2), (2, 3), (3, 1)])
    return g


@pytest.mark.parametrize(
    "cost_fn, expected_weights",
    [
        (lambda u, v: 0, {1: {2: 0}, 2: {3: 0}, 3: {1: 0}}),
        (lambda u, v: u + v, {1: {2: 3}, 2: {3: 5}, 3: {1: 4}}),
        (lambda u, v: -(u + v), {1: {2: None}, 2: {3: None}, 3: {1: None}}),
    ],
)
def test_construct_cost_graph(graph, cost_fn, expected_weights):
    new_graph = construct_cost_graph(
        graph, cost_fn, cost_field_name=DEFAULT_PARAMS["COST_FIELD_NAME"]
    )

    for vi, vj in new_graph.edges:
        assert (
            new_graph[vi][vj].get(DEFAULT_PARAMS["COST_FIELD_NAME"])
            == expected_weights[vi][vj]
        )


@pytest.mark.parametrize(
    "cost_fn, expected_weights",
    [
        (lambda u, v: u * v, {1: {2: 2}, 2: {3: 6}, 3: {1: 3}}),
    ],
)
def test_construct_cost_graph_custom_attribute(graph, cost_fn, expected_weights):
    new_graph = construct_cost_graph(graph, cost_fn, cost_field_name="custom_weight")

    for vi, vj in new_graph.edges:
        assert new_graph[vi][vj]["custom_weight"] == expected_weights[vi][vj]
