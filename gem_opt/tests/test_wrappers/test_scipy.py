import networkx as nx
import numpy as np
import pytest
from scipy.optimize import linprog
from functools import partial
from gem_opt.utils import fake_cost_fn
from gem_opt.utils.graph_utils import construct_cost_graph
from gem_opt.const import DEFAULT_PARAMS
from gem_opt.wrappers import ScipyWrapper


def create_test_data():
    interaction_graph = nx.DiGraph()
    interaction_graph.add_nodes_from([0, 1, 2, 3])
    interaction_graph.add_weighted_edges_from(
        [
            (0, 0, 1),
            (1, 1, 1),
            (2, 2, 1),
            (3, 3, 1),
            (0, 1, 1),
            (1, 2, 2),
            (2, 3, 3),
            (3, 0, 4),
        ],
    )

    c_fn = partial(
        fake_cost_fn, use_const=0.25, sampler=None
    )  # constant costs of magnitude 0.25

    interaction_graph = construct_cost_graph(
        graph=interaction_graph,
        cost_fn=c_fn,
        cost_field_name=DEFAULT_PARAMS["COST_FIELD_NAME"],
    )

    demand = np.array([1, 2, 2, 2], dtype=DEFAULT_PARAMS["FLOAT_DTYPE"])
    supply = np.array([3, 3, 3, 3], dtype=DEFAULT_PARAMS["FLOAT_DTYPE"])

    return interaction_graph, demand, supply


@pytest.fixture
def wrapper():
    interaction_graph, demand, supply = create_test_data()
    return ScipyWrapper(
        interaction_graph=interaction_graph,
        demand=demand,
        supply=supply,
    )


def test_init(wrapper):
    assert isinstance(wrapper, ScipyWrapper)
    assert wrapper.n == 4


def test_generate_objective(wrapper):
    wrapper._generate_objective()
    expected_len = 12
    assert len(wrapper._obj_dict["obj_numerical"]) == expected_len
    assert len(wrapper._obj_dict["coeffs_symbolic"]) == expected_len
    assert len(wrapper._obj_dict["symbolic"]) == expected_len
    assert np.allclose(
        wrapper._obj_dict["obj_numerical"],
        np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0]),
    )


def test_generate_numeric_constraints(wrapper):
    wrapper._generate_numeric_constraints()
    constraints = wrapper._constraints["numerical"]

    assert constraints["lhs_ineq"].shape == (8, 12)
    assert constraints["lhs_eq"].shape == (4, 12)
    assert constraints["rhs_ineq"].shape == (8,)
    assert constraints["rhs_eq"].shape == (4,)


# def test_generate_symbolic_constraints(wrapper):
#     wrapper._generate_symbolic_constraints()
#     constraints = wrapper._constraints["symbolic"]

#     assert len(constraints["ineq"]) == 8
#     assert len(constraints["eq"]) == 4
#     assert len(constraints["bounds"]) == 12


# def test_optimize(wrapper):
#     opt_results = wrapper.optimize(optimizer=linprog)
#     assert opt_results.success == True
