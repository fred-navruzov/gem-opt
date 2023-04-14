import logging
import sys
from functools import partial

import networkx as nx
import numpy as np
from const import DEFAULT_PARAMS
from utils import create_fake_adjacency_matrix, fake_cost_fn
from utils.graph_utils import construct_cost_graph
from utils.metric_utils import calculate_optimized_supply, supply_demand_gap
from wrappers.scipy import ScipyWrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

if __name__ == "__main__":
    n = 5
    adj_matrix = create_fake_adjacency_matrix(n_nodes=n)

    demand = np.array([5, 2, 3, 10, 2])
    supply = np.array([3, 8, 1, 6, 5])

    logger.info(
        f"Example of GEM optimization and supply gap reduction for fake graph with {n} nodes"
    )
    logger.info(f"\nDemand:{demand.tolist()}\nSupply:{supply.tolist()}")
    sampler = np.random.RandomState(42)
    c_fn = partial(
        fake_cost_fn, use_const=0.25, sampler=sampler
    )  # constant costs of magnitude 0.25
    graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    graph = construct_cost_graph(
        graph=graph, cost_fn=c_fn, cost_field_name=DEFAULT_PARAMS["COST_FIELD_NAME"]
    )

    # nx.draw_spring(cG);  uncomment to see viz

    supply_gap_init = supply_demand_gap(
        supply=supply, demand=demand, aggregate=True, weight_matrix=demand
    )
    logger.info(f"Initial supply gap (aggregated): {supply_gap_init:.4f}")

    # create optimization wrapper for scipy's linprog
    opt_wrapper = ScipyWrapper(
        interaction_graph=graph,
        demand=demand,
        supply=supply,
        # other params are set to defaults, see docstring for the details
    )

    # see what problem we are solving (objective with dummy variables and all the constraints)
    logger.info(f"Objective   (symbolic):\n{opt_wrapper.objective_symbolic}")
    logger.info("Constraints (symbolic):")
    ineq_constraints = "\n".join(opt_wrapper.constraints_symbolic["ineq"])
    logger.info(f"Inequalities:\n{ineq_constraints}")
    eq_constraints = "\n".join(opt_wrapper.constraints_symbolic["eq"])
    logger.info(f"Equalities:\n{eq_constraints}")

    logger.info(f"\nObjective   (numerical):\n{opt_wrapper.objective_numerical}")
    logger.info("Constraints (numerical):")
    ineq_constraints = "\n".join(
        [
            str(opt_wrapper.constraints_numerical["lhs_ineq"]),
            str(opt_wrapper.constraints_numerical["rhs_ineq"]),
        ]
    )
    logger.info(f"Inequalities:\n{ineq_constraints}")
    eq_constraints = "\n".join(
        [
            str(opt_wrapper.constraints_numerical["lhs_eq"]),
            str(opt_wrapper.constraints_numerical["rhs_eq"]),
        ]
    )
    logger.info(f"Equalities:\n{eq_constraints}")

    # perform optimization
    # default args used here, see docstring of correspondent optimizer and pass desired set as `optimizer_kwargs` argument
    opt_results = opt_wrapper.optimize()

    # recalculate supply
    supply_optimized = calculate_optimized_supply(
        opt_results=opt_results.x,
        symbolic_order=opt_wrapper.symbolic_order,
        idx_from=len(opt_wrapper.dummy_variables),
    )
    logger.info(f"Optimized supply: {supply_optimized}")

    supply_gap_opt = supply_demand_gap(
        supply=supply_optimized, demand=demand, aggregate=True, weight_matrix=demand
    )
    logger.info(f"Optimized supply gap (aggregated): {supply_gap_opt:.4f}")
