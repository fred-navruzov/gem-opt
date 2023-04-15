## GEM Metric optimization

Quick-and-dirty Python wrapper(s) of [Graph-Based Equilibrium Metrics for
Dynamic Supply-Demand Systems with
Applications to Ride-sourcing Platforms](https://arxiv.org/pdf/2102.05805.pdf)

Contains wrappers to construct and solve correspondent [Linear Programming problem](https://en.wikipedia.org/wiki/Linear_programming):


![init_formulation](https://raw.githubusercontent.com/fred-navruzov/gem-opt/master/images/problem_formulation_init.jpg)

Thus, optimizing global diagnostic measures like supply-demand gap (SD) in aggregated fashion:
<br>selecting a weight measure wᵢ and aggregating the local gaps $m_{i} = \log{s_{i}} - \log{d_{i}}$, 

$A = \frac{\sum_{i=1}^{n} w_{i}\cdot m_{i}}{\sum_{i=1}^{n}w_{i}}$

where $w_{i}$ can represent business weighting, i.e. total demand in an entity $i$ 
<br>We can compute a demand-centric view and a supply-centric view of the marketplace by using the corresponding weights, as [proposed by Lyft](https://eng.lyft.com/quantifying-efficiency-in-ridesharing-marketplaces-affd53043db2):

$A_{d} = \frac{\sum_{i=1}^{n} d_{i}\cdot m_{i}}{\sum_{i=1}^{n}d_{i}}$

$A_{s} = \frac{\sum_{i=1}^{n} s_{i}\cdot m_{i}}{\sum_{i=1}^{n}s_{i}}$

---

## Currently supported wrappers:
- **[SciPy's linprog](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)**

    ---
    **NOTE**

    - Works best for small $N \in [2, 500]$ 
    - For now only **dense** arrays are supported (still, **sparse** format is accepted by SciPy and, thus, may be introduced here in the future), so beware of $O(N^2)$ memory usage!
    ---

    As we have L1-norm in an objective function, and many solvers (including SciPy's) do not support it, we can reformulate the problem.

    Reformulating an L1-norm using additional variables and constraints involves introducing new variables to represent the absolute values of the original variables, and then adding constraints to ensure the correct relationship between the new and original variables. Here's a simple example.

    Consider a linear programming problem with the following L1-norm objective function:

    ```
    minimize |x1| + |x2|

    Subject to:

    x1 + x2 >= 1
    ```

    We can reformulate the L1-norm by introducing two new sets of variables: y1 and y2, which represent the absolute values of x1 and x2, respectively. Then, we can rewrite the objective function and constraints as follows:
    ```
    minimize y1 + y2

    Subject to:

    x1 + x2 >= 1
    x1 - y1 <= 0
    -x1 - y1 <= 0
    x2 - y2 <= 0
    -x2 - y2 <= 0
    ```

    The new constraints ensure that y1 = |x1| and y2 = |x2|. The reformulated problem is now a linear programming problem without an explicit L1-norm in the objective function.

    You can use this approach to reformulate L1-norms in more complex optimization problems, as long as you introduce the appropriate variables and constraints to represent the absolute values of the original variables.

    So, omitting L1-norm in GEM objective will lead us to the next LP problem to solve:

    ![dummy_var_formulation](https://raw.githubusercontent.com/fred-navruzov/gem-opt/master/images/problem_formulation_dummy_vars.jpg)

---

## Example usage
Here is an example of how to decrease supply-demand gap (SD gap), using SciPy's wrapper
```python
import logging
import sys
from functools import partial

import networkx as nx
import numpy as np
from gem_opt.const import DEFAULT_PARAMS
from gem_opt.utils import create_fake_adjacency_matrix, fake_cost_fn
from gem_opt.utils.graph_utils import construct_cost_graph
from gem_opt.utils.metric_utils import calculate_optimized_supply, supply_demand_gap
from gem_opt.wrappers import ScipyWrapper


n = 5
adj_matrix = create_fake_adjacency_matrix(n_nodes=n)

demand = np.array([5, 2, 3, 10, 2])
supply = np.array([3, 8, 1, 6, 5])

logger.info(f"Example of GEM optimization and supply gap reduction for fake graph with {n} nodes")
logger.info(f"\nDemand:{demand.tolist()}\nSupply:{supply.tolist()}")

sampler = np.random.RandomState(42)
c_fn = partial(fake_cost_fn, use_const=0.25, sampler=sampler)  # constant costs of magnitude 0.25
graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
graph = construct_cost_graph(graph=graph, cost_fn=c_fn, cost_field_name=DEFAULT_PARAMS["COST_FIELD_NAME"])

# nx.draw_spring(cG);  uncomment to see viz

supply_gap_init = supply_demand_gap(supply=supply, demand=demand, aggregate=True, weight_matrix=demand)
logger.info(f"Initial supply gap (aggregated): {supply_gap_init:.4f}")
# Initial supply gap (aggregated): -0.2888 - aggregated under-supply

# create optimization wrapper for scipy's linprog
opt_wrapper = ScipyWrapper(
    interaction_graph=graph,
    demand=demand,
    supply=supply,
    # other params are set to defaults, see docstring for the details
)

# see what problem we are solving (objective with dummy variables and all the constraints)
logger.info(f"Objective   (symbolic):\n{opt_wrapper.objective_symbolic}")
# min f(y,S,C,λ=2.0) = 1.0*S1 + 1.0*S2 + 1.0*S3 + 1.0*S4 + 1.0*S5 
# + 0.5*y1_2 + 0.5*y2_1 + 0.5*y2_3 + 0.5*y3_2 + 0.5*y3_4 + 0.5*y4_3 + 0.5*y4_5 + 0.5*y5_4
logger.info("Constraints (symbolic):")
ineq_constraints = "\n".join(opt_wrapper.constraints_symbolic["ineq"])
logger.info(f"Inequalities:\n{ineq_constraints}")
# (-1.0)*S1 + (1.0)*y1_1 + (1.0)*y2_1 <= 5 ...
eq_constraints = "\n".join(opt_wrapper.constraints_symbolic["eq"])
logger.info(f"Equalities:\n{eq_constraints}")
# (1.0)*y1_1 + (1.0)*y1_2 = 3
logger.info(f"\nObjective   (numerical):\n{opt_wrapper.objective_numerical}")
logger.info("Constraints (numerical):")
# [1.  1.  1.  1.  1.  0.  0.5 0.5 0.  0.5 0.5 0.  0.5 0.5 0.  0.5 0.5 0. ]
ineq_constraints = "\n".join(
    [
        str(opt_wrapper.constraints_numerical["lhs_ineq"]),
        str(opt_wrapper.constraints_numerical["rhs_ineq"])
    ]
)
logger.info(f"Inequalities:\n{ineq_constraints}")
eq_constraints = "\n".join(
    [
        str(opt_wrapper.constraints_numerical["lhs_eq"]),
        str(opt_wrapper.constraints_numerical["rhs_eq"])
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
# Optimized supply: [ 5.  3.  3. 10.  2.]

supply_gap_opt = supply_demand_gap(supply=supply_optimized, demand=demand, aggregate=True, weight_matrix=demand)
logger.info(f"Optimized supply gap (aggregated): {supply_gap_opt:.4f}")
# Optimized supply gap (aggregated): 0.0369 - almost reach equilibrium
```