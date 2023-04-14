from typing import Callable, Optional

import networkx as nx
import numpy as np
from scipy.optimize import OptimizeResult, linprog

from gem_opt.const import DEFAULT_PARAMS
from gem_opt.wrappers.base import BaseWrapper


class ScipyWrapper(BaseWrapper):
    """Scipy's `optimize.linprog` wrapper to GEM optimization task."""

    def __init__(
        self,
        *,
        interaction_graph: nx.Graph,
        demand: np.ndarray,
        supply: np.ndarray,
        cost_field_name: str = DEFAULT_PARAMS["COST_FIELD_NAME"],
        float_dtype: str = DEFAULT_PARAMS["FLOAT_DTYPE"],
        int_dtype: str = DEFAULT_PARAMS["INT_DTYPE"],
        lambda_penalty: float = DEFAULT_PARAMS["LAMBDA_PENALTY"],
        symbolic_prefixes: Optional[dict] = None,
        opt_direction: str = DEFAULT_PARAMS["OPT_DIRECTION"],
        add_one_to_idx: bool = DEFAULT_PARAMS["ADD_ONE_TO_IDX"],
    ) -> None:
        """
        Initialize wrapper.

        Parameters
        ----------
        interaction_graph : nx.Graph
            NetworkX's graph (or DiGraph if assymmetric interaction)
            to represent interactions between elements of demand/supply arrays.
        demand : np.ndarray
            1-D array of element-wise demand.
            The ordering of array should match
            the order of nodes in `interaction_graph`.
        supply : np.ndarray
            1-D array of element-wise supply.
            The ordering of array should match
            the order of nodes in `interaction_graph`
            and `demand` ordering.
        cost_field_name : str, optional
            What attribute to attach costs to in `interaction_graph`,
            by default DEFAULT_PARAMS["COST_FIELD_NAME"]
        float_dtype : str, optional
            What floats to use in internal calculations,
            by default DEFAULT_PARAMS["FLOAT_DTYPE"]
        int_dtype : str, optional
            What ints to use in internal calculations,
            by default DEFAULT_PARAMS["INT_DTYPE"]
        lambda_penalty : float, optional
            Penalty term in GEM's objective,
            by default DEFAULT_PARAMS["LAMBDA_PENALTY"].
            Should be positive float.
        symbolic_prefixes: dict, optional
            What prefixes to use (i.e. for costs, gammas)
            in symbolic representation of optimization task.
            by default SYMBOLIC_PREFIXES
        opt_direction: str, optional
            What optimization direction to use
            in symbolic representations of cost function.
        add_one_to_idx: bool, optional
            Whether to start symbolic indexing
            from 1 (True) or 0 (False).
            by default DEFAULT_PARAMS["ADD_ONE_TO_IDX"]
        """
        super().__init__(
            interaction_graph=interaction_graph,
            demand=demand,
            supply=supply,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            lambda_penalty=lambda_penalty,
            cost_field_name=cost_field_name,
            symbolic_prefixes=symbolic_prefixes,
            opt_direction=opt_direction,
            add_one_to_idx=add_one_to_idx,
        )
        # TODO: later, add explicit params here to override defaults
        self._generate_objective()
        self._generate_constraints()

    def _generate_objective(
        self,
        *,
        use_symbolic_cost: bool = False,
        symbolic_round_to: int = 1,
        show_zeroes: bool = False,
    ) -> None:
        """
        Generate numerical (and symbolic) objectives.

        Parameters
        ----------
        use_symbolic_cost : bool, optional
            Whether to replace particular cost constants
            (i.e `0.5` for C_11) by symbolic repr (i.e. `C_11`),
            by default False
        symbolic_round_to : int, optional
            What precision to use for float coefficients
            in symbolic representation, by default 1.
        show_zeroes : bool, optional
            Whether to show variables with zero coefficients
            in symbolic representation, by default False.

        Returns
        -------
        None
        """
        cost = self._notations["cost"]
        s_n = len(self._notations["dummy_l1"]["names"])
        gamma = self._notations["gamma"]
        cost = self._notations["cost"]

        self._obj_dict["obj_numerical"] = np.hstack(
            [
                np.ones(s_n),
                cost["values"][cost["mask"]] * self._notations["lambda"],
            ]
        )
        self._obj_dict["coeffs_symbolic"] = np.hstack(
            [
                np.ones(s_n).round(symbolic_round_to).astype(str),
                np.core.defchararray.add(
                    "λ*",
                    cost["names"][cost["mask"]],
                ),
            ]
        )
        self._obj_dict["symbolic"] = np.hstack(
            [self._notations["dummy_l1"]["names"], gamma["names"][gamma["mask"]]]
        )

        self._obj_dict["symbolic_order"] = {
            k: i for i, k in enumerate(self._obj_dict["symbolic"])
        }

        self._obj_dict[
            "repr"
        ] = f'{self.opt_direction} f(y,S,C,λ={self._notations["lambda"]}) = '
        if use_symbolic_cost:
            self._obj_dict["repr"] += " + ".join(
                f"{c}*{s}"
                for c, s in zip(
                    self._obj_dict["coeffs_symbolic"], self._obj_dict["symbolic"]
                )
            )
        else:
            self._obj_dict["repr"] += " + ".join(
                f"{round(c, symbolic_round_to) if symbolic_round_to > 0 else int(c)}*{s}"
                for c, s in zip(
                    self._obj_dict["obj_numerical"], self._obj_dict["symbolic"]
                )
                if (True if show_zeroes else c > 0)
            )

    def _generate_numeric_constraints(
        self,
    ) -> None:
        """Generate matrix of numerical constraints (equalities, inqeualities) for GEM task."""
        constraints = {}

        n = self.n
        m = len(self._obj_dict["symbolic"])
        constraints["lhs_ineq"] = np.zeros((2 * n, m), dtype=self.float_dtype)

        # fill-in inequalities, all set up to Ax <= b form
        for i, mask in enumerate(self._notations["gamma"]["mask"].T):
            symbols = self._notations["gamma"]["names"][..., i]
            indexes = [self._obj_dict["symbolic_order"][e] for e in symbols[mask]]

            # add left part, primary Ax <= b (demand)
            constraints["lhs_ineq"][i, indexes] = 1  # add gammas' coeffs
            constraints["lhs_ineq"][i, i] = -1  # add -1 to S_i

            # add left part, inverted Ax >= b as -Ax <= -b (negative demand)
            constraints["lhs_ineq"][n + i, indexes] = -1  # add neg gammas' coeffs
            constraints["lhs_ineq"][n + i, i] = -1  # add -1 to S_i

        # add right part
        constraints["rhs_ineq"] = np.hstack(
            [self._notations["demand"]["values"], -self._notations["demand"]["values"]]
        )

        # fill-in equalities, Ax == b
        constraints["lhs_eq"] = np.zeros(shape=(n, m), dtype=self.float_dtype)
        for i, mask in enumerate(self._notations["gamma"]["mask"]):
            symbols = self._notations["gamma"]["names"][i, ...]
            indexes = [self._obj_dict["symbolic_order"][e] for e in symbols[mask]]
            constraints["lhs_eq"][i, indexes] = 1

        constraints["rhs_eq"] = self._notations["supply"]["values"].copy()
        constraints["bounds"] = tuple(
            (0, np.inf) for i in range(len(self._obj_dict["obj_numerical"]))
        )
        self._constraints["numerical"] = constraints

    def _generate_symbolic_constraints(
        self,
        symbolic_round_to: int = 2,
    ) -> dict:
        """Generate matrix of symbolic constraints (equalities, inqeualities) for GEM task."""
        sym_dict = {}
        symbolic_mapping_inv = {
            v: k for k, v in self._obj_dict["symbolic_order"].items()
        }

        # add Ax <= b and Ax = b
        prefixes = {"ineq": self.ops.Le.value, "eq": self.ops.Eq.value}
        for prefix, op in prefixes.items():
            sym_dict[prefix] = []
            for left, right in zip(
                self._constraints["numerical"][f"lhs_{prefix}"],
                self._constraints["numerical"][f"rhs_{prefix}"],
            ):
                mapping = [
                    symbolic_mapping_inv[i] for i, e in enumerate(left) if e != 0
                ]
                left_filtered = [e for e in left if e != 0]
                str_repr = (
                    " + ".join(
                        [
                            f"({round(e, symbolic_round_to) if symbolic_round_to else int(e)})*{m}"
                            for e, m in zip(left_filtered, mapping)
                        ]
                    )
                    + f" {op} {right}"
                )
                sym_dict[prefix].append(str_repr)
        # add individual var constraints
        sym_dict["bounds"] = [
            f"{lb} <= {symbolic_mapping_inv[i]} < {ub}"
            for i, (lb, ub) in enumerate(self._constraints["numerical"]["bounds"])
        ]
        self._constraints["symbolic"] = sym_dict

    def _generate_constraints(self, symbolic_round_to: int = 2):
        """Generate numerical and symbolic constraints."""
        self._generate_numeric_constraints()
        self._generate_symbolic_constraints(symbolic_round_to=symbolic_round_to)

    def optimize(
        self, optimizer: Callable = linprog, optimizer_kwargs: dict = None
    ) -> OptimizeResult:
        """
        Perform GEM optimization given internal obj and constraints.

        By default use SciPy's `optimize.linprog` method.
        """
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            optimizer_kwargs["c"] = self._obj_dict["obj_numerical"]
            optimizer_kwargs["A_ub"] = self.constraints_numerical["lhs_ineq"]
            optimizer_kwargs["b_ub"] = self.constraints_numerical["rhs_ineq"]
            optimizer_kwargs["A_eq"] = self.constraints_numerical["lhs_eq"]
            optimizer_kwargs["b_eq"] = self.constraints_numerical["rhs_eq"]
            optimizer_kwargs["bounds"] = self.constraints_numerical["bounds"]

        opt_results = optimizer(**optimizer_kwargs)
        return opt_results
