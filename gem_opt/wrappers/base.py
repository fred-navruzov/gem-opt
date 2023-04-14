from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Optional

import networkx as nx
import numpy as np

from gem_opt.const import DEFAULT_PARAMS, SYMBOLIC_PREFIXES, SymbolicOps


class BaseWrapper(ABC):
    """Abstract wrapper to GEM optimization task."""

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
        super().__init__()
        self.graph = interaction_graph
        self.demand = demand
        self.supply = supply
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype
        self.lambda_penalty = lambda_penalty
        self.cost_field = cost_field_name
        self.n = len(supply)
        self.ops = SymbolicOps
        self.prefixes = (
            symbolic_prefixes
            if symbolic_prefixes is not None
            else deepcopy(SYMBOLIC_PREFIXES)
        )
        self.opt_direction = opt_direction
        self.add_one_to_idx = add_one_to_idx
        self._notations = defaultdict(dict)
        self._obj_dict = {}
        self._constraints = dict.fromkeys(["numerical", "symbolic"])
        self._validate_input()
        self._populate_notations()

    def _validate_input(self):
        assert (
            self.demand.shape == self.supply.shape
        ), f"demand {self.demand.shape} and supply {self.supply.shape} are of unequal shape"
        assert (
            self.graph.number_of_nodes() == self.n
        ), f"check if your interation graph holds all the nodes: expected {self.n}."
        assert (
            self.demand.ndim == self.supply.ndim == 1
        ), f"only 1-dimensional arrays of supply/demand are supported for now. Got {self.demand.ndim, self.supply.ndim}"
        assert (
            self.lambda_penalty > 0
        ), f"Lambda should be positive float, got {self.lambda_penalty}"

    def _populate_notations(self):
        cost_prefix = self.prefixes.get("cost", "C")
        dummy_var_prefix = self.prefixes.get("dummy", "S")
        supply_prefix = self.prefixes.get("supply", "s")
        demand_prefix = self.prefixes.get("demand", "d")
        gamma_prefix = self.prefixes.get("gamma", "y")
        delimiter_prefix = self.prefixes.get("delimiter", "")

        self._notations["direction"] = self.opt_direction
        self._notations["dummy_l1"]["names"] = np.array(
            [
                f"{dummy_var_prefix}{delimiter_prefix}{i + self.add_one_to_idx}"
                for i in range(self.n)
            ]
        )
        self._notations["demand"]["names"] = np.array(
            [
                f"d{demand_prefix}{delimiter_prefix}{i + self.add_one_to_idx}"
                for i in range(self.n)
            ]
        )
        self._notations["demand"]["values"] = self.demand
        self._notations["supply"]["names"] = np.array(
            [
                f"{supply_prefix}{delimiter_prefix}{i + self.add_one_to_idx}"
                for i in range(self.n)
            ]
        )
        self._notations["supply"]["values"] = self.supply

        # TODO: refactor to deal with sparse formats later
        adj_matrix = nx.adjacency_matrix(
            self.graph, weight="weight", dtype=self.int_dtype
        ).T.todense()
        self._notations["cost"]["values"] = nx.adjacency_matrix(
            self.graph, weight=self.cost_field, dtype=self.float_dtype
        ).T.todense()

        self._notations["cost"]["mask"] = (
            (self._notations["cost"]["values"] > 0)
            # add diagonal to account for zero-costs self-loops in mask
            | np.eye(self.n).astype(bool)
        )
        self._notations["cost"]["names"] = np.array(
            [
                f"{cost_prefix}{delimiter_prefix}{i + self.add_one_to_idx}_{j + self.add_one_to_idx}"
                if self._notations["cost"]["mask"][i, j]
                else np.nan
                for i in range(self.n)
                for j in range(self.n)
            ]
        ).reshape((self.n, self.n))

        self._notations["gamma"]["names"] = np.array(
            [
                f"{gamma_prefix}{delimiter_prefix}{i + self.add_one_to_idx}_{j + self.add_one_to_idx}"
                if adj_matrix[i, j]
                else np.nan
                for i in range(self.n)
                for j in range(self.n)
            ]
        ).reshape((self.n, self.n))
        self._notations["gamma"]["values"] = adj_matrix
        self._notations["gamma"]["mask"] = self._notations["gamma"]["values"].astype(
            bool
        )

        self._notations["lambda"] = self.lambda_penalty

    @abstractmethod
    def _generate_objective(self):
        """Generate numerical (and symbolic) objectives."""

    @abstractmethod
    def _generate_constraints(self):
        """Generate numerical (and symbolic) constraints."""

    @property
    def constraints_numerical(self):
        """Get numerical constraints."""
        return self._constraints["numerical"]

    @property
    def constraints_symbolic(self):
        """Get symbolic constraints."""
        return self._constraints["symbolic"]

    @property
    def objective_numerical(self):
        """Get numerical objective (as vector of coeffs)."""
        return self._obj_dict["obj_numerical"]

    @property
    def objective_symbolic(self):
        """Get symbolic objective (as vector of coeffs)."""
        return self._obj_dict["repr"]

    @property
    def symbolic_order(self):
        """Get the order mapping for symbolic variables used."""
        return self._obj_dict["symbolic_order"]

    @property
    def dummy_variables(self):
        """Get the order mapping for symbolic variables used."""
        return self._notations["dummy_l1"]["names"]

    @abstractmethod
    def optimize(self, optimizer, optimizer_kwargs: dict):
        """Perform GEM optimization given internal obj and constraints."""
