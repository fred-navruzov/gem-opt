"""Metrics module. Holds SD gap metrics and similar."""

from typing import Optional, Union

import numpy as np
import pandas as pd

from gem_opt.const import DEFAULT_PARAMS


def supply_demand_gap(
    *,
    supply: np.ndarray,
    demand: np.ndarray,
    aggregate: bool = False,
    weight_matrix: Optional[np.ndarray] = None,
) -> Union[np.ndarray, float]:
    r"""
    Calculate supply-demand (SD) gap metric :math: `M`.

    Given vectors of demand (d) and supply (s),
    SD gap is defined as :math:`m_{i} = \log{s_{i}} - \log{d_{i}}`
    Optionally computes weighted scalar measure,
    given `weight_matrix` :math: `W`:
    .. math::
    M = \frac{\sum_{i=1}^{n} w_{i} \cdot m_{i}}{\sum_{i=1}^{n} w_{i}}

    Parameters
    ----------
    supply : np.ndarray
        Element-wise supply
    demand : np.ndarray
        Element-wise demand
    aggregate : bool, optional
        Whether to aggregate element-wise metric
        to a single scalar, by default `False`
    weight_matrix : np.ndarray, optional
        What element-wise weights to use
        for aggregation, by default `None`

    Returns
    -------
    gap : Union[np.ndarray, float]
        Gap measure of shape `supply.shape`
        or scalar if `weight_matrix` is present
        and `aggregate` is set to `True`
    """
    gap = np.log(supply / demand)
    if aggregate:
        weight_matrix = np.ones_like(demand) if weight_matrix is None else weight_matrix
        gap = np.nansum(gap * weight_matrix) / np.sum(
            weight_matrix[~np.isnan(weight_matrix)],
        )

    return gap


def calculate_optimized_supply(
    opt_results: np.ndarray,
    symbolic_order: dict[str, int],
    idx_from: int,
    checksum: float = None,
) -> np.ndarray:
    """
    Restore back supply, given `opt_results` from optimized plan Gamma.

    Parameters
    ----------
    opt_results : np.ndarray
        Vector of optimized results (dummy variables + gamma_i_j)
    symbolic_order : dict[str, int]
        Order of symbolic variables,
        {var_name: opt_results_index}.
    idx_from : int
        From what index to start (to omit all dummy variables, if any)
    checksum : float, optional
        Compare restored supply with initial one (sum)
        to assure equality of supply,
        by default None

    Returns
    -------
    np.ndarray
        Optimized supply vector,
        of the same ordering as initial
    """
    # TODO: mb pass BaseWrapper here to access its attributes instead of `idx_from`, `symbolic_order`

    mapping = {
        k: opt_results[i] for i, k in enumerate(symbolic_order.keys()) if i >= idx_from
    }
    agg_df = pd.Series(mapping, name="supply").reset_index()
    agg_df["pos"] = (
        agg_df["index"].str.split("_").str[-1].astype(DEFAULT_PARAMS["INT_DTYPE"])
    )
    supply = agg_df.groupby("pos")["supply"].sum().sort_index().to_numpy()

    if checksum is not None:
        supply_amt = supply.sum()
        assert np.isclose(
            supply_amt, checksum
        ), f"supply amt mismatches:\ngot {supply_amt}\nexpected: {checksum}"

    return supply
