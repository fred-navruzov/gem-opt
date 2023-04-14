import numpy as np
import pytest

from gem_opt.utils.metric_utils import calculate_optimized_supply, supply_demand_gap


# supply demand gap
@pytest.mark.parametrize(
    "supply, demand, expected_gap",
    [
        (np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([0.0, 0.0, 0.0])),
        (
            np.array([2, 4, 6]),
            np.array([1, 2, 3]),
            np.array([0.69314718, 0.69314718, 0.69314718]),
        ),
    ],
)
def test_supply_demand_gap_elementwise(supply, demand, expected_gap):
    gap = supply_demand_gap(supply=supply, demand=demand)
    np.testing.assert_array_almost_equal(gap, expected_gap)


@pytest.mark.parametrize(
    "supply, demand, weight_matrix, aggregate, expected_gap",
    [
        (np.array([1, 2, 3]), np.array([1, 2, 3]), None, True, 0.0),
        (np.array([2, 4, 6]), np.array([1, 2, 3]), None, True, 0.69314718),
        (
            np.array([2, 4, 6]),
            np.array([1, 2, 3]),
            np.array([0.5, 1, 1.5]),
            True,
            0.69314718,
        ),
    ],
)
def test_supply_demand_gap_aggregated(
    supply, demand, weight_matrix, aggregate, expected_gap
):
    gap = supply_demand_gap(
        supply=supply, demand=demand, aggregate=aggregate, weight_matrix=weight_matrix
    )
    np.testing.assert_almost_equal(gap, expected_gap)


def test_supply_demand_gap_invalid_inputs():
    with pytest.raises(ValueError):
        supply_demand_gap(supply=np.array([1, 2, 3]), demand=np.array([1, 2]))


# calculate optimized supply
def_symbolic_order = {"s_1": 0, "s_2": 1, "y_1": 2, "y_2": 3, "y_3": 4, "y_4": 5}
def_opt_results = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


@pytest.mark.parametrize(
    "opt_results, symbolic_order, idx_from, checksum, expected_supply",
    [
        (
            def_opt_results,
            def_symbolic_order,
            2,
            18.0,
            np.array([3.0, 4.0, 5.0, 6.0]),
        ),
        (
            def_opt_results,
            def_symbolic_order,
            2,
            None,
            np.array([3.0, 4.0, 5.0, 6.0]),
        ),
    ],
)
def test_calculate_optimized_supply(
    opt_results, symbolic_order, idx_from, checksum, expected_supply
):
    supply = calculate_optimized_supply(opt_results, symbolic_order, idx_from, checksum)
    np.testing.assert_array_equal(supply, expected_supply)


def test_calculate_optimized_supply_checksum_mismatch():
    with pytest.raises(AssertionError):
        calculate_optimized_supply(
            opt_results=def_opt_results,
            symbolic_order=def_symbolic_order,
            idx_from=2,
            checksum=21.0,  # expect 18
        )
