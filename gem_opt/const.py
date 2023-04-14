from enum import Enum

SYMBOLIC_PREFIXES = {
    "delimiter": "",
    "cost": "C",
    "dummy": "S",
    "supply": "s",
    "demand": "d",
    "gamma": "y",
}


class SymbolicOps(str, Enum):
    """Operations to use in equation's symbolic representation."""

    Ge = ">="
    Le = "<="
    Eq = "="
    Gt = ">"
    Ls = "<"


DEFAULT_PARAMS = {
    "LAMBDA_PENALTY": 2.0,  # lambda in GEM's objective fn
    "ADD_ONE_TO_IDX": True,  # whether symbolic indexes start from 1 (True) or 0 (False)
    "COST_FIELD_NAME": "cost",
    "INT_DTYPE": "int32",
    "FLOAT_DTYPE": "float32",
    "OPT_DIRECTION": "min",
}
