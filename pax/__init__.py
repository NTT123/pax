"""PAX package."""

from . import initializers, nets, nn, utils
from ._src.core import (
    Module,
    PaxFieldKind,
    apply_mp_policy,
    assert_structure_equal,
    enable_eval_mode,
    enable_train_mode,
    flatten_module,
    freeze_parameters,
    pure,
    module_and_value,
    select_parameters,
    select_states,
    unfreeze_parameters,
    update_parameters,
    update_states,
)
from ._src.core.rng import next_rng_key, seed_rng_key
from ._src.nn.dropout import dropout
from ._src.utils import build_update_fn, grad_parameters, scan

STATE = PaxFieldKind.STATE
PARAMETER = PaxFieldKind.PARAMETER
P = PaxFieldKind.PARAMETER
S = PaxFieldKind.STATE

__version__ = "0.4.1-dev0"

__all__ = [
    "apply_mp_policy",
    "assert_structure_equal",
    "build_update_fn",
    "dropout",
    "enable_eval_mode",
    "enable_train_mode",
    "flatten_module",
    "freeze_parameters",
    "grad_parameters",
    "initializers",
    "Module",
    "nets",
    "next_rng_key",
    "nn",
    "PaxFieldKind",
    "pure",
    "module_and_value",
    "scan",
    "seed_rng_key",
    "select_parameters",
    "select_states",
    "unfreeze_parameters",
    "update_parameters",
    "update_states",
    "utils",
]

try:
    del _src  # pylint: disable=undefined-variable
except NameError:
    pass
