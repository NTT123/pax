"""PAX package."""

from pax import graph, nets, nn, utils
from pax._src.core import (
    LazyModule,
    Module,
    ParameterModule,
    PaxKind,
    StateModule,
    apply_mp_policy,
    assert_structure_equal,
    enable_eval_mode,
    enable_train_mode,
    flatten_module,
    freeze_parameters,
    module_and_value,
    pure,
    select_parameters,
    select_states,
    unfreeze_parameters,
    update_parameters,
    update_states,
)
from pax._src.core.rng import next_rng_key, seed_rng_key
from pax._src.nn.dropout import dropout
from pax._src.utils import build_update_fn, grad_parameters, scan, value_and_grad

STATE = PaxKind.STATE
PARAMETER = PaxKind.PARAMETER
P = PaxKind.PARAMETER
S = PaxKind.STATE

__version__ = "0.4.2"

__all__ = (
    "apply_mp_policy",
    "assert_structure_equal",
    "build_update_fn",
    "dropout",
    "enable_eval_mode",
    "enable_train_mode",
    "flatten_module",
    "freeze_parameters",
    "grad_parameters",
    "graph",
    "LazyModule",
    "module_and_value",
    "Module",
    "nets",
    "next_rng_key",
    "nn",
    "ParameterModule",
    "PaxKind",
    "pure",
    "scan",
    "seed_rng_key",
    "select_parameters",
    "select_states",
    "StateModule",
    "unfreeze_parameters",
    "update_parameters",
    "update_states",
    "utils",
    "value_and_grad",
)

try:
    del _src  # pylint: disable=undefined-variable
except NameError:
    pass
