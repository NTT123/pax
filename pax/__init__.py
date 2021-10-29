"""PAX package."""

from pax import initializers, nets, nn, utils
from pax._src.core import (
    AutoModule,
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
from pax._src.utils import (
    apply_gradients,
    build_update_fn,
    grad_mod_val,
    grad_parameters,
    scan,
)

STATE = PaxKind.STATE
PARAMETER = PaxKind.PARAMETER
P = PaxKind.PARAMETER
S = PaxKind.STATE

__version__ = "0.4.2.dev2"

__all__ = (
    "apply_gradients",
    "apply_mp_policy",
    "assert_structure_equal",
    "AutoModule",
    "build_update_fn",
    "dropout",
    "enable_eval_mode",
    "enable_train_mode",
    "flatten_module",
    "freeze_parameters",
    "grad_mod_val",
    "grad_parameters",
    "initializers",
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
)

try:
    del _src  # pylint: disable=undefined-variable
except NameError:
    pass
